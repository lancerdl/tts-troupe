"""
OpenAI-compatible FastAPI speech synthesis server.

Model-agnostic: synthesis is delegated to model adapters discovered under
models/ (declared via model.yaml manifests) through the ModelManager. The
server owns routing (clone voices -> presets -> persona description -> default
voice), request validation, and audio encoding.
"""
import asyncio
import atexit
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

# Ensure repo root is on sys.path so `models` (and sibling model repos) resolve.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from .audio_utils import (  # noqa: E402
    extract_instructions,
    strip_xml_tags,
    strip_bracket_tags,
    to_mp3_bytes,
    to_pcm16,
    to_wav_bytes,
)
from .config import ServerConfig  # noqa: E402
from .log_utils import setup_server_logging  # noqa: E402
from .voice_library import export_voice_list, scan_voice_library  # noqa: E402
from framework.adapter import (  # noqa: E402
    AudioResult,
    ModelNotInitializedError,
    ModelNotReadyError,
    ModelSettings,
    NoModelForCapabilityError,
    SynthesisError,
)
from framework.model_manager import ModelManager  # noqa: E402

setup_server_logging()
logger = logging.getLogger("tts_troupe")

# Global instances (populated by main() / lifespan)
config: Optional[ServerConfig] = None
model_manager: Optional[ModelManager] = None
discovered_voices: Dict[str, Dict[str, str]] = {}
available_presets: List[str] = []
presets_by_model: Dict[str, List[str]] = {}


class SpeechRequest(BaseModel):
    """OpenAI /v1/audio/speech request schema (subset used by this server)."""
    model: str = "voice-select"
    input: str
    voice: str = "alloy"
    response_format: Optional[str] = None
    speed: float = 1.0


# [PART2]

def _preset_voices() -> Dict[str, List[str]]:
    """Manifest-declared preset voices per model (manager capabilities)."""
    if model_manager is not None:
        return model_manager.presets_by_model()
    return {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global discovered_voices, available_presets, presets_by_model

    # Startup sequence
    logger.info("=== Starting TTS Troupe ===")

    # Probe CUDA health immediately upon startup
    cuda_ready = model_manager.check_ready(force=True)
    if not cuda_ready:
        logger.warning("[STARTUP] Initial CUDA health probe indicated CUDA is unavailable or unresponsive!")
    else:
        logger.info("[STARTUP] CUDA subsystem confirmed responsive.")

    # Register adapters (VRAM setup stays lazy until first synthesis).
    model_manager.load()

    # Best-effort transcription pass: archive .wav files lacking both a
    # .reference.txt transcript and a .txt fallback get transcribed via
    # faster-whisper (CUDA -> CPU fallback), persisting .reference.txt.
    # Never fatal to startup; a no-op when the archive is fully transcribed.
    try:
        from .transcribe_voices import transcribe_voice_library

        transcribed = transcribe_voice_library(config.voices_dir, export_csv=False)
        if transcribed:
            logger.info(
                "[STARTUP] Transcribed %d archive voice(s) missing references "
                "via faster-whisper.", transcribed,
            )
    except Exception as e:
        logger.warning(
            "[STARTUP] Archive transcription pass failed (non-fatal): %s", e)

    # Discover clone voices and manifest-declared preset voices (per model)
    discovered_voices = scan_voice_library(config.voices_dir)
    presets_by_model = _preset_voices()
    available_presets = list(dict.fromkeys(
        p for voices in presets_by_model.values() for p in voices))

    # Export the voice lists to the server root:
    #   voices.txt                 - cloneable voices in the archive
    #   <model_name>_voices.txt    - per model: presets + clone archive
    # (whether or not prompts have been converted into ICL tensors yet)
    clone_names = sorted(discovered_voices.keys())
    export_voice_list(os.path.join(REPO_ROOT, "voices.txt"), clone_names)
    for model_name, model_presets in presets_by_model.items():
        export_voice_list(
            os.path.join(REPO_ROOT, f"{model_name}_voices.txt"),
            list(model_presets) + clone_names,
        )

    logger.info("Server ready. Listening at http://%s:%d", config.host, config.port)
    logger.info("OpenAI Provider Endpoint: http://%s:%d/v1/audio/speech", config.host, config.port)
    yield

    # Shutdown sequence
    logger.info("=== Shutting down server ===")
    model_manager.shutdown()


app = FastAPI(title="TTS Troupe", lifespan=lifespan)

_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
async def index():
    """Manual test console (self-contained single-file front-end)."""
    return FileResponse(_STATIC_DIR / "test_console.html", media_type="text/html")


@app.get("/health")
async def health():
    # Force an immediate CUDA health check on every /health request
    cuda_healthy = model_manager.check_ready(force=True)
    return {
        "status": "ok" if cuda_healthy else "degraded",
        "cuda_healthy": cuda_healthy,
        "presets": available_presets,
        "cloned_voices": list(discovered_voices.keys()),
        "voices_dir": config.voices_dir,
        "caches": model_manager.cache_status(),
    }


@app.get("/v1/speakers")
async def speakers(model: Optional[str] = None):
    """Complete built-in speaker catalogue from the loaded models.

    With `model`, only that adapter is queried (unknown -> 404), so a
    catalogue fetch can never load any other model's checkpoint. May load
    the owning checkpoint on first call - the console exposes this behind
    an explicit button so the VRAM cost is a deliberate choice."""
    if model is not None:
        resolved = model_manager.resolve_model(model) if model_manager else None
        if resolved is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown model {model!r}. Available: {', '.join(model_manager.model_names())}")
        return {"catalogue": model_manager.speaker_catalogue(model=resolved)}
    return {"catalogue": model_manager.speaker_catalogue()}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Browser tab icon (also quiets the automatic favicon 404)."""
    return FileResponse(_STATIC_DIR / "favicon.ico", media_type="image/x-icon")


@app.get("/v1/voices")
async def list_voices(model: Optional[str] = None):
    """Voice list. With `model`: that model's custom voices plus the clone
    archive. Without: the flat aggregate plus the per-model preset map."""
    if model is not None:
        resolved = model_manager.resolve_model(model) if model_manager else None
        if resolved is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model {model!r}. Available: {', '.join(model_manager.model_names())}")
        clones = [
            {
                "id": name,
                "type": "clone",
                "has_transcript": bool(discovered_voices[name].get("ref_text")),
            }
            for name in discovered_voices
        ]
        return {"custom": list(presets_by_model.get(resolved, [])), "cloned": clones}
    presets = [{"id": name, "type": "preset"} for name in available_presets]
    clones = [
        {
            "id": name,
            "type": "clone",
            "has_transcript": bool(discovered_voices[name].get("ref_text")),
        }
        for name in discovered_voices
    ]
    return {"voices": presets + clones, "presets_by_model": presets_by_model}


def _match_preset(voice_name: str,
                  presets: Optional[List[str]] = None) -> Optional[str]:
    """Case-insensitive match against a model's preset voices.

    `presets` scopes the match to one model's custom voice list; without it
    the flat aggregate is used (health reporting / legacy callers only)."""
    candidates = available_presets if presets is None else presets
    name_clean = voice_name.strip().lower()
    for preset in candidates:
        if preset.lower() == name_clean:
            return preset
    return None


def _match_clone(voice_name: str) -> Optional[str]:
    """Case-insensitive match against discovered clone voice stems."""
    name_clean = voice_name.strip().lower()
    for clone in discovered_voices:
        if clone.lower() == name_clean:
            return clone
    return None


# [PART3]

def _read_transcript(voice_info: Dict[str, str]) -> str:
    """Re-read the companion transcript at request time so transcript edits
    apply without a restart (the startup scan only snapshots discovery)."""
    path = voice_info.get("ref_text_path", "")
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                return f.read().strip()
        except Exception as e:
            logger.warning("Failed re-reading transcript %s: %s", path, e)
    return voice_info.get("ref_text", "")


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    raw_input = req.input or ""
    input_text = strip_xml_tags(raw_input) if config.strip_tags else raw_input.strip()
    voice_input = req.voice.strip()
    fmt = (req.response_format or config.response_format).lower()

    logger.info(
        "Incoming speech request: model=%r, voice=%r, format=%r, input_len=%d (raw_len=%d)",
        req.model, voice_input, fmt, len(input_text), len(raw_input),
    )
    logger.debug("Raw request input: %r", raw_input)

    # The model parameter is required and must resolve to a registered
    # adapter (name or alias, case-insensitive).
    resolved_model = model_manager.resolve_model(req.model) if model_manager else None
    if resolved_model is None:
        available = ", ".join(model_manager.model_names()) if model_manager else "none"
        logger.warning(
            "Rejecting speech request (400 Bad Request): unknown model %r (available: %s)",
            req.model, available,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {req.model!r}. Available: {available}")

    # The pinned model's capability view drives voice resolution below.
    caps = model_manager.capabilities_for(resolved_model)

    # Per-model meta-tag handling: square brackets are native content for
    # some models ([Sigh], [Laughter]) and noise for others, so they are
    # stripped only where the manifest asks for it.
    if caps is not None and caps.strip_brackets:
        input_text = strip_bracket_tags(input_text)

    # Inline {instruction} directives are model-agnostic: always extracted and
    # always removed from the text. Whether the resolved model receives the
    # value is decided by the voice_default args mapping in its manifest.
    input_text, instruct = extract_instructions(input_text)

    if not input_text.strip():
        logger.info(
            "Speech request input is empty after sanitization (raw input was: %r); returning short silence response.",
            raw_input,
        )
        # Return 10ms of clean silence (240 samples @ 24kHz) to satisfy clients
        silent_audio = np.zeros(240, dtype=np.float32)
        return _encode_response(silent_audio, 24000, fmt)

    # Early health check before starting synthesis transaction.
    if not model_manager.check_ready(force=False):
        logger.error("Rejecting speech synthesis request: CUDA subsystem is unresponsive or hung.")
        raise HTTPException(
            status_code=503,
            detail="CUDA subsystem is unresponsive or hung (e.g. following system sleep). State reset; please retry.",
        )

    if fmt not in ("mp3", "wav", "pcm"):
        logger.warning("Rejecting speech request (400 Bad Request): Unsupported format %r", fmt)
        raise HTTPException(status_code=400, detail=f"Unsupported format {fmt!r}. Supported: mp3, wav, pcm")

    loop = asyncio.get_event_loop()

    def _execute_synthesis() -> AudioResult:
        # Priority 1: the pinned model's custom (preset) voices. Scoping the
        # match to one model's list also resolves cross-model name collisions.
        if caps is not None and caps.voice_default:
            preset_match = _match_preset(voice_input, caps.presets)
            if preset_match:
                logger.info("Executing preset voice synthesis: %r (model=%r)",
                            preset_match, resolved_model)
                return model_manager.synthesize_default(
                    voice=preset_match, text=input_text, instruct=instruct,
                    model=resolved_model)

        # Priority 2: clone archive (only when the model supports cloning)
        if caps is not None and caps.voice_clone:
            clone_match = _match_clone(voice_input)
            if clone_match:
                voice_info = discovered_voices[clone_match]
                logger.info("Executing voice clone synthesis: %r (model=%r)",
                            clone_match, resolved_model)
                return model_manager.synthesize_clone(
                    text=input_text,
                    ref_audio_path=voice_info["ref_audio"],
                    ref_text=_read_transcript(voice_info),
                    model=resolved_model,
                )

        # Priority 3: Treat the voice string as a natural-language persona description
        if voice_input and caps is not None and caps.voice_create:
            logger.info(
                "Voice %r unresolved for model %r; attempting voice creation.",
                voice_input[:60], resolved_model,
            )
            return model_manager.synthesize_create(
                text=input_text, description=voice_input, model=resolved_model)

        logger.warning(
            "Rejecting speech request (400 Bad Request): voice %r is not a custom or clone voice of model %r.",
            voice_input, resolved_model,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Voice {voice_input!r} is not available for model {resolved_model!r}.")

    try:
        result = await loop.run_in_executor(None, _execute_synthesis)
    except HTTPException:
        raise
    except NoModelForCapabilityError as e:
        logger.error("No loaded model supports the requested interaction: %s", e)
        raise HTTPException(status_code=501, detail=str(e))
    except ModelNotInitializedError as e:
        logger.error("Model runtime not initialized: %s", e)
        raise HTTPException(status_code=501, detail=str(e))
    except ModelNotReadyError as e:
        logger.warning("Model temporarily unavailable: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except SynthesisError as e:
        logger.error("Synthesis failed in model %r: %s", e.model_name, e.cause, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e.cause))
    except Exception as e:
        logger.error("Synthesis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return _encode_response(result.audio, result.sample_rate, fmt)


# [PART4]

def _encode_response(audio: np.ndarray, sample_rate: int, fmt: str) -> Response:
    """Encode float32 mono audio into the requested response format."""
    if fmt == "mp3":
        return Response(content=to_mp3_bytes(audio, sample_rate), media_type="audio/mpeg")
    elif fmt == "wav":
        return Response(content=to_wav_bytes(audio, sample_rate), media_type="audio/wav")
    else:  # pcm
        return Response(content=to_pcm16(audio), media_type="audio/pcm")


def main():
    global config, model_manager
    config = ServerConfig.from_args()

    settings = ModelSettings(
        models=config.models,
        device=config.device,
        dtype=config.dtype,
        enable_prefetch=config.enable_prefetch,
        model_idle_timeout=config.model_idle_timeout,
        prompt_ttl=config.prompt_ttl,
        cuda_check_cooldown=config.cuda_check_cooldown,
        cuda_health_timeout=config.cuda_health_timeout,
        synthesis_timeout=config.synthesis_timeout_seconds,
        cold_start_timeout=config.cold_start_timeout_seconds,
        recovery_cooldown=config.recovery_cooldown_seconds,
    )
    models_dir = Path(REPO_ROOT) / "models"
    model_manager = ModelManager(settings, models_dir)

    # Register cleanup on abnormal python termination
    atexit.register(model_manager.shutdown)

    uvicorn.run(app, host=config.host, port=config.port)

if __name__ == "__main__":
    main()