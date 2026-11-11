"""
Utility to auto-transcribe reference audio in the configured voices directory using faster-whisper.
Produces .reference.txt companion files and exports a clean comma-separated voices.txt.
"""
import argparse
import glob
import logging
import os
import sys

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from .config import ServerConfig
from .log_utils import setup_server_logging


def _setup_logging() -> None:
    """Log to both the console (stdout) and per-component log files in logs/."""
    setup_server_logging()



def _ensure_cuda_dlls_on_path() -> None:
    """
    Make CTranslate2 able to find the CUDA 12 runtime DLLs (cublas64_12.dll,
    cudart64_12.dll, cudnn*.dll) that PyTorch bundles inside torch/lib.
    Without this, faster-whisper fails with
    "RuntimeError: Library cublas64_12.dll is not found or cannot be loaded".
    """
    try:
        import torch
    except Exception:
        return
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if not os.path.isdir(torch_lib):
        return
    try:
        # Windows Python 3.8+ API that adds a directory to the DLL search path.
        os.add_dll_directory(torch_lib)
    except (AttributeError, OSError):
        pass
    env_path = os.environ.get("PATH", "")
    if torch_lib not in env_path:
        os.environ["PATH"] = torch_lib + os.pathsep + env_path


def _is_cuda_runtime_error(exc: Exception) -> bool:
    """True when the exception indicates a CUDA runtime / DLL loading problem."""
    msg = str(exc).lower()
    return any(
        k in msg
        for k in ("cublas", "cudart", "cudnn", ".dll", "cuda driver", "library is not found")
    )


_setup_logging()
logger = logging.getLogger("transcribe_voices")


def export_clean_voices_csv(voices_dir: str, voice_stems: list) -> str:
    """Export comma-separated list of voice names (no spaces after commas)."""
    os.makedirs(voices_dir, exist_ok=True)
    out_path = os.path.join(voices_dir, "voices.txt")
    csv_content = ",".join(sorted(voice_stems))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(csv_content)
    logger.info("Exported %d voice name(s) to %s", len(voice_stems), out_path)
    return out_path


def transcribe_voice_library(
    voices_dir: str,
    whisper_model: str = "base.en",
    device: str = "cuda",
    compute_type: str = "float16",
    force: bool = False,
    export_csv: bool = True,
) -> int:
    """Scan voices_dir for .wav files and generate missing .reference.txt transcripts."""
    if not os.path.isdir(voices_dir):
        logger.error("Voices directory not found: %s", voices_dir)
        return 0

    wav_files = glob.glob(os.path.join(voices_dir, "*.wav")) + glob.glob(os.path.join(voices_dir, "**", "*.wav"), recursive=True)
    # Deduplicate: the two glob patterns both match top-level .wav files,
    # which would otherwise cause each top-level file to be processed twice.
    wav_files = list(dict.fromkeys(wav_files))
    if not wav_files:
        logger.warning("No .wav files found in %s", voices_dir)
        return 0

    to_transcribe = []
    all_stems = []

    for wav_path in wav_files:
        stem = os.path.splitext(os.path.basename(wav_path))[0]
        all_stems.append(stem)
        base_path = os.path.splitext(wav_path)[0]
        ref_txt = base_path + ".reference.txt"

        # Strict transcript policy: only .reference.txt counts. Plain .txt
        # files are user meta-data and never satisfy the requirement.
        if force or not os.path.exists(ref_txt):
            to_transcribe.append((wav_path, ref_txt, stem))

    logger.info("Found %d total voice file(s) in %s. %d need transcription.", len(wav_files), voices_dir, len(to_transcribe))

    if to_transcribe:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            logger.error("faster-whisper is not installed. Please run: pip install faster-whisper")
            return 0

        logger.info("Loading Whisper model %r on %s (%s)...", whisper_model, device, compute_type)
        _ensure_cuda_dlls_on_path()
        use_cpu = False
        try:
            model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
        except Exception as e:
            logger.warning("Failed loading on %s (%s), falling back to CPU int8: %s", device, compute_type, e)
            model = WhisperModel(whisper_model, device="cpu", compute_type="int8")
            use_cpu = True

        for idx, (wav_path, ref_txt_path, stem) in enumerate(to_transcribe, 1):
            logger.info("[%d/%d] Transcribing: %s", idx, len(to_transcribe), os.path.basename(wav_path))
            try:
                segments, _info = model.transcribe(wav_path, beam_size=5, vad_filter=True)
                transcript = " ".join(seg.text.strip() for seg in segments).strip()
                if not transcript:
                    transcript = stem

                with open(ref_txt_path, "w", encoding="utf-8") as f:
                    f.write(transcript)
                logger.info("  -> Saved transcript: %r", transcript[:80] + ("..." if len(transcript) > 80 else ""))
            except Exception as e:
                # A CUDA runtime error (e.g. missing cublas64_12.dll) can surface
                # at transcribe-time rather than load-time. Retry once on CPU.
                if not use_cpu and _is_cuda_runtime_error(e):
                    logger.error(
                        "  -> CUDA runtime failure on %s: %s. Reloading on CPU and retrying.",
                        os.path.basename(wav_path),
                        e,
                    )
                    try:
                        model = WhisperModel(whisper_model, device="cpu", compute_type="int8")
                        use_cpu = True
                        segments, _info = model.transcribe(wav_path, beam_size=5, vad_filter=True)
                        transcript = " ".join(seg.text.strip() for seg in segments).strip()
                        if not transcript:
                            transcript = stem
                        with open(ref_txt_path, "w", encoding="utf-8") as f:
                            f.write(transcript)
                        logger.info("  -> Saved transcript (CPU retry): %r", transcript[:80] + ("..." if len(transcript) > 80 else ""))
                    except Exception as e2:
                        logger.error("  -> Failed transcribing %s (CPU retry too): %s", wav_path, e2)
                else:
                    logger.error("  -> Failed transcribing %s: %s", wav_path, e)

    # Re-export clean voices.txt list of discovered voice stems (the
    # standalone CLI's own artifact; the server boot pass skips this so it
    # never touches the archive directory).
    if export_csv:
        export_clean_voices_csv(voices_dir, all_stems)

    return len(to_transcribe)


def main():
    parser = argparse.ArgumentParser(description="Auto-transcribe voices in target voices directory using faster-whisper")
    parser.add_argument("--voices-dir", default=None, help="Voices directory (default: server configured voices_dir)")
    parser.add_argument("--model", default="base.en", help="Whisper model (default: base.en)")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--compute-type", default="float16", help="Compute type (default: float16)")
    parser.add_argument("--force", action="store_true", help="Re-transcribe even if .reference.txt exists")
    args = parser.parse_args()

    # Resolve voices_dir with ServerConfig defaults if not explicitly provided
    voices_dir = args.voices_dir or ServerConfig().voices_dir

    transcribe_voice_library(
        voices_dir=voices_dir,
        whisper_model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        force=args.force,
    )


if __name__ == "__main__":
    main()

