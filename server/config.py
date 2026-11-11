"""
Configuration and settings management for tts-troupe.
Loads settings from config.yaml with overrides from environment variables and CLI arguments.
"""
import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class ServerConfig:
    # Audio asset directory default
    DEFAULT_VOICES_DIR = "voices"

    @staticmethod
    def _load_yaml(config_path: Optional[str]) -> Dict[str, Any]:
        base_dir = os.path.dirname(__file__)
        path = config_path or os.environ.get("TTS_CONFIG")
        if not path:
            candidate = os.path.join(base_dir, "config.yaml")
            if os.path.exists(candidate):
                path = candidate
            else:
                # First run: create a local config.yaml from the committed example.
                example = os.path.join(base_dir, "config.yaml.example")
                if os.path.exists(example):
                    try:
                        with open(example, "r", encoding="utf-8") as f:
                            content = f.read()
                        with open(candidate, "w", encoding="utf-8") as f:
                            f.write(content)
                        logger.info("Created %s from %s", candidate, example)
                        path = candidate
                    except Exception as e:
                        logger.warning("Could not create %s from example: %s", candidate, e)
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                logger.info("Loaded configuration from %s", path)
                return data
            except Exception as e:
                logger.warning("Failed to load config from %s: %s", path, e)
        return {}

    def __init__(
        self,
        models: Optional[List[str]] = None,
        voices_dir: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        response_format: Optional[str] = None,
        model_idle_timeout: Optional[int] = None,
        prompt_ttl: Optional[int] = None,
        cuda_check_cooldown: Optional[float] = None,
        cuda_health_timeout: Optional[float] = None,
        synthesis_timeout: Optional[float] = None,
        cold_start_timeout: Optional[float] = None,
        recovery_cooldown: Optional[float] = None,
        enable_prefetch: Optional[bool] = None,
        strip_tags: Optional[bool] = None,
        config_path: Optional[str] = None,
    ):
        yaml_data = self._load_yaml(config_path)
        srv = yaml_data.get("server", {})
        mdl = yaml_data.get("models", {})
        vce = yaml_data.get("voices", {})
        mem = yaml_data.get("memory_management", {})

        # Precedence: --model (CLI) > TTS_MODELS env > config.yaml models.list.
        # No models are bundled or assumed: if no source is set, load none.
        env_models = self._parse_models(os.environ.get("TTS_MODELS"))
        self.models = (
            list(models)
            if models is not None
            else env_models
            or mdl.get("list")
            or []
        )
        self.device = device or os.environ.get("TTS_DEVICE") or mdl.get("device", "cuda")
        self.dtype = dtype or os.environ.get("TTS_DTYPE") or mdl.get("dtype", "bfloat16")

        raw_voices_dir = (
            voices_dir
            or os.environ.get("TTS_VOICES_DIR")
            or vce.get("voices_dir")
            or self.DEFAULT_VOICES_DIR
        )
        if not os.path.exists(raw_voices_dir):
            local_fallback = os.path.abspath(os.path.join(os.path.dirname(__file__), "voices"))
            if os.path.exists(local_fallback):
                raw_voices_dir = local_fallback
        self.voices_dir = os.path.abspath(raw_voices_dir)

        self.host = host or os.environ.get("TTS_HOST") or srv.get("host", "127.0.0.1")
        self.port = int(port or os.environ.get("TTS_PORT") or srv.get("port", 8000))
        raw_fmt = (
            response_format
            or os.environ.get("TTS_RESPONSE_FORMAT")
            or srv.get("response_format")
            or "mp3"
        )
        self.response_format = str(raw_fmt).lower()

        env_timeout = os.environ.get("TTS_MODEL_IDLE_TIMEOUT")
        if model_idle_timeout is not None:
            self.model_idle_timeout = int(model_idle_timeout)
        elif env_timeout is not None:
            self.model_idle_timeout = int(env_timeout)
        else:
            self.model_idle_timeout = int(mem.get("model_idle_timeout_minutes", 30))

        env_ttl = os.environ.get("TTS_PROMPT_TTL")
        if prompt_ttl is not None:
            self.prompt_ttl = int(prompt_ttl)
        elif env_ttl is not None:
            self.prompt_ttl = int(env_ttl)
        else:
            self.prompt_ttl = int(mem.get("prompt_cache_ttl_minutes", 15))

        env_cooldown = os.environ.get("TTS_CUDA_CHECK_COOLDOWN")
        if cuda_check_cooldown is not None:
            self.cuda_check_cooldown = float(cuda_check_cooldown)
        elif env_cooldown is not None:
            self.cuda_check_cooldown = float(env_cooldown)
        else:
            self.cuda_check_cooldown = float(mem.get("cuda_check_cooldown_seconds", 5.0))

        env_health_timeout = os.environ.get("TTS_CUDA_HEALTH_TIMEOUT")
        if cuda_health_timeout is not None:
            self.cuda_health_timeout = float(cuda_health_timeout)
        elif env_health_timeout is not None:
            self.cuda_health_timeout = float(env_health_timeout)
        else:
            self.cuda_health_timeout = float(mem.get("cuda_health_timeout_seconds", 2.0))

        env_synth_timeout = os.environ.get("TTS_SYNTHESIS_TIMEOUT")
        if synthesis_timeout is not None:
            self.synthesis_timeout_seconds = float(synthesis_timeout)
        elif env_synth_timeout is not None:
            self.synthesis_timeout_seconds = float(env_synth_timeout)
        else:
            self.synthesis_timeout_seconds = float(srv.get("synthesis_timeout_seconds", 60.0))

        env_cold_timeout = os.environ.get("TTS_COLD_START_TIMEOUT")
        if cold_start_timeout is not None:
            self.cold_start_timeout_seconds = float(cold_start_timeout)
        elif env_cold_timeout is not None:
            self.cold_start_timeout_seconds = float(env_cold_timeout)
        else:
            self.cold_start_timeout_seconds = float(srv.get("cold_start_timeout_seconds", 300.0))

        env_recovery_cooldown = os.environ.get("TTS_RECOVERY_COOLDOWN")
        if recovery_cooldown is not None:
            self.recovery_cooldown_seconds = float(recovery_cooldown)
        elif env_recovery_cooldown is not None:
            self.recovery_cooldown_seconds = float(env_recovery_cooldown)
        else:
            self.recovery_cooldown_seconds = float(srv.get("recovery_cooldown_seconds", 60.0))

        if enable_prefetch is not None:
            self.enable_prefetch = enable_prefetch
        else:
            self.enable_prefetch = mdl.get("enable_prefetch", True)

        env_strip_tags = os.environ.get("TTS_STRIP_TAGS")
        if strip_tags is not None:
            self.strip_tags = bool(strip_tags)
        elif env_strip_tags is not None:
            self.strip_tags = env_strip_tags.strip().lower() in ("1", "true", "yes", "on")
        else:
            self.strip_tags = bool(srv.get("strip_tags", True))

    @staticmethod
    def _parse_models(raw: Optional[str]) -> Optional[List[str]]:
        """Parse a comma-separated TTS_MODELS env value into a list."""
        if not raw:
            return None
        models = [m.strip() for m in raw.split(",") if m.strip()]
        return models or None

    @classmethod
    def from_args(cls, args: Optional[argparse.Namespace] = None) -> "ServerConfig":
        if args is None:
            parser = argparse.ArgumentParser(description="TTS Troupe")
            parser.add_argument("--config", default=None, help="Path to config.yaml (default: server/config.yaml)")
            parser.add_argument("--model", dest="models", action="append", default=None,
                                help="Model repo name under models/ to enable (repeatable; overrides config)")
            parser.add_argument("--voices-dir", default=None, help="Directory containing reference audio and companion transcripts")
            parser.add_argument("--host", default=None, help="Server bind host")
            parser.add_argument("--port", type=int, default=None, help="Server bind port")
            parser.add_argument("--device", default=None, help="Torch device (cuda, cpu)")
            parser.add_argument("--dtype", default=None, help="Torch dtype (bfloat16, float16)")
            parser.add_argument("--response-format", default=None, help="Default output format when client omits it (mp3, wav, pcm)")
            parser.add_argument("--model-idle-timeout", type=int, default=None, help="Minutes before unloading idle models from VRAM (0 = disable)")
            parser.add_argument("--prompt-ttl", type=int, default=None, help="Minutes before evicting inactive voice clone prompts")
            parser.add_argument("--synthesis-timeout", type=float, default=None, help="Seconds allowed for a warm-path synthesis wire request (default 60)")
            parser.add_argument("--cold-start-timeout", type=float, default=None, help="Seconds allowed for cold-path work: worker spawn, model load, voice extraction (default 300)")
            parser.add_argument("--recovery-cooldown", type=float, default=None, help="Seconds of fast-503 suppression after a failed re-initialization (default 60)")
            parser.add_argument("--no-prefetch", action="store_true", help="Skip model prefetch at startup")
            parser.add_argument("--strip-tags", dest="strip_tags", action="store_true", default=None, help="Strip <tags> from input text before synthesis while preserving inner text (default: True)")
            parser.add_argument("--no-strip-tags", dest="strip_tags", action="store_false", help="Disable stripping of <tags> from input text")
            args = parser.parse_args()

        return cls(
            models=args.models,
            voices_dir=args.voices_dir,
            host=args.host,
            port=args.port,
            device=args.device,
            dtype=args.dtype,
            response_format=getattr(args, "response_format", None),
            model_idle_timeout=args.model_idle_timeout,
            prompt_ttl=args.prompt_ttl,
            synthesis_timeout=getattr(args, "synthesis_timeout", None),
            cold_start_timeout=getattr(args, "cold_start_timeout", None),
            recovery_cooldown=getattr(args, "recovery_cooldown", None),
            enable_prefetch=False if getattr(args, "no_prefetch", False) else None,
            strip_tags=getattr(args, "strip_tags", None),
            config_path=getattr(args, "config", None),
        )

