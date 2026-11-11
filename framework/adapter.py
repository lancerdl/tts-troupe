"""
Model adapter API for tts-troupe.

Defines the protocol every integrated TTS model must satisfy (ModelAdapter),
the capability metadata the server routes on (ModelCapabilities), and a
declarative YAML manifest interpreter that maps server interaction types onto
model methods.

Model repositories integrate by providing a `model.yaml` manifest at their
root. The manifest declares the adapter factory class and, for stateless
synthesis, declarative call/arg/result mappings. Stateful synthesis (e.g.
voice-clone prompt caching) is implemented natively by the adapter class.
"""
import importlib
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Protocol, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class AudioResult(NamedTuple):
    """Synthesized audio returned by a model adapter."""
    audio: np.ndarray        # float32, mono
    sample_rate: int


class ModelCapabilities(NamedTuple):
    """What a model adapter can do. The server routes on this."""
    voice_default: bool            # named preset voices (e.g. Qwen3 CustomVoice)
    voice_clone: bool              # reference-audio cloning (e.g. Qwen3 Base)
    voice_create: bool             # natural-language voice creation (e.g. VoiceDesign)
    presets: Tuple[str, ...] = ()  # valid preset voice names when voice_default
    default_voice: Optional[str] = None  # fallback when client sends empty voice
    aliases: Tuple[str, ...] = ()  # alternate accepted model names (e.g. "qwen3")
    strip_brackets: bool = False   # remove [tag] directives from input text


class ModelAdapter(Protocol):
    """Protocol implemented by every integrated TTS model's adapter.

    Subprocess runtime: the manager-facing surface. Adapters own their CUDA
    context inside their worker process; setup/lazy-loading is internal to
    the adapter (spawn happens on first synthesis), and unloading a worker is
    process death - there is no separate release call.
    """

    def shutdown(self) -> None:
        """Tear down the adapter; idempotent."""
        ...

    def capabilities(self) -> ModelCapabilities: ...

    def age_caches(self) -> None:
        """Age pass hook (voice TTL sweep; per-checkpoint sweep in Phase 2)."""
        ...

    def is_loaded(self) -> bool:
        """Whether any checkpoint/pipeline is currently resident in VRAM.

        Consumed by the worker harness's self-retirement check (`age`
        command): the adapter owns VRAM residency, so it must report it.
        A conservative True prevents self-retirement when unknown.
        """
        ...

    def health(self, force: bool = False) -> bool:
        """Report CUDA/worker health; may kill a hung worker internally."""
        ...

    # ---- the interaction types ----
    def synthesize_default(self, text: str, voice: str,
                           instruct: Optional[str]) -> AudioResult: ...
    def synthesize_clone(self, text: str, ref_audio_path: str,
                         ref_text: str) -> AudioResult: ...
    def synthesize_create(self, text: str, description: str) -> AudioResult: ...


@dataclass
class ModelSettings:
    """Settings handed to model adapters: which model dirs are enabled, plus
    hardware hints the adapter may use or override."""
    models: list = field(default_factory=list)   # e.g. ["faster-qwen3-tts"]
    device: Optional[str] = None
    dtype: Optional[str] = None
    enable_prefetch: bool = True
    model_idle_timeout: float = 60.0             # minutes
    prompt_ttl: float = 30.0                     # minutes
    cuda_check_cooldown: float = 5.0             # seconds
    cuda_health_timeout: float = 2.0             # seconds
    synthesis_timeout: float = 60.0              # seconds (warm-path wire deadline)
    cold_start_timeout: float = 300.0            # seconds (spawn/load/clone deadline)
    recovery_cooldown: float = 60.0              # seconds (degraded suppression)


class ModelNotReadyError(RuntimeError):
    """Raised when the hardware is unresponsive or the model is not loaded."""


class NoModelForCapabilityError(LookupError):
    """Raised when no loaded model supports the requested interaction type."""


class ModelNotInitializedError(RuntimeError):
    """Raised when a model's runtime environment (venv) is not set up.

    Maps to HTTP 501: the user must run the model's own setup script before
    the model can serve requests.
    """



class SynthesisError(RuntimeError):
    """Raised when a model fails to synthesize audio."""

    def __init__(self, model_name: str, cause: BaseException):
        self.model_name = model_name
        self.cause = cause
        super().__init__(f"{model_name} failed: {cause}")


# ---------------------------------------------------------------------------
# Manifest interpreter
# ---------------------------------------------------------------------------

def _resolve_arg(spec: Any, server_args: Dict[str, Any], model_config: Dict[str, Any]) -> Any:
    """Resolve a manifest arg spec into a concrete value.

    - string        -> passthrough of the named server argument
    - {const: v}    -> literal value
    - {config: k}   -> value from the model's own config.yaml
    """
    if isinstance(spec, dict):
        if "const" in spec:
            return spec["const"]
        if "config" in spec:
            return model_config[spec["config"]]
        raise ValueError(f"Unsupported arg spec: {spec!r}")
    if spec not in server_args:
        raise ValueError(f"Server argument {spec!r} not provided")
    return server_args[spec]


def _index_path(spec: Any) -> list:
    if isinstance(spec, dict) and "index" in spec:
        idx = spec["index"]
        return idx if isinstance(idx, list) else [idx]
    raise ValueError(f"Unsupported result spec: {spec!r}")


def _extract_result(result_spec: Dict[str, Any], raw: Any) -> AudioResult:
    """Extract audio/sample_rate from a model return value via index specs."""
    audio = raw
    for idx in _index_path(result_spec.get("audio", {"index": 0})):
        audio = audio[idx]
    sr = raw
    for idx in _index_path(result_spec.get("sample_rate", {"index": 1})):
        sr = sr[idx]
    return AudioResult(audio=audio, sample_rate=int(sr))


def capabilities_from_manifest(manifest: Dict[str, Any]) -> ModelCapabilities:
    """Derive ModelCapabilities purely from a model.yaml manifest.

    Manifest-only by design: the subprocess proxy must build the capability
    view WITHOUT importing or executing any model code (an absent venv must
    still route and report capabilities).
    """
    caps = manifest.get("capabilities", {})
    vd = caps.get("voice_default")
    vc = caps.get("voice_clone")
    vcr = caps.get("voice_create")
    return ModelCapabilities(
        voice_default=vd is not None,
        voice_clone=vc is not None,
        voice_create=vcr is not None,
        presets=tuple(vd.get("presets", ())) if vd else (),
        default_voice=vd.get("default_voice") if vd else None,
        aliases=tuple(manifest.get("aliases", ()) or ()),
        strip_brackets=bool(manifest.get("strip_brackets", False)),
    )


class MappedAdapter:
    """A ModelAdapter built from a model.yaml manifest.

    Declarative capabilities dispatch through call/args/result mappings onto
    methods of the factory-created adapter object. A capability declared with
    `implementation: native` delegates to the adapter object's own
    synthesize_<type> method.
    """

    def __init__(self, manifest: Dict[str, Any], adapter_obj: Any,
                 model_config: Dict[str, Any], model_dir: Path):
        self.manifest = manifest
        self.name = manifest.get("name", model_dir.name)
        self.source = manifest.get("source", "")
        self.obj = adapter_obj
        self.model_config = model_config
        self.model_dir = model_dir
        self._caps = self._build_capabilities()

    # ---- lifecycle: delegated to the adapter object ----
    def shutdown(self) -> None:
        self.obj.shutdown()

    def is_loaded(self) -> bool:
        """VRAM residency; guarded for legacy adapters lacking the method."""
        loaded = getattr(self.obj, "is_loaded", None)
        return bool(loaded()) if callable(loaded) else True

    def capabilities(self) -> ModelCapabilities:
        return self._caps

    def age_caches(self) -> None:
        """Adapter-level aging hook (transport-blind; called by the cleaner)."""
        age = getattr(self.obj, "age_caches", None)
        if callable(age):
            age()

    def _build_capabilities(self) -> ModelCapabilities:
        return capabilities_from_manifest(self.manifest)

    def _dispatch(self, cap_type: str, server_args: Dict[str, Any]) -> AudioResult:
        block = self.manifest.get("capabilities", {}).get(cap_type)
        if block is None:
            raise NoModelForCapabilityError(
                f"model {self.name!r} does not support {cap_type}"
            )
        if "implementation" in block:
            # Native: the adapter object implements synthesize_<interaction>
            # itself (e.g. capability voice_clone -> synthesize_clone).
            method = getattr(self.obj, f"synthesize_{cap_type.removeprefix('voice_')}")
            raw = method(**server_args)
            # Adapters written as plain (audio, sample_rate) tuples are
            # normalized to AudioResult; AudioResult itself passes through.
            if isinstance(raw, tuple) and not isinstance(raw, AudioResult):
                raw = AudioResult(*raw)
            return raw
        # Declarative: resolve args, call the mapped method, extract the result.
        instance_key = block.get("instance")
        target = self.obj.get_instance(instance_key) if instance_key else self.obj
        method = getattr(target, block["call"])
        declared = {
            spec for spec in (block.get("args") or {}).values()
            if isinstance(spec, str)
        }
        if server_args.get("instruct") and "instruct" not in declared:
            # The caller asked for delivery steering and this model has no
            # channel for it. Not an error - the text was still cleaned - but
            # silence would be indistinguishable from the instruction having
            # been applied, so record it.
            logger.info(
                "[%s] capability %r does not consume 'instruct'; "
                "extracted instruction %r discarded",
                self.name, cap_type, server_args["instruct"])
        resolved = {
            arg_name: _resolve_arg(spec, server_args, self.model_config)
            for arg_name, spec in block.get("args", {}).items()
        }
        raw = method(**resolved)
        return _extract_result(block.get("result", {}), raw)

    def synthesize_default(self, text: str, voice: str,
                           instruct: Optional[str] = None) -> AudioResult:
        return self._dispatch("voice_default", {"text": text, "voice": voice, "instruct": instruct})

    def synthesize_clone(self, text: str, ref_audio_path: str,
                         ref_text: str = "") -> AudioResult:
        return self._dispatch("voice_clone", {"text": text, "ref_audio_path": ref_audio_path, "ref_text": ref_text})

    def synthesize_create(self, text: str, description: str) -> AudioResult:
        return self._dispatch("voice_create", {"text": text, "description": description})


def load_manifest(model_dir: Path) -> Dict[str, Any]:
    """Read and validate model.yaml from a model repo directory."""
    manifest_path = model_dir / "model.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No model.yaml in {model_dir}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    for required in ("name", "module", "setup"):
        if required not in manifest:
            raise ValueError(f"model.yaml in {model_dir} missing required key {required!r}")
    return manifest


def _load_model_config(model_dir: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    config_file = manifest.get("setup", {}).get("config_file")
    if not config_file:
        return {}
    config_path = model_dir / config_file
    if not config_path.exists():
        logger.warning("Model config %s not found in %s; using empty config.",
                       config_file, model_dir)
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _import_module_from_dir(model_dir: Path, module_name: str):
    """Import a module from a model repo directory, making the dir importable."""
    model_dir_str = str(model_dir)
    if model_dir_str not in sys.path:
        sys.path.insert(0, model_dir_str)
    return importlib.import_module(module_name)


def create_adapter(model_dir: Path) -> MappedAdapter:
    """Create a ModelAdapter for a model repo directory.

    Reads model.yaml, imports the declared module, instantiates the factory
    class with the model's own config, and wraps it in a MappedAdapter.
    """
    manifest = load_manifest(model_dir)
    module = _import_module_from_dir(model_dir, manifest["module"])
    factory_name = manifest.get("setup", {}).get("factory")
    if not factory_name:
        raise ValueError(f"model.yaml in {model_dir} missing setup.factory")
    factory = getattr(module, factory_name)
    model_config = _load_model_config(model_dir, manifest)
    adapter_obj = factory(config=model_config, model_dir=model_dir)
    logger.info("Loaded model adapter %r from %s", manifest.get("name"), model_dir)
    return MappedAdapter(manifest, adapter_obj, model_config, model_dir)
