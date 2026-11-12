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
from typing import Any, Dict, List, NamedTuple, Optional, Protocol, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class AudioResult(NamedTuple):
    """Synthesized audio returned by a model adapter."""
    audio: np.ndarray        # float32, mono
    sample_rate: int


class ModelCapabilities(NamedTuple):
    """What a model adapter can do. The server routes on this."""
    voice_preset: bool             # built-in/preset voices (e.g. Qwen3 CustomVoice)
    voice_clone: bool              # reference-audio cloning (e.g. Qwen3 Base)
    voice_create: bool             # natural-language voice creation (e.g. VoiceDesign)
    presets: Tuple[str, ...] = ()  # built-in voice names when voice_preset
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
    def synthesize_preset(self, text: str, voice: str,
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
    - {dict: {...}} -> structured kwarg; each value is itself a spec
    """
    if isinstance(spec, dict):
        if "const" in spec:
            return spec["const"]
        if "config" in spec:
            return model_config[spec["config"]]
        if "dict" in spec:
            # Compose a structured kwarg, e.g. controls={emotion: <label>}.
            return {key: _resolve_arg(value, server_args, model_config)
                    for key, value in (spec["dict"] or {}).items()}
        raise ValueError(f"Unsupported arg spec: {spec!r}")
    if spec not in server_args:
        raise ValueError(f"Server argument {spec!r} not provided")
    return server_args[spec]


def _index_path(spec: Any) -> list:
    if isinstance(spec, dict) and "index" in spec:
        idx = spec["index"]
        return idx if isinstance(idx, list) else [idx]
    raise ValueError(f"Unsupported result spec: {spec!r}")


def _extract_result(result_spec: Dict[str, Any], raw: Any,
                    target: Any = None) -> AudioResult:
    """Extract audio/sample_rate from a model return value.

    ``audio`` and ``sample_rate`` take ``{index: ...}`` specs into ``raw``; an
    empty index list (``{index: []}``) means "the return value itself".
    ``sample_rate`` may instead be ``{attr: <name>}``, read from the target
    object - for models that return only a waveform.
    """
    audio = raw
    for idx in _index_path(result_spec.get("audio", {"index": 0})):
        audio = audio[idx]
    sr_spec = result_spec.get("sample_rate", {"index": 1})
    if isinstance(sr_spec, dict) and "attr" in sr_spec:
        if target is None:
            raise ValueError("sample_rate spec {attr: ...} requires a target")
        sr = getattr(target, sr_spec["attr"])
    else:
        sr = raw
        for idx in _index_path(sr_spec):
            sr = sr[idx]
    return AudioResult(audio=audio, sample_rate=int(sr))


class InstructResolution(NamedTuple):
    """Outcome of resolving an extracted ``{...}`` instruction.

    ``state`` drives capability variant selection:
      - ``absent`` - no usable instruction (none sent, or dropped on a miss)
      - ``mapped`` - matched an ``instruct_map`` entry (``value`` = label)
      - ``raw``    - passed through unmapped (``value`` = original text)
    """
    value: Optional[str]
    state: str


def resolve_instruct(block: Dict[str, Any],
                     instruct: Optional[str]) -> InstructResolution:
    """Resolve an extracted ``{...}`` instruction against a capability block.

    Manifest-driven and model-agnostic: ``instruct_map`` translates
    client-facing terms to model-native labels (e.g. Ming's ``sad`` -> 悲伤),
    and ``instruct_on_miss`` decides what an unmapped value becomes - ``drop``
    (the default once a map exists) or ``raw``.
    """
    text = "" if instruct is None else str(instruct).strip()
    if not text:
        return InstructResolution(None, "absent")
    mapping = {str(key).strip().lower(): value
               for key, value in (block.get("instruct_map") or {}).items()}
    if not mapping:
        return InstructResolution(text, "raw")
    # The server joins multiple directives with ", "; the first hit wins, and
    # a value the client already sent as a label is accepted as-is.
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        if token.lower() in mapping:
            return InstructResolution(mapping[token.lower()], "mapped")
        if token in mapping.values():
            return InstructResolution(token, "mapped")
    on_miss = str(block.get("instruct_on_miss", "drop")).strip().lower()
    if on_miss == "raw":
        return InstructResolution(text, "raw")
    return InstructResolution(None, "absent")


def capabilities_from_manifest(manifest: Dict[str, Any]) -> ModelCapabilities:
    """Derive ModelCapabilities purely from a model.yaml manifest.

    Manifest-only by design: the subprocess proxy must build the capability
    view WITHOUT importing or executing any model code (an absent venv must
    still route and report capabilities).
    """
    caps = manifest.get("capabilities", {})
    vp = caps.get("voice_preset")
    vc = caps.get("voice_clone")
    vcr = caps.get("voice_create")
    return ModelCapabilities(
        voice_preset=vp is not None,
        voice_clone=vc is not None,
        voice_create=vcr is not None,
        presets=tuple(vp.get("presets", ())) if vp else (),
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

    def _capability_block(self, cap_type: str) -> Dict[str, Any]:
        return (self.manifest.get("capabilities", {}) or {}).get(cap_type) or {}

    @staticmethod
    def _declares_instruct(block: Dict[str, Any]) -> bool:
        """Whether any ``args`` spec (recursively, through {dict: ...})
        resolves from the server's ``instruct`` argument."""
        def specs(spec: Any) -> set:
            if isinstance(spec, str):
                return {spec}
            if isinstance(spec, dict) and "dict" in spec:
                found: set = set()
                for value in (spec["dict"] or {}).values():
                    found |= specs(value)
                return found
            return set()

        declared: set = set()
        for spec in (block.get("args") or {}).values():
            declared |= specs(spec)
        return "instruct" in declared

    def accepts_instruct(self, cap_type: str) -> bool:
        """Whether a capability consumes an instruction anywhere.

        True when the block sets ``accepts_instruct``, a declarative ``args``
        spec binds ``instruct``, or a variant selects on / binds ``instruct``.
        """
        block = self._capability_block(cap_type)
        if block.get("accepts_instruct") or self._declares_instruct(block):
            return True
        for variant in block.get("variants") or []:
            if "instruct" in (variant.get("when") or {}):
                return True
            if self._declares_instruct(variant):
                return True
        return False

    def resolve_instruct(self, cap_type: str,
                         instruct: Optional[str]) -> InstructResolution:
        """Resolve an instruction for a capability, logging any discard."""
        text = "" if instruct is None else str(instruct).strip()
        if not self.accepts_instruct(cap_type):
            if text:
                logger.info(
                    "[%s] capability %r does not consume 'instruct'; "
                    "extracted instruction %r discarded",
                    self.name, cap_type, instruct)
            return InstructResolution(None, "absent")
        resolution = resolve_instruct(self._capability_block(cap_type), instruct)
        if resolution.state == "absent" and text:
            logger.info(
                "[%s] capability %r: instruction %r matched no label; dropped",
                self.name, cap_type, instruct)
        return resolution

    def _select_variant(self, cap_type: str, block: Dict[str, Any],
                        resolution: InstructResolution) -> Dict[str, Any]:
        """Pick the variant matching the instruct state.

        A variant without ``when`` is the default; exactly one is required
        (validated when the manifest loads)."""
        variants = block.get("variants")
        if not variants:
            return block
        default = None
        for variant in variants:
            when = variant.get("when")
            if not when:
                default = variant
                continue
            allowed = when.get("instruct")
            allowed = allowed if isinstance(allowed, list) else [allowed]
            if resolution.state in allowed:
                logger.info(
                    "[%s] capability %r: instruction state %r selected variant "
                    "call=%r", self.name, cap_type, resolution.state,
                    variant.get("call"))
                return variant
        if default is None:
            raise ValueError(
                f"capability {cap_type!r}: no variant for instruct state "
                f"{resolution.state!r} and no default declared")
        return default

    def _dispatch(self, cap_type: str, server_args: Dict[str, Any],
                  resolution: Optional[InstructResolution] = None) -> AudioResult:
        block = self.manifest.get("capabilities", {}).get(cap_type)
        if block is None:
            raise NoModelForCapabilityError(
                f"model {self.name!r} does not support {cap_type}"
            )
        block = self._select_variant(
            cap_type, block, resolution or InstructResolution(None, "absent"))
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
        if server_args.get("instruct") and not self._declares_instruct(block):
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
        return _extract_result(block.get("result", {}), raw, target)

    def synthesize_preset(self, text: str, voice: str,
                          instruct: Optional[str] = None) -> AudioResult:
        return self._dispatch("voice_preset", {"text": text, "voice": voice, "instruct": instruct})

    def synthesize_clone(self, text: str, ref_audio_path: str,
                         ref_text: str = "",
                         instruct: Optional[str] = None) -> AudioResult:
        accepts = self.accepts_instruct("voice_clone")
        resolution = self.resolve_instruct("voice_clone", instruct)
        server_args = {"text": text, "ref_audio_path": ref_audio_path,
                       "ref_text": ref_text}
        if accepts:
            # The key is always present (possibly None): a declarative args
            # mapping requires it, and native signatures default it.
            server_args["instruct"] = resolution.value
        return self._dispatch("voice_clone", server_args, resolution=resolution)

    def synthesize_create(self, text: str, description: str) -> AudioResult:
        return self._dispatch("voice_create", {"text": text, "description": description})


PRESETS_FILE = "presets.txt"


def _read_presets(model_dir: Path) -> List[str]:
    """Read a model's built-in voice list from ``<repo>/presets.txt``.

    Comma-separated (newlines tolerated); the same convention as ``voices.txt``.
    Generated by ``tts-troupe-sync-presets`` but valid hand-written. Returns an
    empty list when the file is absent, blank, or unreadable - preset voices are
    then simply unavailable for that model.
    """
    path = model_dir / PRESETS_FILE
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return []
    names: List[str] = []
    for part in text.replace("\r", ",").replace("\n", ",").split(","):
        name = part.strip()
        if name and name not in names:
            names.append(name)
    if not names:
        logger.warning("%s is empty; %s has no preset voices", path, model_dir.name)
    return names


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
    if "voice_default" in (manifest.get("capabilities") or {}):
        logger.warning(
            "%s: capability 'voice_default' was renamed to 'voice_preset'; "
            "the old key is ignored", model_dir.name)
    # The built-in voice list is a generated sidecar, not a manifest key. It
    # fills an existing voice_preset capability and can never create one.
    voice_preset = (manifest.get("capabilities") or {}).get("voice_preset")
    if isinstance(voice_preset, dict):
        presets = _read_presets(model_dir)
        if presets:
            voice_preset["presets"] = presets
    _validate_capabilities(manifest, model_dir)
    return manifest


def _validate_capabilities(manifest: Dict[str, Any], model_dir: Path) -> None:
    """Reject ambiguous capability declarations at load time."""
    for name, block in (manifest.get("capabilities") or {}).items():
        variants = block.get("variants")
        if not variants:
            continue
        if "implementation" in block:
            raise ValueError(
                f"model.yaml in {model_dir}: capability {name!r} declares both "
                f"'implementation' and 'variants'")
        if block.get("prompt_cache"):
            raise ValueError(
                f"model.yaml in {model_dir}: capability {name!r} cannot combine "
                f"'variants' with 'prompt_cache' (the cached path calls the "
                f"mapped synthesize method directly)")
        defaults = [v for v in variants if not v.get("when")]
        if len(defaults) != 1:
            raise ValueError(
                f"model.yaml in {model_dir}: capability {name!r} needs exactly "
                f"one default variant (no 'when'), found {len(defaults)}")
        for variant in variants:
            when = variant.get("when")
            if not when:
                continue
            unknown = set(when) - {"instruct"}
            if unknown:
                raise ValueError(
                    f"model.yaml in {model_dir}: capability {name!r} has "
                    f"unsupported 'when' key(s) {sorted(unknown)}")
            allowed = when["instruct"]
            allowed = allowed if isinstance(allowed, list) else [allowed]
            bad = set(allowed) - {"absent", "mapped", "raw"}
            if bad:
                raise ValueError(
                    f"model.yaml in {model_dir}: capability {name!r} has "
                    f"invalid instruct state(s) {sorted(bad)}")


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
