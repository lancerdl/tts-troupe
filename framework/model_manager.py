"""
Model manager for tts-troupe.

Generic lifecycle manager for model adapters declared via model.yaml manifests
in cloned model repos under models/. Handles config-declared loading, lazy
on-demand VRAM loading, idle aging, and CUDA health gating. No model-specific
knowledge lives here.
"""
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .adapter import (
    AudioResult,
    ModelNotInitializedError,
    ModelNotReadyError,
    ModelSettings,
    NoModelForCapabilityError,
    SynthesisError,
)
from .proxy_adapter import create_proxy_adapter

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages loading, execution locks, and idle unloading for model adapters.

    Each adapter is created from a model repo directory (containing model.yaml)
    listed in the server config's `models:` list. VRAM setup is lazy: an adapter
    is only initialized on the first synthesis request that needs it.
    """

    def __init__(self, settings: ModelSettings, models_dir: Path):
        self.settings = settings
        self.models_dir = models_dir
        self._models: Dict[str, Any] = {}
        self._aliases: Dict[str, str] = {}   # alias (lower) -> canonical model name
        self._last_accessed: Dict[str, float] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._manager_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._shutdown_completed = False
        self._cleaner_thread: Optional[threading.Thread] = None
        self._last_cuda_success_time: float = 0.0

    # ---- lifecycle (server calls from FastAPI lifespan / atexit) ----
    def load(self) -> None:
        """Import declared model repos, create adapters, start the aging thread.

        VRAM setup stays lazy until the first synthesis; prefetch/verification
        is model-specific and handled inside each adapter's setup().
        """
        for name in self.settings.models:
            model_dir = self.models_dir / name
            if not model_dir.is_dir():
                logger.warning("Declared model %r not found in %s; skipping.",
                               name, self.models_dir)
                continue
            try:
                adapter = create_proxy_adapter(model_dir, self.settings)
            except Exception as e:
                logger.error("Failed to load model adapter %r from %s: %s",
                             name, model_dir, e)
                continue
            with self._manager_lock:
                aliases = [a.strip().lower() for a in
                           (getattr(adapter.capabilities(), "aliases", ()) or ())
                           if a.strip()]
                clash = sorted({a for a in aliases if self._aliases.get(a, adapter.name) != adapter.name})
                if clash:
                    logger.error(
                        "Refusing adapter %r: alias collision %s (already owned by %s)",
                        adapter.name, clash,
                        [self._aliases[a] for a in clash],
                    )
                    continue
                for a in aliases:
                    self._aliases[a] = adapter.name
                self._models[adapter.name] = adapter
                self._last_accessed[adapter.name] = 0.0
                self._locks[adapter.name] = threading.Lock()
            logger.info("Registered model adapter %r (capabilities: %s)",
                        adapter.name, adapter.capabilities())
        self._start_idle_cleaner()

    def shutdown(self) -> None:
        """
        Idempotent generic teardown: stops the aging thread, shuts down every
        registered adapter (in-process: release VRAM/state; proxy: shutdown
        the worker process), and marks the manager closed. Subsequent calls
        return immediately.
        """
        self._shutdown_event.set()
        with self._manager_lock:
            if self._shutdown_completed:
                logger.info("[MEMORY] Shutdown already completed; skipping duplicate call.")
                return
            self._shutdown_completed = True
        logger.info("[MEMORY] - Executing server shutdown VRAM cleanup...")
        with self._manager_lock:
            for name in list(self._models.keys()):
                with self._locks[name]:
                    logger.info("[MEMORY] - UNLOADING model from VRAM: %s...", name)
                    self._models[name].shutdown()
                    self._last_accessed[name] = 0.0
        logger.info("[MEMORY] - ALL models unloaded from VRAM.")

    def _start_idle_cleaner(self) -> None:
        def cleaner():
            while not self._shutdown_event.wait(30):
                if self._shutdown_event.is_set():
                    return
                # Check CUDA health before attempting any resource maintenance
                if not self.ensure_cuda_healthy(force=False):
                    logger.warning("[MEMORY] Idle cleaner skipped aging pass: CUDA is not responsive.")
                    continue
                # Age idle models out of VRAM
                try:
                    self.age_models()
                except Exception as e:
                    logger.warning("Error during model aging pass: %s", e)
                # Age model-owned caches (adapters implement their own eviction)
                with self._manager_lock:
                    adapters = list(self._models.values())
                for adapter in adapters:
                    age_caches = getattr(adapter, "age_caches", None)
                    if callable(age_caches):
                        try:
                            age_caches()
                        except Exception as e:
                            logger.warning("Error aging caches for %s: %s", adapter.name, e)

        self._cleaner_thread = threading.Thread(target=cleaner, daemon=True,
                                                name="memory_idle_cleaner")
        self._cleaner_thread.start()
        logger.info(
            "Started unified background memory cleaner (model idle timeout: %d mins, check cooldown: %.1fs)",
            self.settings.model_idle_timeout, self.settings.cuda_check_cooldown
        )

    # ---- synthesis (server resolves the interaction type; manager dispatches) ----
    def _pick(self, capability: str, model: Optional[str] = None) -> str:
        """Return the name of the adapter supporting `capability`.

        With `model` set (a canonical adapter name), only that adapter is
        considered: a missing adapter or missing capability raises
        NoModelForCapabilityError."""
        if model is not None:
            resolved = self.resolve_model(model)
            with self._manager_lock:
                adapter = self._models.get(resolved) if resolved else None
            if adapter is None or not getattr(adapter.capabilities(), capability):
                raise NoModelForCapabilityError(
                    f"model {model!r} does not support {capability!r}")
            return resolved
        with self._manager_lock:
            for name, adapter in self._models.items():
                if getattr(adapter.capabilities(), capability):
                    return name
        raise NoModelForCapabilityError(f"No loaded model supports {capability!r}")

    def resolve_model(self, model: str) -> Optional[str]:
        """Resolve a client model string to a canonical adapter name.

        Matches adapter names first, then declared aliases (both
        case-insensitive). Returns None when nothing matches."""
        key = model.strip().lower()
        if not key:
            return None
        with self._manager_lock:
            for name in self._models:
                if name.lower() == key:
                    return name
            return self._aliases.get(key)

    def model_names(self) -> List[str]:
        """Human-readable list of model names plus their aliases."""
        with self._manager_lock:
            adapters = list(self._models.items())
        out = []
        for name, adapter in adapters:
            aliases = list(getattr(adapter.capabilities(), "aliases", ()) or ())
            out.append(f"{name}" + (f" (aliases: {', '.join(aliases)})" if aliases else ""))
        return out

    def presets_by_model(self) -> Dict[str, List[str]]:
        """Manifest-declared preset voices per adapter name (no loading)."""
        with self._manager_lock:
            return {n: list(a.capabilities().presets) for n, a in self._models.items()}

    def capabilities_for(self, name: str) -> Optional[Any]:
        """Capability view of one registered adapter, by canonical name."""
        with self._manager_lock:
            adapter = self._models.get(name)
        return adapter.capabilities() if adapter else None

    def capabilities(self) -> List:
        """Aggregated capability view from all registered adapters (no loading)."""
        with self._manager_lock:
            return [a.capabilities() for a in self._models.values()]

    def speaker_catalogue(self, model: Optional[str] = None) -> List[Dict[str, Any]]:
        """Complete built-in speaker catalogue per model.

        With `model` (canonical name), only that adapter is queried - the
        caller decides when the checkpoint-load cost is acceptable, and this
        filter makes sure only the requested model's checkpoint can load."""
        with self._manager_lock:
            if model is not None:
                adapters = [a for n, a in self._models.items() if n == model]
            else:
                adapters = list(self._models.values())
        out: List[Dict[str, Any]] = []
        for adapter in adapters:
            entry: Dict[str, Any] = {"model": getattr(adapter, "name", "?")}
            getter = getattr(adapter, "supported_speakers", None)
            if callable(getter):
                try:
                    entry["speakers"] = getter().get("speakers", [])
                except Exception as e:
                    entry["error"] = str(e)
            else:
                entry["speakers"] = []
            out.append(entry)
        return out

    def cache_status(self) -> List[Dict[str, Any]]:
        """Aggregated voice-cache visibility.

        The manager owns which copies are cached and where; the server only
        knows the voice library itself."""
        with self._manager_lock:
            adapters = list(self._models.values())
        out: List[Dict[str, Any]] = []
        for adapter in adapters:
            entry: Dict[str, Any] = {"model": getattr(adapter, "name", "?")}
            info = getattr(adapter, "cache_info", None)
            if callable(info):
                try:
                    entry.update(info() or {})
                except Exception as e:
                    entry["error"] = str(e)
            else:
                entry["cache"] = "none"
            out.append(entry)
        return out

    def synthesize_clone(self, text: str, ref_audio_path: str,
                         ref_text: str = "", instruct: Optional[str] = None,
                         model: Optional[str] = None) -> AudioResult:
        """Synthesize a voice clone from a reference audio file.

        With `model`, the synthesis is pinned to that adapter (which must
        support voice cloning)."""
        name = self._pick("voice_clone", model)
        return self._synthesize(name, "synthesize_clone",
                                text=text, ref_audio_path=ref_audio_path,
                                ref_text=ref_text, instruct=instruct)

    def synthesize_preset(self, voice: str, text: str,
                          instruct: Optional[str] = None,
                          model: Optional[str] = None) -> AudioResult:
        """Synthesize speech with a built-in preset voice.

        With `model`, the synthesis is pinned to that adapter (which must
        support preset voices)."""
        name = self._pick("voice_preset", model)
        return self._synthesize(name, "synthesize_preset",
                                text=text, voice=voice, instruct=instruct)

    def synthesize_create(self, text: str, description: str,
                          model: Optional[str] = None) -> AudioResult:
        """Synthesize speech described by a natural-language persona.

        With `model`, the synthesis is pinned to that adapter (which must
        support voice creation)."""
        name = self._pick("voice_create", model)
        return self._synthesize(name, "synthesize_create",
                                text=text, description=description)

    def _synthesize(self, name: str, method: str, **kwargs) -> AudioResult:
        adapter = self._get_adapter(name)
        with self._locks[name]:
            try:
                result = getattr(adapter, method)(**kwargs)
            except (NoModelForCapabilityError, ModelNotInitializedError,
                    ModelNotReadyError):
                # Routing/environment conditions: preserve their own HTTP
                # mapping (501 / 503) instead of wrapping into a 500.
                raise
            except Exception as e:
                raise SynthesisError(name, e)
        self.record_cuda_success()
        return result

    def _get_adapter(self, name: str) -> Any:
        with self._manager_lock:
            adapter = self._models.get(name)
            if adapter is None:
                raise NoModelForCapabilityError(f"No model named {name!r} loaded")
            self._last_accessed[name] = time.time()
            return adapter

    def record_cuda_success(self) -> None:
        """Advance the cooldown timestamp so subsequent checks skip the probe."""
        self._last_cuda_success_time = time.time()

    def ensure_cuda_healthy(self, force: bool = False) -> bool:
        """
        Verify that the CUDA subsystem is alive before executing operations.

        Every adapter owns its CUDA context inside its worker process and
        answers via its `health()` hook; results are ANDed.

        Methodology:
          1. Cooldown Bypass: if an operation succeeded within the last
             `cuda_check_cooldown` seconds and `force` is False, return True.
          2. Adapter health: a not-running worker is healthy-idle (never an
             error); a hung or CUDA-dead worker is killed by its own health
             hook, and the next request respawns it with a fresh context.
        """
        now = time.time()
        if not force and (now - self._last_cuda_success_time) < self.settings.cuda_check_cooldown:
            return True

        with self._manager_lock:
            adapters = list(self._models.values())

        healthy = True
        for adapter in adapters:
            try:
                if not adapter.health(force):
                    healthy = False
            except Exception as e:
                logger.warning("[CUDA HEALTH] adapter %s health check failed: %s",
                               getattr(adapter, "name", "?"), e)
                healthy = False

        if healthy:
            # Keep the manager-level cooldown warm so the entry bypass stays
            # meaningful.
            self._last_cuda_success_time = time.time()

        return healthy

    # Plan-facing alias: the health gate is called `check_ready` in the
    # migration plan; `ensure_cuda_healthy` is the implementation name.
    check_ready = ensure_cuda_healthy

    def age_models(self) -> int:
        """Age out and unload idle models whose inactivity exceeds the timeout."""
        model_timeout_seconds = self.settings.model_idle_timeout * 60
        if model_timeout_seconds <= 0:
            return 0

        unloaded_count = 0
        now = time.time()
        with self._manager_lock:
            for name in list(self._models.keys()):
                last_used = self._last_accessed.get(name, 0.0)
                if last_used > 0.0 and (now - last_used) > model_timeout_seconds:
                    logger.info(
                        "[MEMORY] Model %s idle for > %d mins. Aging model out of VRAM...",
                        name, self.settings.model_idle_timeout
                    )
                    self.unload_model(name)
                    unloaded_count += 1
        return unloaded_count

    def unload_model(self, name: str) -> None:
        """
        Unload a specific model adapter from VRAM.

        NOTE: Caller MUST hold `self._manager_lock` before calling this method.
        Serializes against ongoing execution via `self._locks[name]`.
        """
        if name in self._models:
            with self._locks[name]:
                logger.info("[MEMORY] - UNLOADING model from VRAM: %s...", name)
                self._models[name].shutdown()
                self._last_accessed[name] = 0.0
                logger.info("[MEMORY] - UNLOADED model from VRAM: %s.", name)


