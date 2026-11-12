"""
Subprocess proxy adapter: transports the ModelAdapter protocol over
JSON-lines on a spawned worker's stdio.

Implements the same surface as MappedAdapter (setup/shutdown/is_loaded/
capabilities/synthesize_*/age_caches) so ModelManager stays transport-blind.
The manager never parses a protocol frame; this module is the only place the
wire protocol exists.

Design contract (see .cline/worker_architecture_plan.md):

- Lazy spawn: no worker process exists until the first setup/synthesis needs
  one. A not-running worker is healthy-idle, never an error.
- Lockstep: one outstanding command at a time; the manager's per-model lock
  serializes synthesis, and the proxy's own lock serializes the cleaner's
  ping/age with it.
- Deadlines are client-side: warm synthesis `synthesis_timeout_seconds`, the
  first synthesis after spawn and the spawn-readiness ping use
  `cold_start_timeout_seconds`, pings use `cuda_health_timeout_seconds`.
- Single-retry: a transport failure (timeout / EOF / dead process) kills the
  worker and retries the command exactly once after a fresh spawn; a second
  failure marks the adapter degraded (fast 503s) for `recovery_cooldown_seconds`.
  A worker-REPORTED error frame never triggers a respawn - the worker survived.
- EOF on the pipe (server death, worker exit, self-retirement) marks the proxy
  not-running; the next request respawns lazily.
"""
import base64
import json
import logging
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .adapter import (
    AudioResult,
    ModelCapabilities,
    ModelNotInitializedError,
    ModelNotReadyError,
    ModelSettings,
    SynthesisError,
    capabilities_from_manifest,
    load_manifest,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_WORKER = _REPO_ROOT / "framework" / "model_worker.py"
_EOF = "__EOF__"


class _WorkerGone(Exception):
    """Transport-level failure: the worker is gone or the pipe is broken."""


class WorkerCommandError(Exception):
    """The worker answered with an error frame (it survived; no respawn)."""

    def __init__(self, error_type: str, message: str):
        self.error_type = error_type
        self.message = message
        super().__init__(f"{error_type}: {message}")


def create_proxy_adapter(model_dir: Path, settings: ModelSettings) -> "ProxyAdapter":
    """Build a ProxyAdapter for a model repo declaring a subprocess runtime.

    Reads only model.yaml - no model code is imported or executed, so an
    absent venv still registers with a full capability view (synthesis then
    returns 501 until the user runs the model's own setup script).
    """
    manifest = load_manifest(model_dir)
    runtime = manifest.get("runtime") or {}
    mode = runtime.get("mode", "inproc")
    if mode != "subprocess":
        raise ValueError(f"runtime.mode {mode!r} is not 'subprocess'")

    python_exe = Path(runtime.get("python", ".venv/Scripts/python.exe"))
    if not python_exe.is_absolute():
        python_exe = model_dir / python_exe
    worker_override = runtime.get("worker")
    if worker_override:
        worker_script = Path(worker_override)
        if not worker_script.is_absolute():
            worker_script = model_dir / worker_override
    else:
        worker_script = _DEFAULT_WORKER

    initialized = python_exe.exists() and worker_script.exists()
    if not initialized:
        logger.warning(
            "[MEMORY] Model %r runtime not initialized (python=%s exists=%s); "
            "synthesis will return 501 until its setup script is run.",
            manifest.get("name", model_dir.name), python_exe,
            python_exe.exists())
    return ProxyAdapter(manifest, model_dir, settings,
                        python_exe=python_exe, worker_script=worker_script,
                        initialized=initialized)


class ProxyAdapter:
    """ModelAdapter implementation backed by a spawned worker process."""

    def __init__(self, manifest: Dict, model_dir: Path, settings: ModelSettings,
                 *, python_exe: Path, worker_script: Path,
                 initialized: bool = True):
        self.manifest = manifest
        self.name = manifest.get("name", model_dir.name)
        self.source = manifest.get("source", "")
        self.model_dir = model_dir
        self.settings = settings
        self._python_exe = python_exe
        self._worker_script = worker_script
        self._caps = capabilities_from_manifest(manifest)
        self._initialized = initialized
        self._lock = threading.RLock()
        self._proc: Optional[subprocess.Popen] = None
        self._running = False
        self._served_since_spawn = False
        self._degraded_until = 0.0
        self._last_success = 0.0
        self._next_id = 0
        self._responses: Optional[queue.Queue] = None

    # ---- lifecycle (driven by the ModelManager) ----

    def shutdown(self) -> None:
        with self._lock:
            if self._proc is not None and self._running:
                try:
                    self._roundtrip("shutdown", {}, deadline=10.0)
                except Exception as exc:
                    logger.info("[%s] shutdown command failed (%s); killing",
                                self.name, exc)
            self._kill()

    def is_loaded(self) -> bool:
        # Local tracking: a synthesis has been served since spawn / restart.
        return self._running and self._served_since_spawn

    def capabilities(self) -> ModelCapabilities:
        return self._caps

    def age_caches(self) -> None:
        """Age pass over the wire. Never spawns: an idle worker has nothing
        to age, and lazy spawn means aging must not create processes."""
        with self._lock:
            if not self._running:
                return
            try:
                self._roundtrip("age", {}, deadline=30.0)
            except Exception as exc:
                # Non-lethal: the next synthesis enforces recovery if this was
                # a hang. Aging must not tear down a worker on a soft failure.
                logger.warning("[%s] age pass failed: %s", self.name, exc)

    def cache_info(self) -> Dict:
        """Voice-cache visibility over the wire (never spawns)."""
        with self._lock:
            if not self._running:
                return {"running": False}
            try:
                return self._roundtrip("cache_info", {}, deadline=10.0)
            except Exception as exc:
                logger.warning("[%s] cache_info failed: %s", self.name, exc)
                return {"running": True, "error": str(exc)}

    def supported_speakers(self) -> Dict:
        """Complete built-in speaker catalogue over the wire.

        May load the owning checkpoint on first call - runs under the
        cold-start deadline via the single-retry ladder."""
        with self._lock:
            self._check_degraded()
            return self._guarded(
                lambda: self._dispatch_ready("speakers", {},
                                             self.settings.cold_start_timeout))

    def health(self, force: bool = False) -> bool:
        """Adapter-owned CUDA/worker health gate for the manager."""
        with self._lock:
            if not self._running:
                return True  # healthy-idle: nothing to probe (never spawns)
            now = time.time()
            if now < self._degraded_until:
                return False
            if not force and (now - self._last_success) < self.settings.cuda_check_cooldown:
                return True
            try:
                resp = self._roundtrip("ping", {},
                                       deadline=self.settings.cuda_health_timeout)
            except Exception as exc:
                # Hung worker: kill it now; the next request respawns with a
                # fresh CUDA context. Non-lethal for the probe itself.
                logger.warning("[%s] health ping failed (%s); worker killed",
                               self.name, exc)
                self._kill()
                return False
            if resp.get("cuda_ok", False):
                self._last_success = time.time()
                return True
            # Worker alive but CUDA unresponsive: retire it so the next use
            # starts with a fresh context.
            logger.warning("[%s] health ping reports CUDA unavailable; "
                           "worker killed", self.name)
            self._kill()
            return False

    def _check_degraded(self) -> None:
        now = time.time()
        if now < self._degraded_until:
            raise ModelNotReadyError(
                f"{self.name} is in recovery cooldown for another "
                f"{self._degraded_until - now:.0f}s; please retry")

    # ---- synthesis ----

    def synthesize_default(self, text: str, voice: str,
                           instruct: Optional[str] = None) -> AudioResult:
        return self._synthesize("synthesize_default",
                                {"text": text, "voice": voice,
                                 "instruct": instruct})

    def synthesize_clone(self, text: str, ref_audio_path: str,
                         ref_text: str = "",
                         instruct: Optional[str] = None) -> AudioResult:
        return self._synthesize("synthesize_clone",
                                {"text": text,
                                 "ref_audio_path": ref_audio_path,
                                 "ref_text": ref_text,
                                 "instruct": instruct})

    def synthesize_create(self, text: str, description: str) -> AudioResult:
        return self._synthesize("synthesize_create",
                                {"text": text, "description": description})

    def _synthesize(self, cmd: str, args: Dict) -> AudioResult:
        # First synthesis after spawn may include the model load; the warm
        # path gets the tighter budget (per plan SS12.2).
        deadline = (self.settings.cold_start_timeout
                    if not self._served_since_spawn
                    else self.settings.synthesis_timeout)
        result = self._send(cmd, args, deadline=deadline)
        self._served_since_spawn = True
        self._last_success = time.time()
        audio = np.frombuffer(base64.b64decode(result["audio_b64"]),
                              dtype=np.float32)
        return AudioResult(audio=audio,
                           sample_rate=int(result["sample_rate"]))

    # ---- wire plumbing ----

    def _send(self, cmd: str, args: Dict, deadline: float):
        """Send one command under the single-retry ladder (plan SS12.2).

        The first attempt includes a lazy spawn if needed. Any transport
        failure (spawn-readiness failure, command timeout, EOF, dead process)
        triggers exactly ONE respawn + retry; a second failure marks the
        adapter degraded for `recovery_cooldown_seconds`. Worker-reported
        error frames and absent-venv conditions never trigger a respawn.
        """
        with self._lock:
            self._check_degraded()
            return self._guarded(
                lambda: self._dispatch_ready(cmd, args, deadline))

    def _dispatch_ready(self, cmd: str, args: Dict, deadline: float):
        if not self._running:
            self._spawn()
        return self._roundtrip(cmd, args, deadline)

    def _guarded(self, operation):
        """Run `operation()` under the single-retry ladder: on transport
        failure kill the worker and re-run it once after a fresh spawn."""
        try:
            return operation()
        except (WorkerCommandError, ModelNotInitializedError):
            # The worker answered with an error (it survived), or the runtime
            # environment is missing: a retry cannot fix either.
            raise
        except Exception as first_exc:
            logger.warning("[%s] operation failed (%s); respawning worker once",
                           self.name, first_exc)
            self._kill()
            try:
                self._spawn()
                return operation()
            except ModelNotInitializedError:
                raise
            except Exception as retry_exc:
                # The re-initialized worker is also bad: assume a protracted
                # problem - retire it, then suppress retries for the cooldown.
                self._kill()
                self._degraded_until = (time.time()
                                        + self.settings.recovery_cooldown)
                logger.error("[%s] re-initialization failed (%s); "
                             "degraded for %.0fs", self.name, retry_exc,
                             self.settings.recovery_cooldown)
                raise ModelNotReadyError(
                    f"{self.name} is in recovery cooldown after a failed "
                    f"re-initialization: {retry_exc}") from retry_exc

    def _roundtrip(self, cmd: str, args: Dict, deadline: float) -> Dict:
        if self._proc is None or not self._running:
            raise _WorkerGone("worker is not running")
        self._next_id += 1
        rid = self._next_id
        frame = json.dumps({"id": rid, "cmd": cmd, "args": args}).encode("utf-8") + b"\n"
        try:
            self._proc.stdin.write(frame)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise _WorkerGone(f"pipe write failed: {exc}") from exc

        deadline_at = time.time() + deadline
        while True:
            remaining = deadline_at - time.time()
            if remaining <= 0:
                raise TimeoutError(f"{cmd} exceeded its {deadline:.0f}s deadline")
            try:
                item = self._responses.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                if self._proc is not None and self._proc.poll() is not None:
                    raise _WorkerGone(
                        f"worker exited (code {self._proc.poll()})")
                continue
            if item[0] == _EOF:
                raise _WorkerGone("worker closed the pipe (EOF)")
            if item[0] != rid:
                continue  # stale frame; lockstep means this should not happen
            _, payload = item
            if payload.get("ok"):
                return payload.get("result") or {}
            err = payload.get("error") or {}
            raise WorkerCommandError(err.get("type", "WorkerError"),
                                     err.get("message", ""))

    # ---- process management ----

    def _spawn(self) -> None:
        if self._proc is not None:
            self._kill()
        if not self._initialized:
            raise ModelNotInitializedError(self._init_message())
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        argv = [str(self._python_exe), str(self._worker_script),
                "--repo", str(self.model_dir),
                "--model-idle-timeout", str(self.settings.model_idle_timeout),
                "--prompt-ttl", str(self.settings.prompt_ttl),
                "--device", str(self.settings.device or "cuda"),
                "--cuda-health-timeout", str(self.settings.cuda_health_timeout)]
        if self.settings.enable_prefetch:
            argv.append("--enable-prefetch")
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        logger.info("[MEMORY] + spawning worker for %r (%s)", self.name,
                    self._python_exe)
        try:
            self._proc = subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=env, creationflags=creationflags)
        except OSError as exc:
            raise ModelNotInitializedError(
                self._init_message(extra=f"interpreter failed: {exc}")) from exc
        self._running = True
        self._served_since_spawn = False
        self._start_pipes()
        # Liveness is proven by the caller's first command response (setup
        # sends an explicit readiness ping; synthesis answers on its own).

    def _kill(self) -> None:
        proc = self._proc
        self._proc = None
        self._running = False
        self._served_since_spawn = False
        if proc is None:
            return
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass

    def _start_pipes(self) -> None:
        responses: "queue.Queue" = queue.Queue()
        self._responses = responses
        proc = self._proc

        def reader():
            try:
                for raw in proc.stdout:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line.decode("utf-8"))
                    except Exception:
                        logger.error("[%s] malformed protocol frame: %r",
                                     self.name, line[:200])
                        continue
                    responses.put((payload.get("id"), payload))
            except Exception as exc:
                logger.error("[%s] protocol reader failed: %s", self.name, exc)
            finally:
                responses.put((_EOF, None))

        def drain():
            log = logging.getLogger(f"worker.{self.name}")
            try:
                for raw in proc.stderr:
                    line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                    if line:
                        log.info(line)
            except Exception:
                pass

        threading.Thread(target=reader, daemon=True,
                         name=f"worker-reader-{self.name}").start()
        threading.Thread(target=drain, daemon=True,
                         name=f"worker-stderr-{self.name}").start()

    def _init_message(self, extra: str = "") -> str:
        missing = []
        if not self._python_exe.exists():
            missing.append(f"interpreter {self._python_exe}")
        if not self._worker_script.exists():
            missing.append(f"worker harness {self._worker_script}")
        detail = "; ".join(missing) if missing else "unknown cause"
        base = (f"Model {self.name!r} runtime is not initialized ({detail}). "
                f"Run the model's own setup script "
                f"(e.g. models/{self.model_dir.name}/setup_windows.bat "
                f"or setup.sh) per its README, then retry.")
        return f"{base} {extra}" if extra else base
