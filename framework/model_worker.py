"""
Generic worker harness for subprocess-isolated TTS models.

Runs INSIDE a model repository's venv, spawned by ``models/proxy_adapter.py``
with PYTHONPATH pointing at the server repo root. Owns the model adapter for
its repo and serves a strict lockstep JSON-line protocol on stdin/stdout:

  request : {"id": N, "cmd": "...", "args": {...}}
  response: {"id": N, "ok": true,  "result": {...}}
            {"id": N, "ok": false, "error": {"type": "...", "message": "..."}}

stdout is reserved exclusively for protocol frames; ``sys.stdout`` is
redirected to a stderr sink at startup so no library ``print`` can corrupt the
stream. All logs (including the adapter's) flow on stderr for the proxy to
re-emit into the server's log partitioning.

Commands (deadlines are enforced by the proxy, client side - this process is
policy-free except for the aging settings passed on argv):

  ping                 tiered CUDA health probe (models.cuda_utils)
  synthesize_preset    {text, voice, instruct}
  synthesize_clone     {text, ref_audio_path, ref_text, instruct}
  synthesize_create    {text, description}
  age                  run adapter aging (voice TTL sweep); reports ``retire``
                       when the adapter has nothing loaded and has been idle
                       past the model idle timeout (self-retirement)
  shutdown             acknowledge and exit

EOF on stdin = go home: the loop simply ends and the process exits with no
cleanup calls. Process death reclaims VRAM; nothing is preserved that is not
already on disk.
"""
import argparse
import base64
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

# Stdlib-only at import time (torch is imported lazily inside cuda_utils).
from framework.cuda_utils import is_cuda_healthy

logger = logging.getLogger("worker")

# Commands the manager's cleaner sends periodically; they do not count as
# activity for the worker's idle clock (self-retirement would be unreachable
# if the 30s heartbeat reset it).
MAINTENANCE_COMMANDS = frozenset({"age", "ping", "cache_info"})


class _StderrSink:
    """Minimal text-stream stand-in for sys.stdout that writes to stderr."""

    encoding = "utf-8"
    errors = "replace"

    def write(self, s: str) -> int:
        return sys.stderr.write(s)

    def writelines(self, lines) -> None:
        for line in lines:
            sys.stderr.write(line)

    def flush(self) -> None:
        sys.stderr.flush()

    def isatty(self) -> bool:
        return False


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="TTS model worker harness")
    parser.add_argument("--repo", required=True,
                        help="Model repository directory (contains model.yaml)")
    parser.add_argument("--model-idle-timeout", type=float, default=60.0,
                        help="Minutes of idleness before the worker may self-retire")
    parser.add_argument("--prompt-ttl", type=float, default=30.0,
                        help="Minutes of voice-prompt TTL (Phase 2: harness-owned cache)")
    parser.add_argument("--device", default="cuda",
                        help="Device hint for the CUDA health probe")
    parser.add_argument("--cuda-health-timeout", type=float, default=2.0,
                        help="Seconds allowed for the CUDA health probe")
    parser.add_argument("--enable-prefetch", action="store_true",
                        help="Verify/download configured checkpoints at worker startup")
    return parser.parse_args(argv)


def _verify_models(model_ids, logger) -> None:
    """Best-effort HF cache verification (Phase 2 step 8): never blocks the
    spawn - an offline machine with a complete cache boots fine."""
    ids = sorted(set(mid for mid in model_ids if mid))
    if not ids:
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.warning("huggingface_hub unavailable; skipping model verification")
        return
    for repo_id in ids:
        try:
            snapshot_download(repo_id)
            logger.info("model verified in HF cache: %s", repo_id)
        except Exception as exc:
            logger.warning("model verification failed for %s (best-effort): %s",
                           repo_id, exc)


def _respond(proto, payload: dict) -> None:
    proto.write(json.dumps(payload).encode("utf-8") + b"\n")
    proto.flush()


def _encode_audio(result) -> dict:
    import numpy as np

    if isinstance(result, tuple):  # plain (audio, sample_rate) adapter return
        audio, sample_rate = result[0], result[1]
    else:
        audio, sample_rate = result.audio, result.sample_rate
    if hasattr(audio, "detach"):   # torch tensor, possibly device-resident
        audio = audio.detach().to("cpu").float().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    return {
        "audio_b64": base64.b64encode(audio.tobytes()).decode("ascii"),
        "sample_rate": int(sample_rate),
        "dtype": "float32",
    }


def _dispatch(cmd: str, args: dict, adapter, cli, last_activity: float,
              cache, hook) -> dict:
    if cmd == "ping":
        ok = is_cuda_healthy(device=cli.device,
                             timeout_seconds=cli.cuda_health_timeout)
        return {"running": True, "cuda_ok": bool(ok)}

    if cmd == "synthesize_preset":
        return _encode_audio(adapter.synthesize_preset(
            text=args["text"], voice=args["voice"],
            instruct=args.get("instruct")))

    if cmd == "synthesize_clone":
        if cache is None or hook is None:
            return _encode_audio(adapter.synthesize_clone(
                text=args["text"], ref_audio_path=args["ref_audio_path"],
                ref_text=args.get("ref_text", ""),
                instruct=args.get("instruct")))
        return _encode_audio(_synthesize_clone_cached(
            args, adapter, cache, hook))

    if cmd == "synthesize_create":
        return _encode_audio(adapter.synthesize_create(
            text=args["text"], description=args["description"]))

    if cmd == "speakers":
        getter = getattr(adapter.obj, "supported_speakers", None)
        return {"speakers": (getter() or []) if callable(getter) else []}

    if cmd == "cache_info":
        info: dict = {}
        if cache is not None:
            info.update(cache.stats())
        info["adapter_loaded"] = bool(adapter.is_loaded())
        return info

    if cmd == "age":
        dropped = []
        if cache is not None:
            cache.sweep()
        age_checkpoints = getattr(adapter.obj, "age_checkpoints", None)
        if callable(age_checkpoints):
            dropped = age_checkpoints(cli.model_idle_timeout * 60) or []
        if cache is not None and hook and hook.get("checkpoint") in dropped:
            # The checkpoint that produced the cached prompts left VRAM:
            # the in-memory mirror must drop with it (disk tensors survive).
            cache.clear_memory()
        age_caches = getattr(adapter, "age_caches", None)
        if callable(age_caches):
            age_caches()
        idle_seconds = time.time() - last_activity
        cache_idle = cache is None or not cache.stats()["memory_entries"]
        retire = (not adapter.is_loaded() and cache_idle
                  and idle_seconds > cli.model_idle_timeout * 60)
        return {"retire": bool(retire), "checkpoints_dropped": dropped}

    if cmd == "shutdown":
        return {}

    raise KeyError(f"unknown command {cmd!r}")


def _synthesize_clone_cached(args: dict, adapter, cache, hook) -> dict:
    text = args["text"]
    ref_audio = args["ref_audio_path"]
    ref_text = args.get("ref_text", "")
    voice_name = Path(ref_audio).stem

    extract = getattr(adapter.obj, hook["extract"])
    synthesize = getattr(adapter.obj, hook["synthesize"])

    prompt = cache.get(ref_audio, ref_text, voice_name)
    if prompt is None:
        logger.info("extracting voice clone prompt: %s", voice_name)
        prompt = extract(ref_audio, ref_text)
        cache.put(ref_audio, ref_text, voice_name, prompt)
    resolution = adapter.resolve_instruct("voice_clone", args.get("instruct"))
    kwargs = {"text": text, "prompt": prompt,
              "ref_audio_path": ref_audio, "ref_text": ref_text}
    if resolution.value is not None:
        kwargs["instruct"] = resolution.value
    return synthesize(**kwargs)


def main(argv=None) -> int:
    cli = _parse_args(argv)

    # Reserve the real stdout for the protocol before anything else can print.
    proto = sys.stdout.buffer
    sys.stdout = _StderrSink()

    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO,
        format="%(asctime)s [worker] %(levelname)s %(name)s: %(message)s")

    # framework.adapter pulls in the repo adapter (and therefore torch) - only
    # imported after argv parsing and the stdout redirect are in place.
    from framework.adapter import create_adapter, load_manifest
    from framework.voice_cache import PersistentPromptCache

    repo = Path(cli.repo).resolve()
    manifest = load_manifest(repo)
    hook = (manifest.get("capabilities", {}).get("voice_clone", {}) or {}
            ).get("prompt_cache")
    adapter = create_adapter(repo)
    logger.info("worker ready: model %r from %s", adapter.name, repo)

    models_map = dict((getattr(adapter, "model_config", {}) or {}).get("models", {})
                      or {})
    cache = None
    if hook:
        scope_id = hashlib.sha256(
            json.dumps(sorted(models_map.items()), sort_keys=True)
            .encode("utf-8")).hexdigest()[:16]
        versions = hook.get("versions", {}) or {}
        cache = PersistentPromptCache(
            cache_dir=repo / "voice_cache",
            scope_id=f"{adapter.name}:{scope_id}",
            adapter_version=str(versions.get("adapter", "0")),
            format_version=str(versions.get("format", "1")),
            ttl_seconds=cli.prompt_ttl * 60.0,
        )
        logger.info("[CACHE] voice prompt cache active at %s (ttl %.0f min)",
                    cache.cache_dir, cli.prompt_ttl)
    else:
        logger.info("no prompt_cache declared: voice clone runs without caching")

    if cli.enable_prefetch:
        _verify_models(models_map.values(), logger)

    # Maintenance probes (age/ping/cache_info) are sent every ~30s by the
    # manager's cleaner; they must not count as activity, or the worker's
    # idle clock never exceeds the self-retirement threshold.
    _maintenance_cmds = MAINTENANCE_COMMANDS
    last_activity = time.time()
    for raw in sys.stdin.buffer:
        line = raw.strip()
        if not line:
            continue
        try:
            req = json.loads(line.decode("utf-8"))
            rid = req.get("id")
            cmd = req.get("cmd", "")
        except Exception as exc:  # malformed frame: report, keep serving
            _respond(proto, {"id": None, "ok": False,
                             "error": {"type": "ProtocolError",
                                       "message": f"bad request frame: {exc}"}})
            continue
        if cmd not in _maintenance_cmds:
            last_activity = time.time()

        try:
            result = _dispatch(cmd, req.get("args") or {}, adapter, cli,
                               last_activity, cache, hook)
        except Exception as exc:
            # A failed synthesis must not kill the worker: report and keep
            # serving. Only a hang or crash ends this process.
            logger.error("command %s failed: %s", cmd, exc)
            _respond(proto, {"id": rid, "ok": False,
                             "error": {"type": type(exc).__name__,
                                       "message": str(exc)}})
            continue

        _respond(proto, {"id": rid, "ok": True, "result": result})

        if cmd == "shutdown":
            break
        if cmd == "age" and result.get("retire"):
            logger.info("self-retiring: nothing loaded, idle past %.0f min",
                        cli.model_idle_timeout)
            break

    # EOF or shutdown: go home. No cleanup calls - process death reclaims VRAM.
    return 0


if __name__ == "__main__":
    sys.exit(main())
