"""
Persistent, model-scoped cache for voice-clone prompt payloads.

Owns the full caching lifecycle described in
.cline/voice_cache_persistence_plan.md:

- fingerprinting: content hashes (sha256 of the reference .wav bytes and the
  reference transcript) behind a stat memo - (size, mtime_ns) is only a
  prefilter; content is authoritative
- versioned keys: sha256(model_id | adapter_version | format_version |
  fingerprint) - a model or extraction-code change self-invalidates
- disk layer: one self-validating ``<key>.pt`` per voice inside the model
  repo, written atomically (tmp + os.replace) with CPU tensors, loaded with
  torch.load(weights_only=True); a corrupt file is deleted and regenerated
- memory layer: a TTL'd hot mirror whose lifetime is a subset of checkpoint
  residency (the harness clears it on unload/checkpoint aging)
- index.json: LRU/eviction metadata only - never required for correctness

This module runs inside the model worker (model venv, torch present). It is
never imported by the server process.
"""
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:  # pragma: no cover - server-side imports never land here
    torch = None


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_cpu(obj: Any) -> Any:
    """Recursively move tensors to CPU so nothing device-resident is saved."""
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu")
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_to_cpu(v) for v in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    return obj


class PersistentPromptCache:
    """Generic persistent cache for clone-prompt payloads (model-agnostic)."""

    def __init__(self, cache_dir: Path, scope_id: str, adapter_version: str,
                 format_version: str, ttl_seconds: float):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.scope_id = scope_id
        self.adapter_version = adapter_version
        self.format_version = format_version
        self.ttl_seconds = ttl_seconds
        self._memory: Dict[str, Dict[str, Any]] = {}
        self._memo: Dict[str, tuple] = {}  # path -> (size, mtime_ns, sha256)
        self._lock = threading.Lock()
        self._index_path = self.cache_dir / "index.json"
        self._index: Dict[str, Dict[str, Any]] = self._load_index()

    # ---- fingerprinting ----

    def _file_hash(self, path: str) -> Optional[str]:
        try:
            st = os.stat(path)
        except OSError:
            return None
        stat_key = (st.st_size, st.st_mtime_ns)
        memoized = self._memo.get(path)
        if memoized and memoized[0] == stat_key:
            return memoized[1]
        digest = _sha256_file(path)
        self._memo[path] = (stat_key, digest)
        return digest

    def fingerprint(self, wav_path: str, ref_text: str) -> Optional[str]:
        """Content-authoritative fingerprint; None when the .wav is unreadable."""
        wav_hash = self._file_hash(wav_path)
        if wav_hash is None:
            return None
        text_hash = hashlib.sha256(ref_text.encode("utf-8")).hexdigest()
        return f"{wav_hash}|{text_hash}"

    def cache_key(self, wav_path: str, ref_text: str) -> Optional[str]:
        fingerprint = self.fingerprint(wav_path, ref_text)
        if fingerprint is None:
            return None
        scope = "|".join([self.scope_id, self.adapter_version,
                          self.format_version, fingerprint])
        return hashlib.sha256(scope.encode("utf-8")).hexdigest()

    # ---- public API ----

    def get(self, wav_path: str, ref_text: str,
            voice_name: str = "") -> Optional[Any]:
        """Memory hit, then disk hit (CPU tensors); None = cache miss."""
        key = self.cache_key(wav_path, ref_text)
        if key is None:
            return None
        now = time.time()
        with self._lock:
            entry = self._memory.get(key)
            if entry is not None:
                entry["last_accessed"] = now
                self._touch_index(key, entry.get("voice_name", voice_name), now)
                return entry["payload"]

        path = self.cache_dir / f"{key}.pt"
        if not path.exists():
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            # Truncated/corrupt file (e.g. crash mid-write): delete and
            # regenerate. Worst case is one re-extraction.
            logger.warning("[CACHE] corrupt cache file %s (%s); regenerating",
                           path.name, exc)
            try:
                path.unlink()
            except OSError:
                pass
            with self._lock:
                self._index.pop(key, None)
                self._save_index()
            return None
        with self._lock:
            self._memory[key] = {"payload": payload, "last_accessed": now,
                                 "voice_name": voice_name}
            self._touch_index(key, voice_name, now)
        logger.info("[CACHE] loaded voice prompt from disk: %s", voice_name)
        return payload

    def put(self, wav_path: str, ref_text: str, voice_name: str,
            payload: Any) -> None:
        key = self.cache_key(wav_path, ref_text)
        if key is None:
            return
        cpu_payload = _to_cpu(payload)
        tmp = self.cache_dir / f"{key}.tmp"
        final = self.cache_dir / f"{key}.pt"
        try:
            torch.save(cpu_payload, tmp)
            os.replace(tmp, final)
        except Exception as exc:
            logger.warning("[CACHE] persisting voice prompt failed: %s", exc)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            return
        now = time.time()
        with self._lock:
            self._memory[key] = {"payload": payload, "last_accessed": now,
                                 "voice_name": voice_name}
            self._touch_index(key, voice_name, now, size=final.stat().st_size)
        logger.info("[CACHE] persisted voice prompt: %s (%s)",
                    voice_name, final.name)

    def sweep(self) -> int:
        """TTL sweep: evict expired MEMORY entries (disk files persist)."""
        now = time.time()
        with self._lock:
            expired = [k for k, e in self._memory.items()
                       if now - e["last_accessed"] > self.ttl_seconds]
            for k in expired:
                self._memory.pop(k, None)
        if expired:
            logger.info("[CACHE] swept %d expired voice prompt(s) from memory",
                        len(expired))
        return len(expired)

    def clear_memory(self) -> None:
        """Drop all memory entries (unload/checkpoint aging); disk survives."""
        with self._lock:
            count = len(self._memory)
            self._memory.clear()
        if count:
            logger.info("[CACHE] cleared %d voice prompt(s) from memory", count)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            memory_entries = len(self._memory)
            voices = sorted({e.get("voice_name", "")
                             for e in self._memory.values()
                             if e.get("voice_name")})
        files = 0
        disk_bytes = 0
        persisted_voices = []
        for entry in self._index.values():
            files += 1
            disk_bytes += entry.get("size_bytes", 0)
            name = entry.get("voice_name", "")
            if name:
                persisted_voices.append(name)
        return {
            "memory_entries": memory_entries,
            "memory_voices": sorted(voices),
            "persisted_files": files,
            "persisted_voices": sorted(persisted_voices),
            "disk_bytes": disk_bytes,
        }

    # ---- index (LRU metadata; never required for correctness) ----

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        try:
            with open(self._index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_index(self) -> None:
        tmp = self._index_path.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._index, f)
            os.replace(tmp, self._index_path)
        except Exception as exc:
            logger.warning("[CACHE] index update failed: %s", exc)

    def _touch_index(self, key: str, voice_name: str, now: float,
                     size: int = None) -> None:
        entry = self._index.get(key) or {}
        entry.update({
            "voice_name": voice_name or entry.get("voice_name", ""),
            "model_id": self.scope_id,
            "adapter_version": self.adapter_version,
            "last_used": now,
            "size_bytes": size if size is not None else entry.get("size_bytes", 0),
        })
        entry.setdefault("created", now)
        self._index[key] = entry
        self._save_index()