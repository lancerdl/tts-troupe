"""Persistence tests (Phase 2, decision 3a).

The REAL harness runs under the MODEL venv interpreter (torch present), so
the actual torch.save/torch.load round-trip is exercised end-to-end through
the manager. Tests are sequential by design (session-scoped worker; the
extraction counter is observed through relayed worker logs via caplog).
"""
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from framework import ModelManager, ModelSettings
from conftest import make_wire_hook_repo

_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def hook_env(tmp_path_factory):
    model_venv_python = (_REPO_ROOT / "models" / "faster-qwen3-tts"
                         / ".venv" / "Scripts" / "python.exe")
    if not model_venv_python.exists():
        pytest.skip("model venv not present - persistence tests need torch")
    ref_wav = _REPO_ROOT / "models" / "faster-qwen3-tts" / "ref_audio.wav"
    if not ref_wav.exists():
        pytest.skip("reference wav not present")

    models_dir = tmp_path_factory.mktemp("hook_models")
    repo = make_wire_hook_repo(models_dir, runtime_python=str(model_venv_python))
    manager = ModelManager(ModelSettings(
        models=["wire-hook-tts"],
        model_idle_timeout=60.0,
        synthesis_timeout=60.0,
        cold_start_timeout=280.0,
        recovery_cooldown=15.0,
        prompt_ttl=30.0,
    ), models_dir)
    manager.load()
    proxy = next(iter(manager._models.values()))
    env = SimpleNamespace(manager=manager, proxy=proxy, repo=repo,
                          ref_wav=str(ref_wav),
                          cache_dir=repo / "voice_cache")
    yield env
    manager.shutdown()


def _clone(env, text, ref_text="hello"):
    return env.manager.synthesize_clone(text=text,
                                        ref_audio_path=env.ref_wav,
                                        ref_text=ref_text)


def _pt_count(env):
    return len(list(env.cache_dir.glob("*.pt")))


def _extract_count(caplog):
    return sum(1 for r in caplog.records
               if "extracting voice clone prompt" in r.getMessage())


def _wait_extract(caplog, before, timeout: float = 3.0) -> int:
    """The extracting log races the response just like the disk-hit log."""
    deadline = time.time() + timeout
    while time.time() < deadline and _extract_count(caplog) == before:
        time.sleep(0.05)
    return _extract_count(caplog)


def _has_log(caplog, needle, timeout: float = 3.0) -> bool:
    """Worker log lines are relayed asynchronously by the proxy's stderr
    drain thread - poll briefly so a line emitted just before the response
    frame is observed reliably."""
    deadline = time.time() + timeout
    while True:
        if any(needle in r.getMessage() for r in caplog.records):
            return True
        if time.time() >= deadline:
            return False
        time.sleep(0.05)


def test_clone_persists_and_reuses(hook_env, caplog):
    caplog.set_level(logging.INFO)
    result = _clone(hook_env, "one")
    assert np.allclose(result.audio, [0.7, 0.8])
    assert result.sample_rate == 12000
    assert _pt_count(hook_env) == 1
    assert _extract_count(caplog) == 1

    # Same voice again: memory hit - no re-extraction.
    result2 = _clone(hook_env, "two")
    assert np.allclose(result2.audio, [0.7, 0.8])
    assert _extract_count(caplog) == 1

    status = hook_env.manager.cache_status()[0]
    assert status["memory_entries"] == 1
    assert status["persisted_files"] == 1
    assert status["persisted_voices"] == ["ref_audio"]


def test_worker_restart_uses_disk_cache(hook_env, caplog):
    caplog.set_level(logging.INFO)
    # Seed a DISTINCT entry in the shared cache directory first.
    before_seed = _pt_count(hook_env)
    _clone(hook_env, "seed", ref_text="restart-seed")
    assert _pt_count(hook_env) == before_seed + 1
    before = _extract_count(caplog)

    hook_env.proxy._kill()  # simulate a crash: memory lost, disk survives
    result = _clone(hook_env, "restart", ref_text="restart-seed")
    assert np.allclose(result.audio, [0.7, 0.8])
    assert _has_log(caplog, "loaded voice prompt from disk")
    assert _extract_count(caplog) == before  # no re-extraction
    assert _pt_count(hook_env) == before_seed + 1


def test_transcript_change_reclones(hook_env, caplog):
    caplog.set_level(logging.INFO)
    before_seed = _pt_count(hook_env)
    _clone(hook_env, "seed", ref_text="transcript-seed")  # entry A
    assert _pt_count(hook_env) == before_seed + 1
    before = _extract_count(caplog)
    _clone(hook_env, "changed", ref_text="different transcript")  # entry B
    assert _wait_extract(caplog, before) == before + 1  # new fingerprint
    assert _pt_count(hook_env) == before_seed + 2


def test_corrupt_cache_regenerates(hook_env, caplog):
    caplog.set_level(logging.INFO)
    ref_text = "corrupt-target"
    before = _extract_count(caplog)
    _clone(hook_env, "seed", ref_text=ref_text)  # create the entry
    assert _wait_extract(caplog, before) == before + 1

    # Corrupt exactly the file this fingerprint maps to.
    pts = sorted(hook_env.cache_dir.glob("*.pt"),
                 key=lambda p: p.stat().st_mtime)
    target = pts[-1]
    target.write_bytes(b"garbage")  # simulate a truncated write

    # A memory hit would shield the corrupt disk file - kill the worker so
    # the next clone reads from disk and hits the corruption.
    hook_env.proxy._kill()
    _clone(hook_env, "after", ref_text=ref_text)  # same fingerprint
    assert _wait_extract(caplog, before + 1) == before + 2
    assert _has_log(caplog, "corrupt cache file")
    assert target.exists() and target.stat().st_size > 7  # regenerated in place


def test_base_drop_clears_memory_but_not_disk(hook_env, caplog):
    caplog.set_level(logging.INFO)
    marker = hook_env.repo / "drop_base"
    marker.write_text("", encoding="utf-8")
    try:
        status = hook_env.manager.cache_status()[0]
        assert status["memory_entries"] >= 1
        hook_env.proxy.age_caches()  # worker: sweep + age_checkpoints -> ["base"]
        status = hook_env.manager.cache_status()[0]
        assert status["memory_entries"] == 0
        assert status["persisted_files"] >= 1  # disk survives the coupling
    finally:
        marker.unlink(missing_ok=True)

    # Next clone is a DISK hit: memory restored without re-extraction.
    before = _extract_count(caplog)
    result = _clone(hook_env, "post-drop")
    assert np.allclose(result.audio, [0.7, 0.8])
    assert _extract_count(caplog) == before
    status = hook_env.manager.cache_status()[0]
    assert status["memory_entries"] == 1


def test_version_bump_new_key(hook_env, caplog):
    caplog.set_level(logging.INFO)
    # Seed the pre-bump key: a distinct fingerprint in the shared cache.
    before_seed = _pt_count(hook_env)
    _clone(hook_env, "seed", ref_text="pre-bump")
    assert _pt_count(hook_env) == before_seed + 1
    before = _extract_count(caplog)

    manifest_path = hook_env.repo / "model.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["capabilities"]["voice_clone"]["prompt_cache"]["versions"]["format"] = "2"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    hook_env.proxy._kill()  # the worker re-reads the manifest at spawn
    _clone(hook_env, "bumped", ref_text="pre-bump")
    assert _wait_extract(caplog, before) == before + 1  # new key -> re-extract
    assert _pt_count(hook_env) == before_seed + 2  # old-format file remains