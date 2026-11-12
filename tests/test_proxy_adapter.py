"""Subprocess-adapter tests: the server/worker relationship (Phase 1).

These run the REAL models/model_worker.py harness against fake model repos in
a torch-free interpreter - which doubles as proof that the proxy path imports
no torch anywhere (an import would raise ImportError and fail the suite).
"""
import time

import numpy as np
import pytest

from framework import ModelManager, ModelSettings
from framework.adapter import ModelNotInitializedError, ModelNotReadyError

_HERE = __file__


def _manager(models_dir, **overrides):
    names = [p.name for p in models_dir.iterdir() if p.is_dir()]
    settings_kwargs = dict(
        models=names,
        model_idle_timeout=60.0,
        synthesis_timeout=10.0,
        cold_start_timeout=30.0,
        recovery_cooldown=5.0,
    )
    settings_kwargs.update(overrides)
    manager = ModelManager(ModelSettings(**settings_kwargs), models_dir)
    manager.load()
    return manager


def _proxy(manager):
    return next(iter(manager._models.values()))


def test_wire_synthesis_end_to_end(wire_models_dir):
    manager = _manager(wire_models_dir)
    proxy = _proxy(manager)

    result = manager.synthesize_preset(voice="alloy", text="hello")
    assert isinstance(result.audio, np.ndarray)
    assert np.allclose(result.audio, [0.1, 0.2, 0.3])
    assert result.sample_rate == 24000
    assert proxy.is_loaded() is True  # local tracking after a served request

    clone = manager.synthesize_clone(text="hey", ref_audio_path="ref.wav",
                                     ref_text="hi")
    assert np.allclose(clone.audio, [0.4, 0.5])
    assert clone.sample_rate == 16000

    manager.shutdown()
    assert proxy._proc is None  # worker reaped on shutdown
    assert proxy.is_loaded() is False


def test_worker_respawns_after_death(wire_models_dir):
    manager = _manager(wire_models_dir)
    proxy = _proxy(manager)

    manager.synthesize_preset(voice="alloy", text="first")
    proc = proxy._proc
    proc.kill()  # simulate a worker crash mid-session
    proc.wait(timeout=5)
    # The proxy still believes it is running until a command detects the EOF.

    result = manager.synthesize_preset(voice="alloy", text="second")
    assert np.allclose(result.audio, [0.1, 0.2, 0.3])
    assert proxy._proc is not None and proxy._proc is not proc
    manager.shutdown()


def test_absent_venv_registers_but_returns_501_type(tmp_path, wire_repo_factory):
    repo = wire_repo_factory(runtime_python=str(tmp_path / "nope" / "python.exe"))
    manager = _manager(tmp_path / "models")

    caps = manager.capabilities()
    assert caps[0].voice_preset is True  # capability view works without a venv

    with pytest.raises(ModelNotInitializedError):
        manager.synthesize_preset(voice="alloy", text="hi")
    manager.shutdown()


def test_age_without_worker_never_spawns(wire_models_dir):
    manager = _manager(wire_models_dir)
    proxy = _proxy(manager)
    proxy.age_caches()  # must be a no-op, not a lazy spawn
    assert proxy._proc is None
    manager.shutdown()


def test_health_without_worker_is_healthy_idle(wire_models_dir):
    manager = _manager(wire_models_dir)
    proxy = _proxy(manager)
    assert proxy.health(force=True) is True
    assert proxy._proc is None  # health probe must not spawn either
    manager.shutdown()


def test_shutdown_is_transport_safe_on_dead_worker(wire_models_dir):
    manager = _manager(wire_models_dir)
    proxy = _proxy(manager)
    manager.synthesize_preset(voice="alloy", text="hi")
    proc = proxy._proc
    proc.kill()
    proc.wait(timeout=5)
    manager.shutdown()  # must not raise even though the worker is gone
    assert proxy._proc is None


def test_hang_worker_single_retry_then_degraded(wire_repo_factory):
    import pathlib
    hang_worker = pathlib.Path(_HERE).parent / "hang_worker.py"
    repo = wire_repo_factory(name="hang-tts",
                             runtime_worker=str(hang_worker))
    manager = _manager(repo.parent,
                       synthesis_timeout=1.0,
                       cold_start_timeout=2.0,
                       recovery_cooldown=5.0)

    with pytest.raises(ModelNotReadyError):
        manager.synthesize_preset(voice="alloy", text="hi")

    proxy = _proxy(manager)
    assert proxy._proc is None  # failed re-init leaves nothing running
    assert time.time() < proxy._degraded_until  # cooldown active

    # During the cooldown, requests fail fast (503 path) without attempts.
    with pytest.raises(ModelNotReadyError):
        manager.synthesize_preset(voice="alloy", text="hi again")

    manager.shutdown()


def test_hang_worker_recovers_after_cooldown(wire_repo_factory):
    import pathlib
    hang_worker = pathlib.Path(_HERE).parent / "hang_worker.py"
    repo = wire_repo_factory(name="hang2-tts",
                             runtime_worker=str(hang_worker))
    manager = _manager(repo.parent,
                       synthesis_timeout=0.5,
                       cold_start_timeout=1.0,
                       recovery_cooldown=1.0)

    with pytest.raises(ModelNotReadyError):
        manager.synthesize_preset(voice="alloy", text="hi")

    # Simulate the cooldown elapsing (real policy: wall-clock wait).
    proxy = _proxy(manager)
    proxy._degraded_until = time.time() - 0.01

    # Still hangs -> one attempt again -> degraded again. The ladder repeats.
    with pytest.raises(ModelNotReadyError):
        manager.synthesize_preset(voice="alloy", text="hi")
    manager.shutdown()