"""Manager-layer tests (Phase 2: subprocess-only runtime).

The manager requires proxy adapters, so these tests route
`create_proxy_adapter` to in-process MappedAdapters wrapped with the
proxy-shaped surface (the health hook) via a monkeypatch seam - manager
logic is fully exercised without spawning subprocesses.
"""
import time

import numpy as np
import pytest

from framework import ModelManager, ModelSettings
from framework.adapter import (
    AudioResult,
    NoModelForCapabilityError,
    SynthesisError,
    create_adapter,
)


class _ProxyShape:
    """Wraps an in-process MappedAdapter with the proxy-shaped surface the
    manager expects (the health hook), so tests run without subprocesses."""

    def __init__(self, mapped):
        self._m = mapped
        self.shutdown_count = 0

    @property
    def name(self):
        return self._m.name

    @property
    def obj(self):
        return self._m.obj

    def capabilities(self):
        return self._m.capabilities()

    def shutdown(self):
        self.shutdown_count += 1
        return self._m.shutdown()

    def age_caches(self):
        return self._m.age_caches()

    def health(self, force: bool = False) -> bool:
        return True

    def synthesize_preset(self, text, voice, instruct=None):
        return self._m.synthesize_preset(text=text, voice=voice,
                                          instruct=instruct)

    def synthesize_clone(self, text, ref_audio_path, ref_text="",
                         instruct=None):
        return self._m.synthesize_clone(text=text,
                                        ref_audio_path=ref_audio_path,
                                        ref_text=ref_text,
                                        instruct=instruct)

    def synthesize_create(self, text, description):
        return self._m.synthesize_create(text=text, description=description)


def _manager(models_dir, monkeypatch, **overrides):
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

    import framework.model_manager as mm

    def fake_create(model_dir, settings):
        return _ProxyShape(create_adapter(model_dir))

    monkeypatch.setattr(mm, "create_proxy_adapter", fake_create)
    manager.load()
    return manager


def _fake(manager):
    return next(iter(manager._models.values())).obj


def _adapter(manager):
    return next(iter(manager._models.values()))


# ---- manifest dispatch ----

def test_capabilities_reported(fake_models_dir, monkeypatch):
    manager = _manager(fake_models_dir, monkeypatch)
    caps = _adapter(manager).capabilities()
    assert caps.voice_preset is True
    assert caps.presets == ("alloy", "echo")
    assert caps.voice_clone is True
    assert caps.voice_create is True
    manager.shutdown()


def test_dispatch_default_resolves_args(fake_models_dir, monkeypatch):
    manager = _manager(fake_models_dir, monkeypatch)
    result = manager.synthesize_preset(voice="echo", text="hello")
    fake = _fake(manager)
    assert fake.calls == [("gen_preset", "hello", "echo", 0.9, "fast")]
    assert isinstance(result, AudioResult)
    assert result.audio == "fake-audio"
    assert result.sample_rate == 24000
    manager.shutdown()


def test_dispatch_create_routes_through_instance(fake_models_dir, monkeypatch):
    manager = _manager(fake_models_dir, monkeypatch)
    result = manager.synthesize_create(text="warm", description="deep calm voice")
    fake = _fake(manager)
    assert ("instance:gen_create", "warm", "deep calm voice") in fake.calls
    assert isinstance(result, AudioResult)
    assert (result.audio == np.array([0.5, 0.5])).all()
    assert result.sample_rate == 11025
    manager.shutdown()


def test_dispatch_clone_is_native(fake_models_dir, monkeypatch):
    manager = _manager(fake_models_dir, monkeypatch)
    result = manager.synthesize_clone(text="hey", ref_audio_path="ref.wav", ref_text="hi")
    fake = _fake(manager)
    assert ("synthesize_clone", "hey", "ref.wav", "hi", None) in fake.calls
    assert result == ("clone-audio", 16000)
    manager.shutdown()


def test_minimal_manifest_dispatch(minimal_models_dir, monkeypatch):
    manager = _manager(minimal_models_dir, monkeypatch)
    result = manager.synthesize_preset(voice="alloy", text="hi")
    fake = _fake(manager)
    assert fake.calls == [("gen_preset", "hi", "alloy", None, None)]
    assert result.audio == "fake-audio"
    assert result.sample_rate == 24000
    manager.shutdown()


def test_no_model_for_capability(minimal_models_dir, monkeypatch):
    manager = _manager(minimal_models_dir, monkeypatch)
    with pytest.raises(NoModelForCapabilityError):
        manager.synthesize_create(text="x", description="y")
    manager.shutdown()


# ---- lifecycle ----

def test_unload_then_resynthesize(fake_models_dir, monkeypatch):
    manager = _manager(fake_models_dir, monkeypatch)
    fake = _fake(manager)

    manager.synthesize_preset(voice="alloy", text="hi")

    with manager._manager_lock:
        manager.unload_model("fake-tts")
    assert fake._loaded is False
    assert fake.shutdown_count == 1

    # The next synthesis runs again through the (unchanged) adapter.
    manager.synthesize_preset(voice="alloy", text="hi2")
    assert fake.calls[-1] == ("gen_preset", "hi2", "alloy", 0.9, "fast")
    manager.shutdown()


def test_age_models_unloads_idle(fake_models_dir, monkeypatch):
    manager = _manager(fake_models_dir, monkeypatch)
    fake = _fake(manager)
    manager.synthesize_preset(voice="alloy", text="hi")

    # Simulate idleness beyond the 60-minute timeout.
    manager._last_accessed["fake-tts"] = time.time() - 3601
    assert manager.age_models() == 1
    assert _adapter(manager).shutdown_count == 1
    assert fake.shutdown_count == 1
    manager.shutdown()


def test_shutdown_is_idempotent(fake_models_dir, monkeypatch):
    manager = _manager(fake_models_dir, monkeypatch)
    manager.synthesize_preset(voice="alloy", text="hi")

    manager.shutdown()
    first_shutdowns = _adapter(manager).shutdown_count
    assert first_shutdowns == 1
    assert manager._shutdown_completed is True

    manager.shutdown()
    assert _adapter(manager).shutdown_count == first_shutdowns  # no-op repeat
    manager._cleaner_thread.join(timeout=5)
    assert not manager._cleaner_thread.is_alive()


def test_declared_model_missing_is_skipped(tmp_path):
    settings = ModelSettings(models=["nonexistent-tts"])
    manager = ModelManager(settings, tmp_path / "models")
    manager.load()  # logs a warning, registers nothing
    assert manager._models == {}
    manager.shutdown()


def test_missing_model_yaml_is_skipped(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "empty-tts").mkdir()
    settings = ModelSettings(models=["empty-tts"])
    manager = ModelManager(settings, models_dir)
    manager.load()
    assert manager._models == {}
    manager.shutdown()


def test_missing_runtime_block_is_rejected(tmp_path):
    # Subprocess is the only runtime: a model.yaml without a runtime block
    # fails registration (the model is skipped with an error log).
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "noruntime-tts").mkdir()
    (models_dir / "noruntime-tts" / "model.yaml").write_text(
        "name: noruntime-tts\nmodule: adapter\n", encoding="utf-8")
    settings = ModelSettings(models=["noruntime-tts"])
    manager = ModelManager(settings, models_dir)
    manager.load()
    assert manager._models == {}
    manager.shutdown()


# ---- error handling ----

def test_synthesis_error_wraps_model_failure(broken_models_dir, monkeypatch):
    manager = _manager(broken_models_dir, monkeypatch)
    with pytest.raises(SynthesisError) as excinfo:
        manager.synthesize_preset(voice="alloy", text="hi")
    assert excinfo.value.model_name == "broken-tts"
    assert isinstance(excinfo.value.cause, RuntimeError)
    assert "boom" in str(excinfo.value)
    manager.shutdown()

# ---- model resolution (strict model parameter / aliases) ----

def _aliased_manager(tmp_path, monkeypatch, aliases=("qwen3", "qwen")):
    import copy
    from conftest import FAKE_MANIFEST, make_model_repo
    manifest = copy.deepcopy(FAKE_MANIFEST)
    manifest["aliases"] = list(aliases)
    make_model_repo(tmp_path, "fake-tts", manifest, config={"pitch": 0.9})
    return _manager(tmp_path, monkeypatch)


def test_resolve_model_name_and_aliases(tmp_path, monkeypatch):
    manager = _aliased_manager(tmp_path, monkeypatch)
    assert manager.resolve_model("fake-tts") == "fake-tts"
    assert manager.resolve_model("FAKE-TTS") == "fake-tts"   # name, case-insensitive
    assert manager.resolve_model("qwen3") == "fake-tts"      # alias
    assert manager.resolve_model(" QWEN ") == "fake-tts"     # alias, case+whitespace
    assert manager.resolve_model("nope") is None
    assert manager.resolve_model("") is None
    manager.shutdown()


def test_model_names_lists_aliases(tmp_path, monkeypatch):
    manager = _aliased_manager(tmp_path, monkeypatch)
    assert manager.model_names() == ["fake-tts (aliases: qwen3, qwen)"]
    manager.shutdown()


def test_presets_by_model(fake_models_dir, monkeypatch):
    manager = _manager(fake_models_dir, monkeypatch)
    assert manager.presets_by_model() == {"fake-tts": ["alloy", "echo"]}
    manager.shutdown()


def test_synthesize_pinned_to_model(tmp_path, monkeypatch):
    manager = _aliased_manager(tmp_path, monkeypatch)
    manager.synthesize_preset(voice="alloy", text="hi", model="qwen3")
    fake = _fake(manager)
    assert fake.calls == [("gen_preset", "hi", "alloy", 0.9, "fast")]
    manager.shutdown()


def test_synthesize_pinned_rejects_missing_capability(tmp_path, monkeypatch):
    import copy
    from conftest import FAKE_MANIFEST, MINIMAL_MANIFEST, make_model_repo
    make_model_repo(tmp_path, "fake-tts", copy.deepcopy(FAKE_MANIFEST))
    make_model_repo(tmp_path, "minimal-tts", copy.deepcopy(MINIMAL_MANIFEST))
    manager = _manager(tmp_path, monkeypatch)
    # minimal-tts has no voice_clone capability: pinning must reject it...
    with pytest.raises(NoModelForCapabilityError):
        manager.synthesize_clone(text="x", ref_audio_path="a.wav", model="minimal-tts")
    # ...while unpinned routing still finds the capable adapter.
    manager.synthesize_clone(text="x", ref_audio_path="a.wav")
    fake = _fake(manager)
    assert ("synthesize_clone", "x", "a.wav", "", None) in fake.calls
    manager.shutdown()


def test_alias_collision_refuses_second_adapter(tmp_path, monkeypatch):
    import copy
    from conftest import FAKE_MANIFEST, make_model_repo
    first = copy.deepcopy(FAKE_MANIFEST)
    first["name"] = "first-tts"
    first["aliases"] = ["qwen3"]
    make_model_repo(tmp_path, "first-tts", first)
    second = copy.deepcopy(FAKE_MANIFEST)
    second["name"] = "second-tts"
    second["aliases"] = ["qwen3"]
    make_model_repo(tmp_path, "second-tts", second)
    manager = _manager(tmp_path, monkeypatch)
    assert "first-tts" in manager._models
    assert "second-tts" not in manager._models
    assert manager.resolve_model("qwen3") == "first-tts"
    manager.shutdown()


def test_strip_brackets_capability_parsed(tmp_path, monkeypatch):
    import copy
    from conftest import FAKE_MANIFEST, make_model_repo
    manifest = copy.deepcopy(FAKE_MANIFEST)
    manifest["strip_brackets"] = True
    make_model_repo(tmp_path, "fake-tts", manifest, config={"pitch": 0.9})
    manager = _manager(tmp_path, monkeypatch)
    caps = _adapter(manager).capabilities()
    assert caps.strip_brackets is True
    manager.shutdown()


def test_speaker_catalogue_model_filter(fake_models_dir, monkeypatch):
    manager = _manager(fake_models_dir, monkeypatch)
    # Fake adapters expose no supported_speakers hook: empty catalogue entries.
    assert manager.speaker_catalogue() == [{"model": "fake-tts", "speakers": []}]
    assert manager.speaker_catalogue(model="fake-tts") == [
        {"model": "fake-tts", "speakers": []}]
    assert manager.speaker_catalogue(model="ghost") == []
    manager.shutdown()
