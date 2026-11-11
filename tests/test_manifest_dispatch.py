"""Phase-1 tests: YAML manifest interpretation and MappedAdapter dispatch."""
import numpy as np
import pytest

from conftest import FAKE_MANIFEST, make_model_repo
from framework import NoModelForCapabilityError
from framework.adapter import (
    MappedAdapter,
    _resolve_arg,
    create_adapter,
    load_manifest,
)


@pytest.fixture
def adapter(fake_models_dir):
    return create_adapter(fake_models_dir / "fake-tts")


def test_create_adapter_binds_factory_and_config(adapter, fake_models_dir):
    assert adapter.name == "fake-tts"
    assert adapter.source == "https://example.com/fake-tts"
    assert adapter.obj.config == {"pitch": 0.9}
    assert adapter.obj.model_dir == fake_models_dir / "fake-tts"


def test_capabilities_built_from_manifest(adapter):
    caps = adapter.capabilities()
    assert caps.voice_default is True
    assert caps.voice_clone is True
    assert caps.voice_create is True
    assert caps.presets == ("alloy", "echo")
    assert caps.default_voice == "alloy"


def test_voice_default_passthrough_config_and_const_args(adapter):
    result = adapter.synthesize_default(text="hi", voice="alloy", instruct=None)
    assert result.audio == "fake-audio"
    assert result.sample_rate == 24000
    assert ("gen_preset", "hi", "alloy", 0.9, "fast") in adapter.obj.calls


def test_voice_default_with_custom_instruct_passthrough(adapter):
    adapter.synthesize_default(text="hi", voice="echo", instruct="cheerful")
    assert ("gen_preset", "hi", "echo", 0.9, "fast") in adapter.obj.calls


def test_instance_selection_and_index_extraction(adapter):
    result = adapter.synthesize_create(text="hello", description="warm narrator")
    assert np.allclose(result.audio, [0.5, 0.5])
    assert result.sample_rate == 11025
    # The instance method was called, not the adapter object's own.
    assert ("instance:gen_create", "hello", "warm narrator") in adapter.obj.calls


def test_native_voice_clone_implementation(adapter):
    result = adapter.synthesize_clone(text="hey", ref_audio_path="ref.wav",
                                      ref_text="transcript")
    assert result.audio == "clone-audio"
    assert result.sample_rate == 16000
    assert ("synthesize_clone", "hey", "ref.wav", "transcript") in adapter.obj.calls


def test_unmapped_capability_raises(adapter):
    with pytest.raises(NoModelForCapabilityError):
        adapter._dispatch("does_not_exist", {})


def test_unsupported_arg_spec_raises(fake_models_dir):
    manifest = {**FAKE_MANIFEST, "name": "argerr",
                "capabilities": {
                    "voice_default": {
                        "call": "gen_preset",
                        "args": {"text": {"weird": 1}, "voice": "voice"},
                    },
                }}
    make_model_repo(fake_models_dir, "argerr", manifest, {"pitch": 0.9})
    adapter = create_adapter(fake_models_dir / "argerr")
    with pytest.raises(ValueError):
        adapter.synthesize_default(text="hi", voice="alloy", instruct=None)


def test_resolve_arg_passthrough_missing_raises():
    with pytest.raises(ValueError):
        _resolve_arg("voice", {"text": "hi"}, {})


def test_resolve_arg_dict_specs():
    assert _resolve_arg({"const": 42}, {}, {}) == 42
    assert _resolve_arg({"config": "pitch"}, {}, {"pitch": 0.9}) == 0.9
    with pytest.raises(ValueError):
        _resolve_arg({"unknown": 1}, {}, {})


def test_load_manifest_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path)


def test_load_manifest_missing_required_key(tmp_path):
    (tmp_path / "model.yaml").write_text("name: only-name\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(tmp_path)


def test_mapped_adapter_shutdown_delegation(adapter):
    fake = adapter.obj
    adapter.shutdown()
    assert adapter.is_loaded() is False
    assert fake.shutdown_count == 1


def test_worker_harness_maintenance_commands_exclude_real_activity():
    """age/ping/cache_info must not reset the worker's idle clock."""
    from framework.model_worker import MAINTENANCE_COMMANDS
    assert MAINTENANCE_COMMANDS == frozenset({"age", "ping", "cache_info"})


def test_mapped_adapter_is_loaded_guarded_without_method(tmp_path):
    """A legacy adapter without is_loaded() reports loaded -> never retires."""
    from framework.adapter import MappedAdapter

    class _Legacy:
        def __init__(self):
            self.shutdown_count = 0

        def shutdown(self):
            self.shutdown_count += 1

    manifest = {"name": "legacy-tts", "capabilities": {}}
    proxy = MappedAdapter(manifest, _Legacy(), {}, tmp_path)
    assert proxy.is_loaded() is True
    proxy.shutdown()
    assert proxy.obj.shutdown_count == 1


def test_manager_routes_missing_capability_via_pick(minimal_models_dir):
    from framework import ModelManager, ModelSettings
    manager = ModelManager(ModelSettings(models=["minimal-tts"]),
                           minimal_models_dir)
    manager.load()
    with pytest.raises(NoModelForCapabilityError):
        manager.synthesize_create(text="x", description="d")
