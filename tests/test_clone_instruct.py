"""Clone-path instruction channel: manifest resolution and worker forwarding.

Covers the voice_clone `accepts_instruct` / `instruct_map` / `instruct_on_miss`
keys, the MappedAdapter gate, and the harness-cached worker branch.
"""
import numpy as np
import pytest

from conftest import FAKE_MANIFEST, make_model_repo
from framework.adapter import (
    InstructResolution,
    MappedAdapter,
    _extract_result,
    _resolve_arg,
    _validate_capabilities,
    create_adapter,
    resolve_instruct,
)
from framework.model_worker import _synthesize_clone_cached

MAPPED = {
    "accepts_instruct": True,
    "instruct_on_miss": "drop",
    "instruct_map": {"happy": "高兴", "sad": "悲伤"},
}


# ---- resolve_instruct (pure, manifest-driven) ----

def test_resolve_instruct_maps_a_key_case_insensitively():
    assert resolve_instruct(MAPPED, "sad") == InstructResolution("悲伤", "mapped")
    assert resolve_instruct(MAPPED, "  Sad  ") == InstructResolution("悲伤", "mapped")


def test_resolve_instruct_accepts_a_label_the_client_already_sent():
    assert resolve_instruct(MAPPED, "高兴") == InstructResolution("高兴", "mapped")


def test_resolve_instruct_first_hit_wins_across_joined_directives():
    # The server joins multiple directives with ", ".
    assert resolve_instruct(MAPPED, "sad, slow").value == "悲伤"
    assert resolve_instruct(MAPPED, "slow, sad").value == "悲伤"


def test_resolve_instruct_miss_drops_to_absent():
    assert resolve_instruct(MAPPED, "wistful") == InstructResolution(None, "absent")


def test_resolve_instruct_miss_can_pass_through_as_raw():
    assert resolve_instruct({**MAPPED, "instruct_on_miss": "raw"},
                            "wistful") == InstructResolution("wistful", "raw")


def test_resolve_instruct_without_a_map_is_raw():
    assert resolve_instruct({"accepts_instruct": True},
                            "cheerful") == InstructResolution("cheerful", "raw")


def test_resolve_instruct_blank_yields_absent():
    assert resolve_instruct(MAPPED, None) == InstructResolution(None, "absent")
    assert resolve_instruct(MAPPED, "   ") == InstructResolution(None, "absent")


# ---- MappedAdapter gate ----

def test_clone_forwards_instruct_when_the_capability_accepts_it(fake_models_dir):
    adapter = create_adapter(fake_models_dir / "fake-tts")
    adapter.synthesize_clone(text="hi", ref_audio_path="ref.wav",
                             ref_text="t", instruct="cheerful")
    assert ("synthesize_clone", "hi", "ref.wav", "t",
            "cheerful") in adapter.obj.calls


def test_clone_omits_instruct_when_the_capability_lacks_the_channel(
        fake_models_dir):
    manifest = {**FAKE_MANIFEST, "name": "no-instruct-clone",
                "capabilities": {"voice_clone": {"implementation": "native"}}}
    make_model_repo(fake_models_dir, "no-instruct-clone", manifest, {})
    adapter = create_adapter(fake_models_dir / "no-instruct-clone")
    # The kwarg must never reach the adapter (an adapter whose synthesize_clone
    # takes no `instruct` parameter must keep working).
    adapter.synthesize_clone(text="hi", ref_audio_path="ref.wav",
                             instruct="cheerful")
    assert ("synthesize_clone", "hi", "ref.wav", "",
            None) in adapter.obj.calls


def test_declarative_args_naming_instruct_counts_as_a_channel(tmp_path):
    class _Obj:
        def synth(self, **kwargs):
            return ([0.0], 24000)

    block = {"call": "synth", "result": {"audio": {"index": 0},
                                         "sample_rate": {"index": 1}}}
    without = MappedAdapter(
        {"name": "d", "capabilities": {"voice_clone": {**block, "args": {"text": "text"}}}},
        _Obj(), {}, tmp_path)
    with_ = MappedAdapter(
        {"name": "d", "capabilities": {"voice_clone": {
            **block, "args": {"text": "text", "instruct": "instruct"}}}},
        _Obj(), {}, tmp_path)
    assert without.accepts_instruct("voice_clone") is False
    assert with_.accepts_instruct("voice_clone") is True


# ---- worker: harness-cached clone branch ----

class _StubCache:
    def __init__(self):
        self.entries = {}

    def get(self, ref_audio, ref_text, voice_name):
        return self.entries.get(ref_audio)

    def put(self, ref_audio, ref_text, voice_name, prompt):
        self.entries[ref_audio] = prompt


class _HookObj:
    """Records what the cached path forwards to hook['synthesize']."""

    def __init__(self):
        self.calls = []

    def extract_prompt(self, ref_audio_path, ref_text=""):
        return {"prompt": ref_audio_path}

    def synth(self, text, prompt, ref_audio_path="", ref_text="", **extra):
        self.calls.append(dict(text=text, prompt=prompt,
                               ref_audio_path=ref_audio_path,
                               ref_text=ref_text, **extra))
        return ("audio", 24000)


class _StubAdapter:
    def __init__(self, resolution):
        self.obj = _HookObj()
        self._resolution = resolution

    def resolve_instruct(self, cap_type, instruct):
        assert cap_type == "voice_clone"
        return self._resolution


HOOK = {"extract": "extract_prompt", "synthesize": "synth"}


def test_cached_clone_forwards_the_resolved_instruct():
    adapter = _StubAdapter(InstructResolution("悲伤", "mapped"))
    _synthesize_clone_cached(
        {"text": "hi", "ref_audio_path": "ref.wav", "ref_text": "t",
         "instruct": "sad"}, adapter, _StubCache(), HOOK)
    call = adapter.obj.calls[-1]
    assert call["instruct"] == "悲伤"
    assert call["text"] == "hi"


def test_cached_clone_omits_instruct_when_nothing_resolved():
    adapter = _StubAdapter(InstructResolution(None, "absent"))
    _synthesize_clone_cached({"text": "hi", "ref_audio_path": "ref.wav"},
                             adapter, _StubCache(), HOOK)
    assert "instruct" not in adapter.obj.calls[-1]


# ---- spec composition, result specs and variant selection ----

def test_dict_arg_composes_a_structured_kwarg():
    assert _resolve_arg({"dict": {"emotion": "instruct"}},
                        {"instruct": "悲伤"}, {}) == {"emotion": "悲伤"}


def test_nested_dict_instruct_counts_as_a_channel(tmp_path):
    class _Obj:
        def make(self, **kwargs):
            return ([0.0], 24000)

    manifest = {"name": "d", "capabilities": {"voice_clone": {
        "call": "make",
        "args": {"text": "text", "controls": {"dict": {"emotion": "instruct"}}},
        "result": {"audio": {"index": 0}, "sample_rate": {"index": 1}}}}}
    adapter = MappedAdapter(manifest, _Obj(), {}, tmp_path)
    assert adapter.accepts_instruct("voice_clone") is True


def test_result_attr_reads_sample_rate_from_the_target():
    class _Target:
        sample_rate = 24000

    result = _extract_result(
        {"audio": {"index": []}, "sample_rate": {"attr": "sample_rate"}},
        np.zeros(4, dtype=np.float32), _Target())
    assert result.sample_rate == 24000
    assert result.audio.shape == (4,)


class _Obj:
    def __init__(self):
        self.calls = []

    def make(self, **kwargs):
        self.calls.append(kwargs)
        return ([0.0], 24000)


VARIANTS = {"name": "v", "capabilities": {"voice_clone": {
    "instruct_map": {"sad": "悲伤"},
    "instruct_on_miss": "drop",
    "variants": [
        {"when": {"instruct": "mapped"}, "call": "make",
         "args": {"text": "text", "controls": {"dict": {"emotion": "instruct"}}},
         "result": {"audio": {"index": 0}, "sample_rate": {"index": 1}}},
        {"call": "make",
         "args": {"text": "text"},
         "result": {"audio": {"index": 0}, "sample_rate": {"index": 1}}},
    ]}}}


def test_variant_selection_uses_the_mapped_state(tmp_path):
    obj = _Obj()
    adapter = MappedAdapter(VARIANTS, obj, {}, tmp_path)
    adapter.synthesize_clone(text="hi", ref_audio_path="r", instruct="sad")
    assert obj.calls[-1] == {"text": "hi", "controls": {"emotion": "悲伤"}}


def test_variant_selection_falls_back_to_the_default(tmp_path):
    obj = _Obj()
    adapter = MappedAdapter(VARIANTS, obj, {}, tmp_path)
    adapter.synthesize_clone(text="hi", ref_audio_path="r", instruct="wistful")
    assert obj.calls[-1] == {"text": "hi"}


def test_variants_and_implementation_are_mutually_exclusive(tmp_path):
    manifest = {"capabilities": {"voice_clone": {
        "implementation": "native", "variants": [{"call": "make"}]}}}
    with pytest.raises(ValueError):
        _validate_capabilities(manifest, tmp_path)


def test_variants_require_exactly_one_default(tmp_path):
    manifest = {"capabilities": {"voice_clone": {"variants": [
        {"when": {"instruct": "mapped"}, "call": "make"}]}}}
    with pytest.raises(ValueError):
        _validate_capabilities(manifest, tmp_path)
