"""Clone-path instruction channel: manifest resolution and worker forwarding.

Covers the voice_clone `accepts_instruct` / `instruct_map` / `instruct_on_miss`
keys, the MappedAdapter gate, and the harness-cached worker branch.
"""
from conftest import FAKE_MANIFEST, make_model_repo
from framework.adapter import MappedAdapter, create_adapter, resolve_instruct
from framework.model_worker import _synthesize_clone_cached

MAPPED = {
    "accepts_instruct": True,
    "instruct_on_miss": "drop",
    "instruct_map": {"happy": "高兴", "sad": "悲伤"},
}


# ---- resolve_instruct (pure, manifest-driven) ----

def test_resolve_instruct_maps_a_key_case_insensitively():
    assert resolve_instruct(MAPPED, "sad") == "悲伤"
    assert resolve_instruct(MAPPED, "  Sad  ") == "悲伤"


def test_resolve_instruct_accepts_a_label_the_client_already_sent():
    assert resolve_instruct(MAPPED, "高兴") == "高兴"


def test_resolve_instruct_first_hit_wins_across_joined_directives():
    # The server joins multiple directives with ", ".
    assert resolve_instruct(MAPPED, "sad, slow") == "悲伤"
    assert resolve_instruct(MAPPED, "slow, sad") == "悲伤"


def test_resolve_instruct_miss_drops_by_default():
    assert resolve_instruct(MAPPED, "wistful") is None


def test_resolve_instruct_miss_can_pass_through_raw():
    assert resolve_instruct({**MAPPED, "instruct_on_miss": "raw"},
                            "wistful") == "wistful"


def test_resolve_instruct_without_a_map_passes_through():
    assert resolve_instruct({"accepts_instruct": True}, "cheerful") == "cheerful"


def test_resolve_instruct_blank_yields_none():
    assert resolve_instruct(MAPPED, None) is None
    assert resolve_instruct(MAPPED, "   ") is None


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
    def __init__(self, resolved):
        self.obj = _HookObj()
        self._resolved = resolved

    def resolve_instruct(self, cap_type, instruct):
        assert cap_type == "voice_clone"
        return self._resolved


HOOK = {"extract": "extract_prompt", "synthesize": "synth"}


def test_cached_clone_forwards_the_resolved_instruct():
    adapter = _StubAdapter("悲伤")
    _synthesize_clone_cached(
        {"text": "hi", "ref_audio_path": "ref.wav", "ref_text": "t",
         "instruct": "sad"}, adapter, _StubCache(), HOOK)
    call = adapter.obj.calls[-1]
    assert call["instruct"] == "悲伤"
    assert call["text"] == "hi"


def test_cached_clone_omits_instruct_when_nothing_resolved():
    adapter = _StubAdapter(None)
    _synthesize_clone_cached({"text": "hi", "ref_audio_path": "ref.wav"},
                             adapter, _StubCache(), HOOK)
    assert "instruct" not in adapter.obj.calls[-1]
