"""Ming's declarative capability mapping (real manifest + adapter, fake pipeline).

There are no synthesis methods left in the adapter: model.yaml maps every task
onto MingTTS.synthesize. These tests load the real files and assert the mapped
task/args for presets, both clone variants, and free-form design.
"""
from pathlib import Path

import numpy as np
import pytest

from framework.adapter import create_adapter

ROOT = Path(__file__).resolve().parents[1]
MING = ROOT / "models" / "Ming-3.0-tts"


class _FakePipeline:
    sample_rate = 24000

    def __init__(self):
        self.calls = []

    def synthesize(self, **kwargs):
        self.calls.append(kwargs)
        return np.zeros(4, dtype=np.float32)


@pytest.fixture
def adapter():
    mapped = create_adapter(MING)
    fake = _FakePipeline()
    mapped.obj._instances["tts"] = fake   # bypass the ming_tts import/load
    return mapped, fake


def test_preset_uses_the_ip_task(adapter):
    mapped, fake = adapter
    result = mapped.synthesize_preset(text="hi", voice="UK_George")
    (call,) = fake.calls
    assert call["task"] == "IP"
    assert call["controls"] == {"ip": "UK_George"}
    assert call["use_zero_spk_emb"] is True
    assert result.sample_rate == 24000


def test_plain_clone_uses_the_tts_task(adapter):
    mapped, fake = adapter
    mapped.synthesize_clone(text="hi", ref_audio_path="ref.wav", ref_text="t")
    (call,) = fake.calls
    assert call["task"] == "TTS"
    assert call["prompt_text"] == "t"
    assert call["language_id"] == "English"
    assert call["use_spk_emb"] is True
    assert "controls" not in call


def test_mapped_instruction_selects_the_emotion_variant(adapter):
    mapped, fake = adapter
    mapped.synthesize_clone(text="hi", ref_audio_path="ref.wav",
                            ref_text="t", instruct="sad")
    (call,) = fake.calls
    assert call["task"] == "EMOTION"
    assert call["controls"] == {"emotion": "悲伤"}
    # EMOTION conditions on the speaker embedding only: no transcript.
    assert "prompt_text" not in call


def test_native_label_input_still_maps(adapter):
    mapped, fake = adapter
    mapped.synthesize_clone(text="hi", ref_audio_path="ref.wav", instruct="悲伤")
    assert fake.calls[-1]["controls"] == {"emotion": "悲伤"}


def test_unmatched_instruction_falls_back_to_the_tts_variant(adapter):
    mapped, fake = adapter
    mapped.synthesize_clone(text="hi", ref_audio_path="ref.wav",
                            ref_text="t", instruct="wistful")
    (call,) = fake.calls
    assert call["task"] == "TTS"
    assert "controls" not in call


def test_freeform_description_uses_the_freeform_task(adapter):
    mapped, fake = adapter
    mapped.synthesize_create(text="hi", description="an old raspy man")
    (call,) = fake.calls
    assert call["task"] == "FREEFORM"
    assert call["instruction"] == "an old raspy man"
    assert call["use_zero_spk_emb"] is True


def test_supported_speakers_is_a_list(adapter):
    mapped, _ = adapter
    speakers = mapped.obj.supported_speakers()
    assert speakers == ["US_Emma", "UK_George", "UK_Charlotte", "灵小柔", "灵楷"]
