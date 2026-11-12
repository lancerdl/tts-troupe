"""Ming adapter: an instruction selects the EMOTION task on the clone path."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_ming_module():
    path = ROOT / "models" / "Ming-3.0-tts" / "adapter.py"
    spec = importlib.util.spec_from_file_location("ming_adapter_under_test",
                                                  path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeWaveform:
    """Stands in for a torch tensor (the server venv has no torch)."""

    def detach(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def float(self):
        return self

    def numpy(self):
        return np.zeros(4, dtype=np.float32)


class _FakePipeline:
    sample_rate = 24000

    def __init__(self):
        self.calls = []

    def synthesize(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeWaveform()


def _adapter_with_fake_pipeline():
    adapter = _load_ming_module().MingTTSModel(config={}, model_dir=None)
    fake = _FakePipeline()
    adapter._instances["tts"] = fake   # bypass get_instance's model import
    return adapter, fake


def test_plain_clone_uses_the_tts_task():
    adapter, fake = _adapter_with_fake_pipeline()
    adapter.synthesize_clone("hello", "ref.wav", "transcript")
    (call,) = fake.calls
    assert call["task"] == "TTS"
    assert call["prompt_text"] == "transcript"
    assert call["language_id"] == "English"
    assert call["use_spk_emb"] is True
    assert "controls" not in call


def test_instruct_label_selects_the_emotion_task():
    adapter, fake = _adapter_with_fake_pipeline()
    adapter.synthesize_clone("hello", "ref.wav", "transcript", instruct="悲伤")
    (call,) = fake.calls
    assert call["task"] == "EMOTION"
    assert call["controls"] == {"emotion": "悲伤"}
    assert call["use_spk_emb"] is True
    # EMOTION conditions on the speaker embedding only: no transcript, no
    # explicit target language.
    assert "prompt_text" not in call
    assert "language_id" not in call


def test_clone_result_carries_sample_rate():
    adapter, _ = _adapter_with_fake_pipeline()
    audio, sample_rate = adapter.synthesize_clone("hello", "ref.wav")
    assert sample_rate == 24000
    assert audio.shape == (4,)
