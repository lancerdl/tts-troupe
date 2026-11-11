"""Shared fixtures for the Phase-1 (GPU-free) model-layer tests."""
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

ADAPTER_MODULE = textwrap.dedent(
    """
    import numpy as np


    class _DesignModel:
        def __init__(self, owner):
            self._owner = owner

        def gen_create(self, text, description):
            self._owner.calls.append(("instance:gen_create", text, description))
            return (("inst", np.array([0.5, 0.5])), 11025)


    class FakeModel:
        def __init__(self, config=None, model_dir=None):
            self.config = config or {}
            self.model_dir = model_dir
            self._loaded = False
            self.setup_count = 0
            self.shutdown_count = 0
            self.clear_state_count = 0
            self.calls = []
            self.instances = {"design": _DesignModel(self)}

        def setup(self):
            self._loaded = True
            self.setup_count += 1

        def shutdown(self):
            self._loaded = False
            self.shutdown_count += 1

        def is_loaded(self):
            return self._loaded

        def clear_state(self):
            self.clear_state_count += 1

        def get_instance(self, key):
            return self.instances[key]

        def gen_preset(self, text, voice, pitch=None, mode=None):
            self.calls.append(("gen_preset", text, voice, pitch, mode))
            return ("fake-audio", 24000)

        def gen_create(self, text, description):
            self.calls.append(("gen_create", text, description))
            return (("payload", np.array([0.1, 0.2, 0.3])), 8000)

        def synthesize_clone(self, text, ref_audio_path, ref_text):
            self.calls.append(("synthesize_clone", text, ref_audio_path, ref_text))
            return ("clone-audio", 16000)


    class BrokenModel(FakeModel):
        def gen_preset(self, text, voice, pitch=None, mode=None):
            raise RuntimeError("boom")
    """
)

FAKE_MANIFEST = {
    "name": "fake-tts",
    "source": "https://example.com/fake-tts",
    "module": "adapter_module",
    "setup": {"factory": "FakeModel", "config_file": "config.yaml"},
    "capabilities": {
        "voice_default": {
            "call": "gen_preset",
            "args": {
                "text": "text",
                "voice": "voice",
                "pitch": {"config": "pitch"},
                "mode": {"const": "fast"},
            },
            "presets": ["alloy", "echo"],
            "default_voice": "alloy",
        },
        "voice_create": {
            "instance": "design",
            "call": "gen_create",
            "args": {"text": "text", "description": "description"},
            "result": {"audio": {"index": [0, 1]}, "sample_rate": {"index": 1}},
        },
        "voice_clone": {"implementation": "native"},
    },
}

# Same adapter module, but only the voice_default capability is mapped.
MINIMAL_MANIFEST = {
    "name": "minimal-tts",
    "module": "adapter_module",
    "setup": {"factory": "FakeModel"},
    "capabilities": {
        "voice_default": {
            "call": "gen_preset",
            "args": {"text": "text", "voice": "voice"},
        },
    },
}

# gen_preset always raises: exercises the manager's exception mapping.
BROKEN_MANIFEST = {
    "name": "broken-tts",
    "module": "adapter_module",
    "setup": {"factory": "BrokenModel"},
    "capabilities": {
        "voice_default": {
            "call": "gen_preset",
            "args": {"text": "text", "voice": "voice"},
        },
    },
}


def make_model_repo(models_dir: Path, name: str, manifest: dict,
                    config: dict | None = None) -> Path:
    """Materialize a fake model repo (model.yaml + adapter module + config)."""
    model_dir = models_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "adapter_module.py").write_text(ADAPTER_MODULE, encoding="utf-8")
    (model_dir / "model.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    if config is not None:
        (model_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return model_dir


@pytest.fixture
def fake_models_dir(tmp_path) -> Path:
    models_dir = tmp_path / "models"
    make_model_repo(models_dir, "fake-tts", FAKE_MANIFEST, {"pitch": 0.9})
    return models_dir


@pytest.fixture
def minimal_models_dir(tmp_path) -> Path:
    models_dir = tmp_path / "models"
    make_model_repo(models_dir, "minimal-tts", MINIMAL_MANIFEST, {"pitch": 1.0})
    return models_dir


@pytest.fixture
def broken_models_dir(tmp_path) -> Path:
    models_dir = tmp_path / "models"
    make_model_repo(models_dir, "broken-tts", BROKEN_MANIFEST, {"pitch": 1.0})
    return models_dir


# ---------------------------------------------------------------------------
# Subprocess-runtime fixtures (Phase 1: server/worker relationship)
# ---------------------------------------------------------------------------

# Adapter module whose payloads are numpy arrays - wire-transport-compatible
# (the harness encodes float32 bytes; string payloads cannot cross the pipe).
WIRE_ADAPTER_MODULE = textwrap.dedent(
    """
    import numpy as np


    class WireModel:
        def __init__(self, config=None, model_dir=None):
            self.config = config or {}
            self.model_dir = model_dir
            self._loaded = False
            self.setup_count = 0
            self.shutdown_count = 0
            self.calls = []

        def setup(self):
            self._loaded = True
            self.setup_count += 1

        def shutdown(self):
            self._loaded = False
            self.shutdown_count += 1

        def is_loaded(self):
            return self._loaded

        def age_caches(self):
            self.calls.append(("age_caches",))

        def gen_preset(self, text, voice, instruct=None):
            self.calls.append(("gen_preset", text, voice, instruct))
            return (np.array([0.1, 0.2, 0.3], dtype=np.float32), 24000)

        def synthesize_clone(self, text, ref_audio_path, ref_text):
            self.calls.append(("synthesize_clone", text, ref_audio_path, ref_text))
            return (np.array([0.4, 0.5], dtype=np.float32), 16000)
    """
)

WIRE_MANIFEST = {
    "name": "wire-tts",
    "module": "adapter_module",
    "setup": {"factory": "WireModel"},
    "runtime": {
        "mode": "subprocess",
        "python": sys.executable,  # factory resolves absolute paths as-is
    },
    "capabilities": {
        "voice_default": {
            "call": "gen_preset",
            "args": {"text": "text", "voice": "voice", "instruct": "instruct"},
        },
        "voice_clone": {"implementation": "native"},
    },
}


def make_wire_repo(models_dir: Path, name: str = "wire-tts",
                   runtime_python: str = None,
                   runtime_worker: str = None) -> Path:
    """Materialize a subprocess-runtime fake repo (wire-compatible payloads)."""
    manifest = dict(WIRE_MANIFEST)
    runtime = dict(manifest["runtime"])
    if runtime_python:
        runtime["python"] = runtime_python
    if runtime_worker:
        runtime["worker"] = runtime_worker
    manifest["runtime"] = runtime
    model_dir = models_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "adapter_module.py").write_text(WIRE_ADAPTER_MODULE,
                                                 encoding="utf-8")
    (model_dir / "model.yaml").write_text(yaml.safe_dump(manifest),
                                          encoding="utf-8")
    return model_dir


@pytest.fixture
def wire_models_dir(tmp_path) -> Path:
    models_dir = tmp_path / "models"
    make_wire_repo(models_dir)
    return models_dir


@pytest.fixture
def wire_repo_factory(tmp_path):
    """Factory fixture for wire repos with custom runtime overrides."""
    def factory(name="wire-tts", runtime_python=None, runtime_worker=None):
        models_dir = tmp_path / "models"
        return make_wire_repo(models_dir, name, runtime_python, runtime_worker)
    return factory


# Adapter module with a clone_prompt hook whose payloads are torch tensors -
# exercises the real torch.save/torch.load round-trip (runs under the model
# venv interpreter, where torch exists).
WIRE_HOOK_ADAPTER_MODULE = textwrap.dedent(
    """
    import os
    import numpy as np
    import torch


    class WireHookModel:
        def __init__(self, config=None, model_dir=None):
            self.config = config or {}
            self.model_dir = model_dir
            self._loaded = False
            self.calls = []
            self.extract_count = 0

        def shutdown(self):
            self._loaded = False

        def is_loaded(self):
            return self._loaded

        def age_caches(self):
            pass

        def age_checkpoints(self, idle_timeout_seconds):
            # Test-controlled: the marker file makes Base report dropped, so
            # the Base=>cache-memory coupling is asserted deterministically
            # without racing the periodic cleaner.
            marker = os.path.join(str(self.model_dir or "."), "drop_base")
            return ["base"] if os.path.exists(marker) else []

        def gen_preset(self, text, voice, instruct=None):
            self.calls.append(("gen_preset", text, voice))
            return (np.array([0.1, 0.2], dtype=np.float32), 24000)

        def extract_clone_prompt(self, ref_audio_path, ref_text=""):
            self.extract_count += 1
            self.calls.append(("extract", ref_audio_path, ref_text))
            return {"spk": torch.ones(2, dtype=torch.float32),
                    "meta": ref_text}

        def synthesize_with_prompt(self, text, prompt, ref_audio_path="", ref_text=""):
            self.calls.append(("synthesize_with_prompt", text, prompt["meta"],
                               ref_audio_path, ref_text))
            return (np.array([0.7, 0.8], dtype=np.float32), 12000)
    """
)

WIRE_HOOK_MANIFEST = {
    "name": "wire-hook-tts",
    "module": "adapter_module",
    "setup": {"factory": "WireHookModel"},
    "runtime": {
        "mode": "subprocess",
        "python": "",  # filled in by the factory (model venv interpreter)
    },
    "capabilities": {
        "voice_default": {
            "call": "gen_preset",
            "args": {"text": "text", "voice": "voice", "instruct": "instruct"},
        },
        "voice_clone": {
            "implementation": "native",
            "clone_prompt": {
                "extract": "extract_clone_prompt",
                "synthesize": "synthesize_with_prompt",
                "checkpoint": "base",
                "versions": {"adapter": "0.5.4", "format": "1"},
            },
        },
    },
}


def make_wire_hook_repo(models_dir: Path, name: str = "wire-hook-tts",
                        runtime_python: str = None,
                        format_version: str = "1") -> Path:
    """Materialize a hook-capable subprocess repo (torch payload adapter)."""
    manifest = dict(WIRE_HOOK_MANIFEST)
    runtime = dict(manifest["runtime"])
    runtime["python"] = runtime_python or ""
    manifest["runtime"] = runtime
    manifest["capabilities"]["voice_clone"]["clone_prompt"]["versions"]["format"] = format_version
    model_dir = models_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "adapter_module.py").write_text(WIRE_HOOK_ADAPTER_MODULE,
                                                 encoding="utf-8")
    (model_dir / "model.yaml").write_text(yaml.safe_dump(manifest),
                                          encoding="utf-8")
    return model_dir


@pytest.fixture
def wire_hook_repo_factory(tmp_path):
    """Factory for hook-capable repos; skips if the model venv is absent."""
    model_venv_python = (Path(__file__).resolve().parents[1]
                         / "models" / "faster-qwen3-tts"
                         / ".venv" / "Scripts" / "python.exe")
    if not model_venv_python.exists():
        pytest.skip("model venv not present - persistence tests need torch")
    def factory(name="wire-hook-tts", format_version="1"):
        models_dir = tmp_path / "models"
        return make_wire_hook_repo(models_dir, name,
                                   runtime_python=str(model_venv_python),
                                   format_version=format_version)
    return factory