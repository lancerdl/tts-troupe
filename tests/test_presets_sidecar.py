"""presets.txt sidecar: read at registration, merge into the capability view."""
from pathlib import Path

import yaml

from framework.adapter import PRESETS_FILE, load_manifest
from server.sync_presets import _interpreter, _write


def _repo(tmp_path, sidecar=None, voice_preset=True):
    capabilities = {}
    if voice_preset:
        capabilities["voice_preset"] = {
            "call": "synthesize_preset",
            "args": {"text": "text", "voice": "voice"},
        }
    manifest = {"name": "t", "module": "adapter",
                "setup": {"factory": "F"}, "capabilities": capabilities}
    (tmp_path / "model.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    if sidecar is not None:
        (tmp_path / PRESETS_FILE).write_text(sidecar, encoding="utf-8")
    return tmp_path


def test_sidecar_supplies_presets(tmp_path):
    manifest = load_manifest(_repo(tmp_path, "Vivian, Serena ,Uncle_Fu"))
    assert manifest["capabilities"]["voice_preset"]["presets"] == [
        "Vivian", "Serena", "Uncle_Fu"]


def test_sidecar_absent_leaves_no_presets(tmp_path):
    manifest = load_manifest(_repo(tmp_path))
    assert "presets" not in manifest["capabilities"]["voice_preset"]


def test_sidecar_cannot_invent_the_capability(tmp_path):
    manifest = load_manifest(_repo(tmp_path, "Vivian", voice_preset=False))
    assert "voice_preset" not in manifest["capabilities"]


def test_sidecar_tolerates_newlines_and_duplicates(tmp_path):
    manifest = load_manifest(_repo(tmp_path, "Vivian\nSerena,Vivian\n"))
    assert manifest["capabilities"]["voice_preset"]["presets"] == [
        "Vivian", "Serena"]


def test_blank_sidecar_leaves_no_presets(tmp_path):
    manifest = load_manifest(_repo(tmp_path, " ,\n,  "))
    assert "presets" not in manifest["capabilities"]["voice_preset"]


def test_unicode_names_survive(tmp_path):
    manifest = load_manifest(_repo(tmp_path, "US_Emma,灵小柔"))
    assert manifest["capabilities"]["voice_preset"]["presets"] == [
        "US_Emma", "灵小柔"]


def test_real_manifests_expose_their_sidecars():
    root = Path(__file__).resolve().parents[1] / "models"
    for name, first in (("faster-qwen3-tts", "Vivian"),
                        ("Ming-3.0-tts", "US_Emma")):
        manifest = load_manifest(root / name)
        presets = manifest["capabilities"]["voice_preset"].get("presets", [])
        assert presets and presets[0] == first


def test_writer_emits_comma_separated(tmp_path):
    _write(tmp_path, ["Vivian", "灵小柔"])
    assert (tmp_path / PRESETS_FILE).read_text(encoding="utf-8") == "Vivian,灵小柔"


def test_interpreter_resolves_relative_to_the_repo(tmp_path):
    assert (_interpreter(tmp_path, {"runtime": {"python": ".venv/x"}})
            == tmp_path / ".venv/x")
    absolute = tmp_path / "py.exe"
    assert _interpreter(tmp_path, {"runtime": {"python": str(absolute)}}) == absolute


def test_legacy_voice_default_key_is_ignored(tmp_path, caplog):
    import logging

    from framework.adapter import capabilities_from_manifest

    (tmp_path / "model.yaml").write_text(yaml.safe_dump({
        "name": "legacy", "module": "a", "setup": {"factory": "F"},
        "capabilities": {"voice_default": {"call": "synthesize_preset"}},
    }), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        manifest = load_manifest(tmp_path)
    assert capabilities_from_manifest(manifest).voice_preset is False
    assert "renamed to 'voice_preset'" in caplog.text


def test_model_dirs_resolves_names_and_aliases(tmp_path, monkeypatch):
    from server import sync_presets

    repo = tmp_path / "My-Model"
    repo.mkdir()
    (repo / "model.yaml").write_text(yaml.safe_dump(
        {"name": "my-model", "aliases": ["mm"], "module": "a",
         "setup": {"factory": "F"}}), encoding="utf-8")
    monkeypatch.setattr(sync_presets, "MODELS_DIR", tmp_path)
    assert sync_presets._model_dirs(["mm"]) == [repo]
    assert sync_presets._model_dirs(["MY-MODEL"]) == [repo]
    assert sync_presets._model_dirs([]) == [repo]
