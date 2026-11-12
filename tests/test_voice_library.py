"""Voice-library scan and export tests (transcript preference + exports)."""
from server.voice_library import (
    export_voice_list,
    model_voice_options,
    scan_voice_library,
)


def _write(path, content=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_reference_txt_is_the_transcript(tmp_path):
    _write(tmp_path / "A.reference.txt", "reference transcript")
    _write(tmp_path / "A.txt", "metadata")
    _write(tmp_path / "A.wav", "RIFF")
    voices = scan_voice_library(str(tmp_path))
    assert voices["A"]["ref_text"] == "reference transcript"
    assert voices["A"]["ref_text_path"].endswith("A.reference.txt")


def test_plain_txt_is_not_a_transcript(tmp_path):
    _write(tmp_path / "B.txt", "assumed transcript")
    _write(tmp_path / "B.wav", "RIFF")
    voices = scan_voice_library(str(tmp_path))
    assert voices["B"]["ref_text"] == ""
    assert voices["B"]["ref_text_path"] == ""


def test_wav_without_transcript_yields_empty_ref(tmp_path):
    _write(tmp_path / "C.wav", "RIFF")
    voices = scan_voice_library(str(tmp_path))
    assert voices["C"]["ref_text"] == ""
    assert voices["C"]["ref_text_path"] == ""


def test_recursive_scan(tmp_path):
    _write(tmp_path / "sub" / "D.wav", "RIFF")
    _write(tmp_path / "sub" / "D.reference.txt", "deep")
    voices = scan_voice_library(str(tmp_path))
    assert voices["D"]["ref_text"] == "deep"


def test_missing_directory_yields_empty(tmp_path):
    assert scan_voice_library(str(tmp_path / "nope")) == {}


def test_export_voice_list_writes_comma_separated(tmp_path):
    out = tmp_path / "voices.txt"
    export_voice_list(str(out), ["X", "Y", "Z"])
    assert out.read_text(encoding="utf-8") == "X,Y,Z"


def test_export_voice_list_per_model_filename(tmp_path):
    out = tmp_path / "some-model_voices.txt"
    export_voice_list(str(out), ["alloy", "X"])
    assert out.read_text(encoding="utf-8") == "alloy,X"


def test_model_voice_options_design_target_first():
    assert model_voice_options(["Vivian", "Ryan"], ["x"], "design") == [
        "design", "Vivian", "Ryan", "x"]


def test_model_voice_options_without_design_support():
    assert model_voice_options(["Vivian"], ["x"]) == ["Vivian", "x"]


def test_model_voice_options_only_design_target():
    assert model_voice_options([], ["x"], "design") == ["design", "x"]