"""Audio-utils tests (tag stripping and instruction extraction)."""
from server.audio_utils import (
    extract_instructions,
    strip_bracket_tags,
    strip_xml_tags,
)


def test_bracket_tags_removed_whole():
    assert strip_bracket_tags("[angry] Hello there") == "Hello there"
    assert strip_bracket_tags("Hello [whisper] there") == "Hello there"
    assert strip_bracket_tags("[tone: angry] Stop!") == "Stop!"


def test_bracket_tags_whitespace_and_punctuation():
    assert strip_bracket_tags("Wait [pause] , then go") == "Wait, then go"
    assert strip_bracket_tags("[a] [b] speak") == "speak"
    # A tag-only line collapses to an empty line; surrounding lines survive.
    assert strip_bracket_tags("line one\n [x] \nline two") == "line one\n\nline two"


def test_overlong_bracket_token_preserved():
    long = "x" * 61
    assert strip_bracket_tags(f"a [{long}] b") == f"a [{long}] b"


def test_bracket_free_text_unchanged():
    assert strip_bracket_tags("plain text (with parens) {and braces}") == \
        "plain text (with parens) {and braces}"


def test_empty_input():
    assert strip_bracket_tags("") == ""


def test_xml_stripper_still_preserves_inner_text():
    assert strip_xml_tags("keep <whisper>this</whisper>") == "keep this"


# --------------------------------------------------------------------------
# Instruction extraction: {..} is the only directive delimiter.
# --------------------------------------------------------------------------

def test_instruction_extracted_and_removed_from_text():
    assert extract_instructions("{angry} Hello there") == ("Hello there", "angry")
    assert extract_instructions("Hello {whisper} there") == ("Hello there", "whisper")
    assert extract_instructions("no directives here") == ("no directives here", None)


def test_instruction_label_prefix_is_dropped():
    assert extract_instructions("{tone: sad} Stop!") == ("Stop!", "sad")
    assert extract_instructions("{emotion: happy} Go") == ("Go", "happy")
    assert extract_instructions("{style: bold} Go") == ("Go", "bold")


def test_multiple_instructions_join_into_one_value():
    # One request still synthesizes in a single pass.
    assert extract_instructions("{angry}{slow} Hello") == ("Hello", "angry, slow")


def test_instructions_clean_whitespace_around_punctuation():
    assert extract_instructions("Wait {pause} , then go") == ("Wait, then go", "pause")


def test_parens_and_brackets_are_not_instructions():
    # Parens are ordinary prose; [..] is native content for some models and is
    # handled per model by strip_bracket_tags, never by this function.
    text = "He paused (laughing) and said [Sigh] hello"
    assert extract_instructions(text) == (text, None)


def test_overlong_instruction_token_preserved():
    long = "x" * 61
    assert extract_instructions(f"a {{{long}}} b") == (f"a {{{long}}} b", None)


def test_instruction_only_input_yields_empty_text():
    # The caller's empty-input check turns this into the silence response.
    assert extract_instructions("{angry}") == ("", "angry")
    assert extract_instructions("{angry}   ") == ("", "angry")


def test_extract_instructions_empty_input():
    assert extract_instructions("") == ("", None)