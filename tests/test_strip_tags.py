"""
Unit tests for XML/HTML-like tag stripping in TTS requests.
"""
import unittest
from server.audio_utils import strip_xml_tags
from server.config import ServerConfig


class TestStripXmlTags(unittest.TestCase):
    def test_basic_tag_removal(self):
        text = "Hello <whisper>world</whisper>!"
        self.assertEqual(strip_xml_tags(text), "Hello world!")

    def test_nested_tags(self):
        text = "<outer><inner>Text inside</inner></outer>"
        self.assertEqual(strip_xml_tags(text), "Text inside")

    def test_tags_with_attributes(self):
        text = 'Hello <colour name="blue" opacity="0.8">colorful world</colour>.'
        self.assertEqual(strip_xml_tags(text), "Hello colorful world.")

    def test_self_closing_tags(self):
        text = "First sentence.<break time='500ms'/> Second sentence."
        self.assertEqual(strip_xml_tags(text), "First sentence. Second sentence.")

    def test_spaces_around_tags_and_punctuation(self):
        text = "We may get requests with <tags> flavour </tags> . It works !"
        self.assertEqual(strip_xml_tags(text), "We may get requests with flavour. It works!")

    def test_bracket_tags_preserved(self):
        # Bracket tags like [whisper] are native content for some models and
        # are stripped per model by strip_brackets; strip_xml_tags ignores them.
        text = "<tag>[whisper] Softly spoken text</tag>"
        self.assertEqual(strip_xml_tags(text), "[whisper] Softly spoken text")

    def test_multiline_text(self):
        text = "Line 1 <tag>\nLine 2</tag>\nLine 3"
        self.assertEqual(strip_xml_tags(text), "Line 1\nLine 2\nLine 3")

    def test_empty_and_whitespace(self):
        self.assertEqual(strip_xml_tags(""), "")
        self.assertEqual(strip_xml_tags("   "), "")
        self.assertEqual(strip_xml_tags("<tag></tag>"), "")
        self.assertEqual(strip_xml_tags("<tag>   </tag>"), "")

    def test_config_defaults_and_overrides(self):
        # Default is True
        cfg = ServerConfig()
        self.assertTrue(cfg.strip_tags)

        # Explicit False in constructor
        cfg_disabled = ServerConfig(strip_tags=False)
        self.assertFalse(cfg_disabled.strip_tags)

        # Explicit True in constructor
        cfg_enabled = ServerConfig(strip_tags=True)
        self.assertTrue(cfg_enabled.strip_tags)


if __name__ == "__main__":
    unittest.main()

