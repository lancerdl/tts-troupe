"""
Audio manipulation, tag segmentation, normalization, and format encoders.
"""
import glob
import io
import logging
import os
import re
import shutil
import struct
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Ensure ffmpeg from User Environment Path or WinGet is available in PATH for pydub mp3 encoding
if not shutil.which("ffmpeg"):
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
            user_path, _ = winreg.QueryValueEx(key, "Path")
            for p_entry in user_path.split(";"):
                if "ffmpeg" in p_entry.lower() and os.path.isdir(p_entry):
                    os.environ["PATH"] = p_entry + os.pathsep + os.environ.get("PATH", "")
                    break
    except Exception:
        pass

if not shutil.which("ffmpeg"):
    for base_dir in [os.environ.get("LOCALAPPDATA", ""), os.environ.get("ProgramFiles", ""), os.environ.get("ProgramFiles(x86)", "")]:
        if base_dir and os.path.isdir(base_dir):
            matches = glob.glob(os.path.join(base_dir, "**", "ffmpeg.exe"), recursive=True)
            if matches:
                ffmpeg_bin = os.path.dirname(matches[0])
                os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")
                break


def strip_xml_tags(text: str) -> str:
    """
    Remove XML/HTML-like tags (e.g. <whisper>, </tags>, <flavour colour="blue">)
    from text while preserving the enclosed content.
    Cleans up duplicate whitespace and spacing around punctuation created by tag stripping.
    """
    if not text:
        return ""
    # Remove XML/HTML-like opening, closing, and self-closing tags
    cleaned = re.sub(r'<[^>]+>', '', text)
    # Remove whitespace preceding punctuation that might have been left by tag boundaries
    cleaned = re.sub(r'[\t ]+([.,!?;:])', r'\1', cleaned)
    # Collapse multiple inline spaces/tabs to a single space
    cleaned = re.sub(r'[\t ]+', ' ', cleaned)
    # Clean up whitespace immediately adjacent to line breaks
    cleaned = re.sub(r'[\t ]*\n[\t ]*', '\n', cleaned)
    return cleaned.strip()


def strip_bracket_tags(text: str) -> str:
    """
    Remove square-bracket meta-tags (e.g. [angry], [whisper], [tone: sad])
    from text. Tags are directives, not content, so the whole token is
    removed (unlike strip_xml_tags, which preserves inner text). Tokens are
    length-bounded so pathological input cannot consume a line, and {..}/(..)
    are left untouched. Cleans up whitespace like strip_xml_tags.
    """
    if not text:
        return ""
    cleaned = re.sub(r'\[[^\[\]]{1,60}\]', '', text)
    # Remove whitespace preceding punctuation that might have been left by tag boundaries
    cleaned = re.sub(r'[\t ]+([.,!?;:])', r'\1', cleaned)
    # Collapse multiple inline spaces/tabs to a single space
    cleaned = re.sub(r'[\t ]+', ' ', cleaned)
    # Clean up whitespace immediately adjacent to line breaks
    cleaned = re.sub(r'[\t ]*\n[\t ]*', '\n', cleaned)
    return cleaned.strip()


def extract_instructions(raw_text: str) -> Tuple[str, Optional[str]]:
    """
    Extract inline ``{instruction}`` directives (e.g. ``{angry}``,
    ``{tone: sad}``) from text, returning ``(clean_text, instructions)``.

    Curly braces are the instruction delimiter and are always removed from
    the text, because an instruction is a directive rather than content.
    Whether the model ever receives the value is decided elsewhere, by the
    ``voice_preset.args`` mapping in its ``model.yaml``.

    The other two delimiters are deliberately untouched here. Square brackets
    are native content for some models (``[Sigh]``, ``[Laughter]`` spoken as
    vocal events by Ming-3.0-tts) and are handled per model by
    ``strip_bracket_tags``; parentheses are ordinary prose and must never be
    read as directives.

    Multiple directives are joined with ", " so a request still synthesizes
    in a single pass.
    """
    if not raw_text:
        return "", None

    found = [item.strip() for item in re.findall(r'\{([^{}]{1,60})\}', raw_text)]
    found = [
        re.sub(r'^(tone|emotion|style|voice)\s*:\s*', '', item, flags=re.IGNORECASE).strip('*_~"').strip()
        for item in found
    ]
    found = [item for item in found if item]

    cleaned = re.sub(r'\{[^{}]{1,60}\}', '', raw_text)
    # Same whitespace cleanup as the tag strippers, for consistency.
    cleaned = re.sub(r'[\t ]+([.,!?;:])', r'\1', cleaned)
    cleaned = re.sub(r'[\t ]+', ' ', cleaned)
    cleaned = re.sub(r'[\t ]*\n[\t ]*', '\n', cleaned)

    return cleaned.strip(), (", ".join(found) if found else None)


def normalize_audio(pcm: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """RMS loudness normalization with peak limiting to ensure uniform volume across all voices."""
    if len(pcm) == 0:
        return pcm
    rms = np.sqrt(np.mean(pcm ** 2))
    if rms > 1e-6:
        current_db = 20 * np.log10(rms)
        gain = 10 ** ((target_db - current_db) / 20)
        pcm = pcm * gain
    # Peak limit to prevent clipping distortion
    max_val = np.max(np.abs(pcm))
    if max_val > 0.95:
        pcm = pcm * (0.95 / max_val)
    return pcm


def to_pcm16(pcm: np.ndarray) -> bytes:
    """Convert float32 audio array to 16-bit PCM bytes with normalization."""
    pcm = normalize_audio(pcm)
    return np.clip(pcm * 32768, -32768, 32767).astype(np.int16).tobytes()


def make_wav_header(sample_rate: int, data_len: int = 0xFFFFFFFF) -> bytes:
    """Generate a standard 44-byte RIFF WAV header."""
    n_channels = 1
    bits = 16
    byte_rate = sample_rate * n_channels * bits // 8
    block_align = n_channels * bits // 8
    riff_size = 0xFFFFFFFF if data_len == 0xFFFFFFFF else 36 + data_len
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", riff_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate, byte_rate, block_align, bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_len))
    return buf.getvalue()


def to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 audio array to complete WAV file bytes."""
    raw = to_pcm16(pcm)
    return make_wav_header(sample_rate, len(raw)) + raw


def to_mp3_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 audio array to MP3 bytes using pydub."""
    try:
        from pydub import AudioSegment
    except ImportError:
        raise RuntimeError(
            "response_format='mp3' requires pydub, a required dependency of "
            "tts-troupe; the install appears incomplete")
    segment = AudioSegment(
        to_pcm16(pcm),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    buf = io.BytesIO()
    segment.export(buf, format="mp3")
    return buf.getvalue()

