"""
Voice library scanning and export for tts-troupe.

Scans the configured voices directory for reference-audio clone voices
(.wav + optional transcript) and exports the selectable voice lists for
SillyTavern UI: ``voices.txt`` (the clone archive) and one
``<model_name>_voices.txt`` per model (reserved design target when the model
supports it, then its built-in preset voices, then the clone archive).
"""
import glob
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger("tts_troupe.voice_library")


def scan_voice_library(voices_dir: str) -> Dict[str, Dict[str, str]]:
    """Scan configured directory for .wav files and companion transcripts."""
    voices = {}
    if not os.path.isdir(voices_dir):
        logger.warning("Voices directory not found: %s", voices_dir)
        return voices

    wav_files = glob.glob(os.path.join(voices_dir, "*.wav")) + glob.glob(
        os.path.join(voices_dir, "**", "*.wav"), recursive=True
    )
    # Deduplicate: the two glob patterns both match top-level .wav files.
    wav_files = list(dict.fromkeys(wav_files))
    for wav_path in wav_files:
        stem = os.path.splitext(os.path.basename(wav_path))[0]
        ref_text = ""
        found_transcript_path = ""
        base_path = os.path.splitext(wav_path)[0]
        # Strict transcript policy: only .reference.txt counts. Plain .txt
        # files are user meta-data, never transcripts (the boot transcription
        # pass generates missing .reference.txt files).
        candidate = base_path + ".reference.txt"
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8-sig") as f:
                    ref_text = f.read().strip()
                    found_transcript_path = candidate
            except Exception as e:
                logger.warning("Failed reading transcript for %s: %s", candidate, e)

        voices[stem] = {
            "ref_audio": wav_path,
            "ref_text": ref_text,
            "ref_text_path": found_transcript_path,
        }
    logger.info("Discovered %d voice clone(s) in %s", len(voices), voices_dir)
    return voices


def model_voice_options(presets: List[str], clones: List[str],
                        design_target: Optional[str] = None) -> List[str]:
    """Ordered selectable voices for one model.

    The reserved design target (when the model supports voice design) comes
    first, then the model's built-in preset voices, then the clone archive -
    the order a client presents them in.
    """
    ordered = [design_target] if design_target else []
    ordered.extend(presets)
    ordered.extend(clones)
    return ordered


def export_voice_list(path: str, voice_names: List[str]) -> None:
    """Write a clean comma-separated voice list to an explicit path.

    Used for the server-root exports: ``voices.txt`` (all cloneable voices
    in the configured archive) and ``<model_name>_voices.txt`` (one per
    model: its design target when supported, its preset voices, then the
    clone archive)."""
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(",".join(voice_names))
        logger.info("Exported %d voice(s) to %s", len(voice_names), path)
    except Exception as e:
        logger.warning("Failed writing voice list %s: %s", path, e)