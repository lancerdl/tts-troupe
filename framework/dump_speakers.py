"""One-shot probe: print a model's built-in speaker list as JSON.

Runs INSIDE the model's own venv (like ``model_worker``), because it imports the
model's adapter module and may load a checkpoint to enumerate speakers:

    <model-venv-python> -m framework.dump_speakers --repo <repo dir>

Used by ``tts-troupe-sync-presets``, which writes the result to the
``presets.txt`` sidecar the server reads at registration time.
"""
import argparse
import json
import sys
from pathlib import Path

from framework.adapter import create_adapter


def main(argv=None) -> int:
    # Model speaker names may be non-ASCII; never depend on the console codepage.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(
        description="Dump a model's built-in speaker list as JSON.")
    parser.add_argument("--repo", required=True,
                        help="Model repo directory (contains model.yaml)")
    args = parser.parse_args(argv)

    adapter = create_adapter(Path(args.repo).resolve())
    getter = getattr(adapter.obj, "supported_speakers", None)
    if not callable(getter):
        print(json.dumps({"error": "adapter has no supported_speakers()"}))
        return 1
    speakers = getter()
    if not isinstance(speakers, (list, tuple)):
        print(json.dumps({"error": "supported_speakers() did not return a list"}))
        return 1
    print(json.dumps([str(name) for name in speakers], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
