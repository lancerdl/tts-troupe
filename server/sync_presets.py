"""Generate ``models/<repo>/presets.txt`` from a model's own speaker list.

The built-in voice list is not a hand-maintained manifest key: this tool asks
the model (``supported_speakers()`` in the model's venv), then writes a
comma-separated sidecar the server reads at registration time:

    tts-troupe-sync-presets            # every repo under models/
    tts-troupe-sync-presets ming qwen3 # named repos only

Running the model may load a checkpoint (Qwen3's CustomVoice, for instance), so
this is a deliberate, one-off operation - never startup work.
"""
import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
PRESETS_FILE = "presets.txt"

logger = logging.getLogger("tts_troupe.sync_presets")


def _all_repos() -> List[Path]:
    return sorted(path.parent for path in MODELS_DIR.glob("*/model.yaml"))


def _identifiers(repo: Path) -> set:
    """Names a repo answers to: its directory, manifest name, and aliases."""
    manifest = yaml.safe_load((repo / "model.yaml").read_text(encoding="utf-8")) or {}
    names = {repo.name.lower(), str(manifest.get("name", "")).strip().lower()}
    names |= {str(a).strip().lower() for a in (manifest.get("aliases") or [])}
    return {n for n in names if n}


def _model_dirs(names: List[str]) -> List[Path]:
    """Resolve requested names (repo dir, manifest name, or alias) to repos."""
    repos = _all_repos()
    if not names:
        return repos
    resolved: List[Path] = []
    for requested in names:
        key = requested.strip().lower()
        match = next((repo for repo in repos if key in _identifiers(repo)), None)
        resolved.append(match or (MODELS_DIR / requested).resolve())
    return resolved


def _interpreter(repo: Path, manifest: dict) -> Path:
    """Resolve the model's worker interpreter (relative to the repo)."""
    python = (manifest.get("runtime") or {}).get("python") or ".venv/Scripts/python.exe"
    path = Path(python)
    return path if path.is_absolute() else repo / path


def _probe(repo: Path, python: Path) -> List[str]:
    """Ask the model for its speakers by running the probe in its own venv."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [str(python), "-m", "framework.dump_speakers", "--repo", str(repo)],
        capture_output=True, text=True, encoding="utf-8", env=env, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(
            f"probe failed (exit {proc.returncode}): {proc.stderr.strip()[-600:]}")
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    if isinstance(payload, dict) and "error" in payload:
        raise RuntimeError(payload["error"])
    return [str(name) for name in payload]


def _write(repo: Path, names: List[str]) -> Path:
    path = repo / PRESETS_FILE
    tmp = repo / (PRESETS_FILE + ".tmp")
    tmp.write_text(",".join(names), encoding="utf-8")
    os.replace(tmp, path)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Write each model's presets.txt from supported_speakers().")
    parser.add_argument("models", nargs="*",
                        help="Model repo names under models/ (default: all)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    failures = 0
    for repo in _model_dirs(args.models):
        if not (repo / "model.yaml").is_file():
            logger.error("%s: no model.yaml", repo)
            failures += 1
            continue
        manifest = yaml.safe_load((repo / "model.yaml").read_text(encoding="utf-8"))
        name = manifest.get("name", repo.name)
        if not (manifest.get("capabilities") or {}).get("voice_preset"):
            logger.info("%s: no voice_preset capability; skipping", name)
            continue
        python = _interpreter(repo, manifest)
        if not python.exists():
            logger.error("%s: interpreter not found: %s", name, python)
            failures += 1
            continue
        try:
            names = _probe(repo, python)
        except Exception as exc:
            logger.error("%s: %s", name, exc)
            failures += 1
            continue
        if not names:
            logger.error("%s: model reported no speakers; %s left untouched",
                         name, PRESETS_FILE)
            failures += 1
            continue

        path = repo / PRESETS_FILE
        previous = []
        if path.exists():
            previous = [p.strip() for p in
                        path.read_text(encoding="utf-8").split(",") if p.strip()]
        if previous == names:
            logger.info("%s: up to date (%d voices)", name, len(names))
            continue
        _write(repo, names)
        logger.info("%s: %s %s -> %s", name,
                    "updated" if previous else "wrote",
                    PRESETS_FILE, ",".join(names))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
