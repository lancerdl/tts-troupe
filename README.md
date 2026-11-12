# TTS Troupe

A troupe of voices for your Text-To-Speech needs!

This repository is a high-performance, modular speech synthesis server exposing an **OpenAI-compatible** audio speech API (`POST /v1/audio/speech`). It is designed for bot/chat frontends such as SillyTavern, and works with any client that speaks the OpenAI TTS protocol.

The server is **model-agnostic** and ships with **no models bundled**. Synthesis is delegated to pluggable model adapters that you drop into the gitignored `models/` directory, each declared by a `model.yaml` manifest. Adding, swapping, or removing a model is a filesystem operation — no server code changes.

Any TTS backend can be integrated as long as its adapter can expose three optional capabilities: **preset voices**, **reference-audio voice cloning**, and **natural-language voice design**. See [Adding a new TTS model](#adding-a-new-tts-model) for the contract, and [Example integrations](#example-integrations) for two worked, end-to-end examples (`faster-qwen3-tts` and `Ming-3.0-tts`).

## Architecture

Three layers, one dependency direction. **Deep-dive: [`ARCHITECTURE.md`](ARCHITECTURE.md)** — wire protocol, worker lifecycle, cache ownership, and how downloaded models integrate.

```
server/          ->  framework/          <-  models/<model-repo>/
FastAPI,             the model API:       independent git repos: you drop
routing, voice       ModelAdapter types,  in model.yaml + adapter; each
library, audio       YAML interpreter,    has its own venv + deps and
exports              ModelManager         imports framework.* inside it
                     proxy + worker harness
```

- **`server/`** — FastAPI endpoints, model-pinned voice routing, voice archive pipeline, audio utils, logging, config. Knows nothing about specific models.
- **`framework/`** — the generic model layer shipped with this repo: `ModelAdapter` protocol, YAML manifest interpreter, `ModelManager`, subprocess proxy + worker harness, CUDA health checks, voice cache.
- **`models/<model-repo>/`** — independent git repositories cloned into the gitignored `models/` drop directory. The upstream repo brings its own code, dependencies, and virtual environment; **you** overlay the integration files on top — a `model.yaml` manifest at its root plus the adapter class the manifest names.

### Server environment

The server runs from its own virtual environment at the repo root (no torch —
models live in their own venvs). Create it once:

```
uv venv .venv --python 3.10
uv pip install -e ".[transcribe]" pytest --python .venv\Scripts\python.exe
```

Without `uv`, the stdlib `venv` module is equivalent and does include `pip`:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[transcribe]" pytest
```

`start_tts_server.bat` and `transcribe_voices.bat` automatically use
`.venv\Scripts\python.exe`. The `transcribe` extra brings in
`faster-whisper` for the offline voice-library transcriber.

### Cloning the model repo

Model repositories are plain git clones (gitignored, not vendored). Each model
owns its runtime: it declares its own dependencies and installs them into
**its own virtual environment** however it chooses, which the server never
touches. Two installation shapes are supported:

- **An installable package** — the repo ships a `pyproject.toml` plus a setup
  script, installed with `uv pip install -e .` (falling back to `pip`). Its code
  is imported through that install.
- **A directory of loose modules** — the repo ships no packaging metadata at
  all, only `requirements.txt` and setup steps in its own README. Its code is
  imported from its directory at runtime, which the worker adds to `sys.path`.

The server process stays dependency-light (FastAPI/uvicorn only — no torch).
Clone a model repo into `models/` and run its setup once; both shapes are
demonstrated end to end in [Example integrations](#example-integrations).

The repo's `model.yaml` declares how it runs. A `runtime:` block with
`mode: subprocess` makes the manager spawn the model in its own venv and talk
to it over stdio JSON (no HTTP in the model layer):

```yaml
runtime:
  mode: subprocess
  python: .venv/Scripts/python.exe
```

The server discovers model repos from the `models:` list in `config.yaml` (or the `TTS_MODELS` env var). Each model's `model.yaml` carries a `source:` URL logged at startup. **If a model's venv is missing, the model still registers** (capabilities and routing work) and synthesis returns **501** with a message directing you to the model's own setup instructions — the project assumes nothing about how models are initialized.

### Adding a new TTS model

A model integrates by (a) being a git repo that owns its own dependencies and setup — either an installable package (`pyproject.toml` + setup script) or a directory of modules plus `requirements.txt` — (b) being given a `model.yaml` at its root (which **you** author) declaring its adapter factory and capability mappings plus a `runtime:` block, and (c) being added to the `models:` list in `config.yaml` — zero server changes. Models with conflicting dependencies coexist because each runs in its own interpreter.

The annotated [`model.yaml.template`](model.yaml.template) at the repo root is the full reference — every key with its required/optional status, default, and the arg/result spec forms (`{const:}`, `{config:}`, `{dict:}`; result `{index:}` / `{attr:}`). Copy it to `model.yaml` in the model repo and edit.

#### Built-in voices (`presets.txt`)

A model's built-in voice list is **not** a manifest key. It lives in a comma-separated sidecar, `models/<repo>/presets.txt`:

```
Vivian,Serena,Uncle_Fu,Dylan,Eric,Ryan,Aiden,Ono_Anna,Sohee
```

The server reads it at registration (a file read — no model code, no checkpoint load). Generate it by asking the model, which runs `supported_speakers()` in the model's own venv (this may load a checkpoint, so it is deliberate and one-off):

```
tts-troupe-sync-presets              # every repo under models/
tts-troupe-sync-presets ming qwen3   # repo dir, manifest name, or alias
```

You can also write the file by hand. Without it the model simply has no preset voices: preset names fall through to the clone/design checks and then `400`.

#### Authoring `adapter.py`

The companion file is the model's **lifecycle**, not its capabilities. The mapper itself lives in `framework/` (`MappedAdapter`); `adapter.py` is the model **host** — the object the mapping's calls target, plus construction and residency. Write it only for what `model.yaml` cannot express:

- **the factory class** named by `setup.factory`, constructed as `factory(config=<setup.config_file dict>, model_dir=<Path>)`;
- **`get_instance(key)`** — only when a capability names an `instance:`;
- optionally `is_loaded`, `age_caches`, `age_checkpoints`, and `supported_speakers` (must return a **list** — the worker wraps it under `{"speakers": …}`).

Everything else belongs in the manifest: task/argument shaping, structured kwargs (`controls: {dict: {emotion: instruct}}`), per-instruction variants, and result extraction. Reach for a native `synthesize_<preset|clone|create>` method only when the grammar genuinely cannot express the call — runtime branching beyond `instruct` presence, multi-step call chains, argument coercions, or a private prompt cache.

[`adapter.py.template`](adapter.py.template) at the repo root is the skeleton to copy. The two worked integrations show the split: `Ming-3.0-tts` is entirely `model.yaml` plus a lifecycle-only adapter (its `TTS`, `EMOTION`, `IP` and `FREEFORM` tasks are all mapped), while `faster-qwen3-tts` keeps hooks for its three checkpoints and prompt cache.

---

## Features

- **OpenAI-compatible endpoint**: `POST /v1/audio/speech` with `mp3`, `wav`, and `pcm` output.
- **Manual test console**: open the server root in a browser — `http://127.0.0.1:8000/` by default (the `server.host`/`server.port` from `config.yaml`). It serves a self-contained web page for manual testing — voice picker, synthesis with in-browser playback, error surfacing (501/503), and a live cache-status panel.
- **Pluggable model adapters**: synthesis is delegated to models declared under `models/` via `model.yaml` manifests — adding a new TTS model requires zero server changes.
- **Required, alias-aware `model` selection**: requests name a model by its canonical name or a declared alias; voices resolve within that model — the reserved `custom` design target first, then custom presets, then the clone archive.
- **Text conditioning**: `<tag>` markup is stripped globally (`strip_tags`); `{instruction}` directives are extracted and passed as delivery steering where the model accepts it; `[tag]` is left alone for models that speak it natively (e.g. `[Sigh]`-style vocal tags) and stripped per model via the manifest `strip_brackets` flag otherwise.
- **Real-time CUDA-graph inference** with streaming-capable generation.
- **Lazy VRAM loading**: models load into GPU memory on first use, not at startup.
- **Startup model verification**: all configured models are checked / downloaded into the Hugging Face cache on boot.
- **On-demand voice-prompt caching**: cloned-voice features are extracted once and reused, with a TTL so idle voices are evicted.
- **Idle model unload**: unused models are freed from VRAM so it can coexist with other GPU apps (ComfyUI, etc.).
- **Unified background memory cleaner**: a single daemon thread watches both the loaded-model table and the voice-prompt cache.
- **Graceful shutdown**: all VRAM is freed and the CUDA cache is cleared on exit.
- **Dual logging**: console (stdout) plus persistent `tts_server.log`.
- **Voice library tooling**: `faster-whisper` auto-transcription of reference clips, plus a clean comma-separated `voices.txt` export for SillyTavern.

---

## How It Works

### Model-Pinned Voice Resolution

Every request names a `model` — a canonical adapter name or declared alias,
case-insensitive; unknown models are rejected with `400`. The voice then
resolves **within that model** (per-voice checkpoint selection, immune to name
collisions across models):

```
Incoming request { model, voice, input, response_format }
                        │
                        ▼
     model resolves to a registered adapter?  ── NO → 400 (lists available models)
                        │ YES (pinned)
                        ▼
   ┌─────────────────────────────────────────────────────────┐
   │ 1. voice matches a built-in preset name?                │
   │    (from <repo>/presets.txt)                            │
   └───────────────────────────┬─────────────────────────────┘
                               │ YES → model's preset capability (+ instruct steering)
                               │ NO
                               ▼
   ┌─────────────────────────────────────────────────────────┐
   │ 2. voice matches a clone stem in the voice archive?    │
   │    (e.g. "AliceW" → AliceW.wav + AliceW.reference.txt) │
   └───────────────────────────┬─────────────────────────────┘
                               │ YES → model's voice-clone capability
                               │ NO
                               ▼
   ┌─────────────────────────────────────────────────────────┐
   │ 3. voice is the reserved design target "custom"?        │
   │    (the {description} directive in `input` drives it)   │
   └───────────────────────────┬─────────────────────────────┘
                               │ YES → model's voice-design capability
                               │ NO / unsupported → 400
```

**Custom presets**

- Matches one of the model's built-in preset voices, read from
  `<repo>/presets.txt` (generate it with `tts-troupe-sync-presets`). The list
  is mirrored as `<model_name>_voices.txt` in the server root for SillyTavern.
- Inline tags in the input text (e.g. `[whisper]`, `[tone: angry]`, `{sad}`,
  `(sigh)`) are parsed into per-segment `instruct` steering for real-time
  emotional control.

**Voice clones**

- The reference `.wav` stem (without extension) is used as the voice name.
- A companion transcript (`<name>.reference.txt`) is read
  at request time for higher-quality in-context-learning (ICL) cloning.
- The extracted speaker prompt is cached on first request (memory mirror +
  persisted `.pt`) and reused for subsequent requests of the same voice
  (subject to the prompt TTL).
- A `{...}` directive in the input is resolved against the model's
  `instruct_map`: on a model that declares one (Ming), `{sad}` on a cloned
  voice selects its native emotion control, and a directive that maps to no
  label falls back to a plain clone.

**Voice Design** (the reserved `custom` target)

- Set `voice` to `custom` and put the voice description in a `{...}` directive
  in `input` (e.g. `{"voice": "custom", "input": "{a warm elderly scholar} Good morning."}`).
  The directive is extracted from the text and passed as the design
  description — for models that support `voice_create`. `custom` with no
  `{...}` present, or a model without voice design, is rejected with `400`.
- Any other `voice` string that matches neither a preset nor a clone stem is
  rejected with `400`: an unmatched voice string is never silently read as a
  description.
- This gives you an endless supply of one-off voices without touching the voice library.

### Memory Lifecycle

| Phase | Behavior |
| --- | --- |
| **Startup** | The model's checkpoints are verified in the Hugging Face cache and downloaded if missing. No VRAM is used yet. |
| **First use of a model** | Weights are loaded into VRAM and CUDA graphs are captured. |
| **Voice clone prompt** | Extracted once per voice on first request; cached in RAM with a TTL. |
| **Background clean-up** | A daemon thread sweeps every 30s (a fixed cadence) and applies the configurable thresholds from `config.yaml`: unloads models idle past `model_idle_timeout_minutes` and evicts prompts stale past `prompt_cache_ttl_minutes`. |
| **CUDA Health & Sleep Resilience** | Pre-transaction health checks and bounded timeouts catch hung/frozen CUDA states (e.g. after Windows sleep). Corrupted references are dumped cleanly without deadlocking `torch.cuda` calls. |
| **Shutdown** | The cleaner thread is signalled to stop and every model worker is asked to exit over the wire (falling back to a hard `kill`). VRAM is reclaimed by the process exiting — no explicit CUDA or `gc.collect()` cleanup is called. |

Every add/remove of a model or voice to/from memory is logged to console and `tts_server.log` with a `[MEMORY]` marker. Failure states and recovery dumps are marked with `[CUDA HEALTH]`.

---

## Configuration

All settings live in **`config.yaml`** (in this directory), which is auto-created from **`config.yaml.example`** on first run and is **gitignored** (machine-specific). The precedence is:

```
CLI arguments  >  environment variables  >  config.yaml  >  built-in defaults
```

- `config.yaml.example` — committed, portable template. Edit values there when you want to change defaults for new installs.
- `config.yaml` — your local file. **This is the one you actually edit** for this machine (e.g. set `voices_dir` to your own path). It is never tracked by git.
- If `config.yaml` is missing, it is recreated automatically from the example on next startup.

### What to tailor

The template ships with workable defaults, so you normally change only a couple of things in your `config.yaml`:

- **`voices.voices_dir`** — point this at your own reference-audio folder (default `voices`). This is the one you'll almost always change.
- **Timeouts** — `server.synthesis_timeout_seconds`, `server.cold_start_timeout_seconds`, and `server.recovery_cooldown_seconds` bound worker calls on the wire; `memory_management.model_idle_timeout_minutes` and `memory_management.prompt_cache_ttl_minutes` govern VRAM and prompt-cache aging. Adjust to suit your hardware and usage.

Everything else has a sensible default. The committed [`server/config.yaml.example`](server/config.yaml.example) is the full reference, with every key documented inline.

## Environment variables

Every setting can also be supplied as an environment variable — handy for containers, CI, or a one-off run. You do **not** need them when you have a proper `config.yaml`; when both are present the environment variables win (and command-line flags win over both).

- `TTS_CONFIG` — path to an alternate `config.yaml`
- `TTS_HOST` / `TTS_PORT` — bind address and port
- `TTS_MODELS` — comma-separated model repo names under `models/`
- `TTS_DEVICE` / `TTS_DTYPE` — torch device and precision hints
- `TTS_RESPONSE_FORMAT` — default output format when the client omits one
- `TTS_STRIP_TAGS` — `true`/`false` (or `1`/`0`) to control `<tag>` stripping
- `TTS_VOICES_DIR` — reference-audio folder
- `TTS_MODEL_IDLE_TIMEOUT` / `TTS_PROMPT_TTL` — aging thresholds (minutes)
- `TTS_CUDA_CHECK_COOLDOWN` / `TTS_CUDA_HEALTH_TIMEOUT` — CUDA probe timing (seconds)
- `TTS_SYNTHESIS_TIMEOUT` / `TTS_COLD_START_TIMEOUT` / `TTS_RECOVERY_COOLDOWN` — worker wire deadlines (seconds)

---

## Quickstart

### Prerequisites

- At least one model repo cloned under `models/` with its own virtual environment set up — see [Cloning the model repo](#cloning-the-model-repo) and [Example integrations](#example-integrations). With none present the server still starts; requests simply have no model to route to.
- Python 3.10+ with PyTorch + CUDA **in that model's own virtual environment** (each model ships its own setup — the server venv needs no torch).
- The model's checkpoints available in the local Hugging Face cache (or reachable online on first boot — `enable_prefetch: true` will download them).
- A configured `voices_dir` containing reference `.wav` files and optional `.reference.txt` transcripts.

### Start the server

```
start_tts_server.bat
```

This is the one-stop entry point: it launches the whole server, and the server also runs the voice-library transcription pass on boot (see [Auto-transcription](#auto-transcription)). You do **not** need to run `transcribe_voices.bat` as well.

or directly:

```
python -m server
```

(Optional CLI overrides, e.g. `--port 8080`, `--voices-dir <path>` — see `tts-troupe --help` or `python -m server --help`.)

The server is ready when you see:

```
Application startup complete.
Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### Basic request

```
curl -s http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "<model-name-or-alias>", "input": "Hello, world!", "voice": "<voice-name>", "response_format": "mp3"}' \
  --output hello.mp3
```

`<model-name-or-alias>` and `<voice-name>` must match a model you have actually integrated and registered (for the `faster-qwen3-tts` example below, `"model": "qwen3"` and `"voice": "Vivian"`).

---

## Example integrations

The server assumes nothing about your models. Below are two complete
integrations — one per installation shape described in
[Cloning the model repo](#cloning-the-model-repo). Both are upstream projects
you clone yourself; nothing is bundled with this repo.

### Example 1 — `faster-qwen3-tts` (installable package)

[`faster-qwen3-tts`](https://github.com/andimarafioti/faster-qwen3-tts) is an
installable package exposing the full range of Qwen3-TTS capabilities:

- **Voice Cloning** (`Base` checkpoint) — zero-shot cloning from reference audio in your local voice library.
- **Preset Speakers** (`CustomVoice` checkpoint) — built-in voices (Vivian, Ryan, Serena, …) with live emotion/tone steering.
- **Voice Design** (`VoiceDesign` checkpoint) — synthesize an entirely new voice from a natural-language description.

Clone it into `models/` and run its setup once:

```
git clone https://github.com/andimarafioti/faster-qwen3-tts models/faster-qwen3-tts
models\faster-qwen3-tts\setup_windows.bat   # creates models/faster-qwen3-tts/.venv
```

Add a `model.yaml` at the repo root (abridged here — see the annotated
[`model.yaml.template`](model.yaml.template) for the full shape) and register
it in `config.yaml`:

```yaml
name: faster-qwen3-tts
aliases: [qwen3]
runtime:
  mode: subprocess
  python: .venv/Scripts/python.exe
```

```yaml
models:
  list: ["faster-qwen3-tts"]
```

Requests may then name the model as `faster-qwen3-tts` or its alias `qwen3`.

**Official Qwen3-TTS checkpoints** — only the following exist on Hugging Face;
the model's own `config.yaml` defaults already use valid combinations:

| Type | 0.6B | 1.7B |
| --- | --- | --- |
| `Base` (voice clone) | ✅ | ✅ |
| `CustomVoice` (presets) | ✅ | ✅ |
| `VoiceDesign` (voice design) | ❌ | ✅ |

> ⚠️ There is **no** `0.6B-VoiceDesign` release. If you see a 401 / "not a valid model identifier" error, check that `design_model` points to the 1.7B variant.

### Example 2 — `Ming-3.0-tts` (directory of loose modules)

[`Ming-3.0-tts`](https://github.com/inclusionAI/Ming-3.0-tts) (0.5b checkpoint)
provides multilingual synthesis with native **paralinguistic inline tags**
(`[Sigh]`, `[Laughter]`, `[Cough]`, … spoken as vocal events), built-in premium
voices (`US_Emma`, `UK_George`, …) via the `IP` task, zero-shot cloning, and
free-form natural-language voice control.

It ships no packaging metadata — no `pyproject.toml`, no `setup.py` — so set up
its venv with the steps in its own README (`requirements.txt`); text
normalization stays off (`pynini` has no reliable Windows wheels). Its code is
imported from the directory at runtime, so a `model.yaml` at its root (see
[`model.yaml.template`](model.yaml.template)) registers it like any other model:

```yaml
name: Ming-3.0-tts
aliases: [ming]
runtime:
  mode: subprocess
  python: .venv/Scripts/python.exe
```

```yaml
models:
  list: ["Ming-3.0-tts"]
```

Because that manifest leaves `strip_brackets` unset (default `false`), its
`[Sigh]`-style tags pass through and are spoken as vocal events.

---

## Voice Library & the `voices_dir`

Reference clones live in the folder configured by `voices.voices_dir` (default: `voices` relative to the server working directory; configure it in `config.yaml`).

```
AliceW.wav
AliceW.reference.txt        <- optional transcript (recommended)
Arnold.wav
...and so on
```

- The **file stem** (without `.wav`) is the voice name clients use, e.g. `"AliceW"`, `"Arnold"`.
- A companion transcript (`.reference.txt`) dramatically improves clone quality via ICL conditioning.
- Drop new `.wav` files in at any time — they are discovered on server start (or by re-running the transcriber).

### Auto-transcription

Transcription is part of the server boot, not a separate step: on startup the server runs a `faster-whisper` pass over the configured `voices_dir` (best-effort, non-fatal) and writes `<stem>.reference.txt` for any clip that lacks a transcript. Launching the server with `start_tts_server.bat` therefore covers it — you do **not** need to run `transcribe_voices.bat` as well. Clips dropped into the archive are transcribed on the next boot.

**User should validate outputs of automated transcription for accuracy and best results.**

The same pass is exposed as a standalone helper, for re-transcribing without restarting the server:

```
python -m server.transcribe_voices
```

or double-click `transcribe_voices.bat`.

- Only `<Name>.reference.txt` counts as a transcript; plain `.txt` files are treated as user meta-data and ignored.
- Any `.wav` without a `.reference.txt` is transcribed and written as `<stem>.reference.txt`.
- The standalone helper also writes a `voices/voices.txt` export; the server's boot pass skips that artifact, and the server writes its own exports in the server root (see below).

### Server-root voice exports

After every startup the server writes two comma-separated lists next to the server code (workspace root):

- `voices.txt` — all cloneable voices found in the configured archive.
- `<model_name>_voices.txt` — one per registered model, in selection order: `custom` when the model supports voice design, then its preset voices, then the clone archive (the model's full addressable voice space, regardless of whether prompts have been converted into ICL tensors yet).

Paste `<model_name>_voices.txt` into SillyTavern's **Available Voices** field.

---

## Request Reference

### `POST /v1/audio/speech`

OpenAI-compatible request body:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `input` | string | ✅ | Text to synthesize. Inline tags allowed: `[whisper]`, `[tone: angry]`, `{sad}`, `(sigh)`. A `{...}` directive drives voice design when `voice: "custom"`, and on a cloned voice selects the model's control where its `instruct_map` declares one. |
| `voice` | string | ✅ | Resolved within the selected model: the reserved `custom` design target first, then its custom (preset) voices, then clone stems. Anything else is rejected with `400`. |
| `model` | string | ✅ | Must match a registered model name or a declared alias. Unknown models are rejected with `400`. |
| `response_format` | string | ❌ | `mp3`, `wav`, or `pcm`. When omitted, falls back to `server.response_format` from `config.yaml` (default `mp3`). |
| `speed` | float | ❌ | Accepted for compatibility (not yet applied). |

Returns audio bytes with the matching `Content-Type`.

### `GET /health`

Probes CUDA device health on demand and returns the service status, loaded voice directories, and preset availability:

```json
{
  "status": "ok",
  "cuda_healthy": true,
  "custom_presets": ["<preset>", "…"],
  "cloned_voices": ["AliceW", "Arnold"],
  "voices_dir": "drive:\\path\\to\\voices"
}
```

*Note: If CUDA is frozen or unresponsive (e.g. after PC sleep), `"status"` returns `"degraded"` and `"cuda_healthy"` returns `false`.*

### `GET /v1/voices`

Returns a structured JSON list of all available voices (presets and clones), each tagged with its type and whether the model supports `instruct` steering.

### `GET /v1/speakers`

Returns each model's **complete built-in speaker catalogue**, introspected live from the model in its worker: `{"catalogue": [{"model": "…", "speakers": [...]}]}`. Unlike `/v1/voices`, this asks the model itself, so the **first call may load the checkpoint** into VRAM. Pass `?model=<name>` to query a single model (unknown → `404`) so a fetch can't pull another model into memory.

`/v1/voices` is the *addressable* set — the manifest-declared presets plus your clone archive, known without loading anything. `/v1/speakers` is what the checkpoint natively knows, typically a superset of the manifest's `presets`.

---

## SillyTavern Integration

1. Start the server.
2. In SillyTavern, open the **TTS** extension and select **OpenAI Compatible** as the provider.
3. Set the **Provider Endpoint** to `http://127.0.0.1:8000/v1/audio/speech`.
4. Paste the server root's `<model_name>_voices.txt` (one per registered model) into **Available Voices (comma separated)**.
5. Select voices per character.

SillyTavern's `openai-compatible.js` always requests `response_format: "mp3"`, which this server natively supports.

---

## Logging

- **Console**: all logs are printed to stdout; this is your live view of requests and memory events.
- **Log file**: the same stream is appended to `tts_server.log` in the repository root.

Notable markers:

| Marker | Meaning |
| --- | --- |
| `[MEMORY] + LOADED model into VRAM` | A model was brought into GPU memory on first use. |
| `[MEMORY] - UNLOADING model from VRAM` | A model is being freed (idle timeout or shutdown). |
| `[MEMORY] - UNLOADED model from VRAM` | VRAM freed and CUDA cache cleared. |
| `[MEMORY] + ADDED voice clone prompt` | A cloned voice's prompt was extracted and cached. |
| `[MEMORY] - REMOVED voice clone prompt` | An expired voice prompt was evicted (TTL). |
| `[MEMORY] - CLEARED all ... voice clone prompt(s)` | Prompt cache cleared (startup/shutdown). |
| `Executing ... synthesis` (adapter-defined) | Each model adapter logs its own routing decisions; the exact strings come from the adapter (the `faster-qwen3-tts` example logs `Executing Base voice clone synthesis`, `Executing CustomVoice preset synthesis`, and `attempting VoiceDesign synthesis`). |

---

## Potential Enhancements

### Operational Improvements
- **Dependency Validation**: Add startup checks for required dependencies (FFmpeg, CUDA, PyTorch version)
- **Configuration Validation**: Validate config.yaml structure and required fields at startup
- **Hot-Reload Support**: Support configuration reload without server restart
- **FFmpeg Discovery**: Automatic FFmpeg path discovery with fallback options

### Security Features
- **Authentication**: Basic auth or API key support for production deployments
- **Rate Limiting**: Request rate limiting to prevent abuse
- **CORS Configuration**: Configurable CORS settings for web applications

### Monitoring & Diagnostics
- **Health Checks**: More detailed health endpoint with model loading status
- **Performance Metrics**: Request latency, memory usage, and GPU utilization metrics
- **Prometheus Integration**: Export metrics for monitoring systems

### Deployment Features
- **Docker Support**: Containerized deployment with optimized base images
- **Systemd Service**: Production service configuration for Linux systems
- **Windows Service**: Native Windows service installation

---

## Code Quality Assessment

### Architectural Strengths
- **Separation of Concerns**: Clear separation between server, manager, and worker components
- **Thread Safety**: Proper use of threading.Lock for shared resource access
- **Resource Management**: Efficient GPU memory management with idle timeout unloading
- **Error Handling**: Comprehensive error handling with appropriate logging
- **Performance Optimization**: Model caching and prompt caching for repeated requests

### Code Quality Indicators
- **Type Hints**: Extensive use of Python type hints throughout codebase
- **Logging**: Well-structured logging with appropriate log levels
- **Documentation**: Comprehensive docstrings and inline comments
- **Modular Design**: Clean module separation with single responsibility principle
- **Configuration Hierarchy**: Sensible configuration precedence (CLI > env vars > config file > defaults)

---

## Troubleshooting

- **Port 8000 already in use** — another `python -m server` instance is running. Stop it (or run with `--port <other>`).
- **"is not a valid model identifier"** — the model isn't cached and can't be reached (or the repo ID is wrong). Verify the checkpoint exists (see the model's own docs, or the `faster-qwen3-tts` example in [Example integrations](#example-integrations)) and that `enable_prefetch` can reach `huggingface.co`.
- **Voice not resolving as a clone** — confirm the `.wav` is inside `voices_dir` and restarted the server after adding it. The fast lookup runs at startup.
- **`faster-whisper` not installed** — install the `transcribe` extra before using `transcribe_voices` (see [Server environment](#server-environment)).
- **`ffmpeg` missing for MP3** — install `ffmpeg` and ensure it is on `PATH` (used by `pydub`).
- **High variance with voice design** — a natural-language voice-design request samples a fresh vocal identity each time. For a stable character, generate one design, save it as a reference clip in `voices_dir`, and use that voice name for subsequent turns.