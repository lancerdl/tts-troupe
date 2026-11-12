# TTS Troupe

**A troupe of voices for your text-to-speech needs.**

TTS Troupe is a self-hosted, OpenAI-compatible text-to-speech server for Windows
and NVIDIA GPUs. It gives chat frontends that speak the OpenAI audio API (such as
SillyTavern) a single endpoint backed by whichever TTS model you choose to install.
No models are bundled: you drop a model repository into `models/`, describe it with
a small manifest and adapter, and the server routes requests to it.

## Why TTS Troupe

TTS Troupe is built to provide a single TTS server that's capable of handling a
variety of requests and without committing to a single model:

- **One API, many models.** Synthesis is delegated to pluggable model adapters.
  Adding, swapping, or removing a model is a filesystem operation — no server
  code changes.
- **Conflicting dependencies stop being a problem.** Each model runs in its own
  Python virtual environment as a subprocess. A model that pins
  `transformers==5.15.1` and one that needs a newer version can coexist; the
  server venv stays torch-free.
- **Three ways to get a voice.** Built-in presets, zero-shot cloning from a
  reference clip, and natural-language voice design — all behind the same
  `voice` field.
- **Predictable voice routing.** Every request names a model, and voices resolve
  inside that model. There is no cross-model fallback and no name collision
  between models.
- **VRAM that plays well with other GPU apps.** Models load lazily, unload after
  an idle timeout, and clone prompts are cached with a TTL. TTS Troupe ages
  itself out of VRAM when idle, so it can share a GPU with ComfyUI or Stable
  Diffusion.
- **Recovers from the states that usually wedge a GPU server.** CUDA health
  probes, bounded timeouts, and worker restarts handle hung contexts (including
  after Windows sleep); a degraded model returns `503` instead of stacking
  requests on a dead pipe.
- **Works with the clients you already use.** OpenAI-compatible
  `POST /v1/audio/speech`, `mp3`/`wav`/`pcm` output, and a voice-list export
  ready to paste into apps like SillyTavern.
- **Voice library tooling included.** Reference clips are auto-transcribed with
  faster-whisper on boot, and per-model voice lists are exported for frontends.
- **A test console in the browser.** Pick a model and voice, synthesize, play
  the result, and watch cache and CUDA status without writing a client.

## Use cases

### As an OpenAI-compatible TTS endpoint

Any client that can call OpenAI's audio speech API can call TTS Troupe. Point it
at `http://127.0.0.1:8000/v1/audio/speech` and send `model`, `voice`, `input`,
and optionally `response_format`.

### With SillyTavern

1. Start the server.
2. In SillyTavern, open the **TTS** extension and select **OpenAI Compatible**
   as the provider.
3. Set the **Provider Endpoint** to
   `http://127.0.0.1:8000/v1/audio/speech`.
4. Set the **Model** field to the TTS Troupe model name or alias you want to
   use, e.g. `qwen3` or `faster-qwen3-tts`. The voice list is model-specific, so
   set this before copying voices.
5. Paste the matching `<model_name>_voices.txt` from the server root into
   **Available Voices (comma separated)**. For the example above, that is
   `faster-qwen3-tts_voices.txt`.
6. Select voices per character.

SillyTavern's `openai-compatible.js` always requests `response_format: "mp3"`,
which this server natively supports.

### From the browser test console

Open `http://127.0.0.1:8000/` after starting the server. The console is covered
in [Manual test mode](#manual-test-mode-the-test-console).

### The three kinds of voice

This is the part that matters most when choosing a model. A model can support
any combination of the three; the server only routes to what the model declares.

#### 1. Preset voices

Built-in speakers shipped by the model. They need no reference audio and are the
most stable choice for a recurring character. The server reads a model's preset
list from a comma-separated `presets.txt` sidecar in the model repo, so the list
is known at registration without loading a checkpoint. If preset voices exist,
they will be near the front of the contents of the model's
`<model_name>_voices.txt` file.

Examples: Qwen3-TTS's `CustomVoice` checkpoint (Vivian, Ryan, Serena, …) and
Ming's `IP` task (`US_Emma`, `UK_George`, …). TTS Troupe calls this capability
**preset voices**.

Where the model supports it, inline tags with the text can steer delivery.

For example:
- `{sad}` to emotionally condition text
- `[sigh]` for non-verbal sounds

These tags will be ignored if the model doesn't understand them. See
[Text conditioning](#text-conditioning) for more details.

#### 2. Voice cloning

Zero-shot cloning from a reference `.wav` in your voice library. The file stem
(without `.wav`) is the voice name clients send. A companion
`<Name>.reference.txt` transcript is optional but dramatically improves clone
quality through In-Context-Learning (ICL) conditioning.

The extracted speaker prompt is cached on first request and reused, subject to
the prompt TTL. This is the right choice when you have a specific voice you can
supply a clip for.

Some models may support inline tags for voice clones like the ones described in
the Preset voices section above.

#### 3. Voice design

Set `voice` to `design` and put the voice description in a `{...}` directive in
`input`:

```json
{"model": "qwen3", "voice": "design", "input": "{a warm elderly scholar} Good morning."}
```

The directive is extracted from the text and passed to the model as the design
description. This gives you an endless supply of one-off voices without touching
the voice library.

A design request samples a fresh vocal identity each time. For a recurring
character, generate one design, save the clip into the voice library, and clone
it thereafter.

#### How a voice is resolved

Every request names a `model` — a canonical adapter name or a declared alias,
case-insensitive. Unknown models are rejected with `400`. The voice then
resolves **within that model**:

```
Incoming request { model, voice, input, response_format }
                        │
                        ▼
     model resolves to a registered adapter?  ── NO → 400 (lists available models)
                        │ YES (pinned)
                        ▼
   1. voice is the reserved design target "design"?
      (the {description} directive in `input` drives it)
                        │ YES → model's voice-design capability
                        │ NO
                        ▼
   2. voice matches a built-in preset name?
      (from <repo>/presets.txt)
                        │ YES → model's preset capability (+ instruct steering)
                        │ NO
                        ▼
   3. voice matches a clone stem in the voice archive?
      (e.g. "AliceW" → AliceW.wav + AliceW.reference.txt)
                        │ YES → model's voice-clone capability
                        │ NO / unsupported → 400
```

An unmatched voice string is never silently read as a description.

#### Text conditioning

Three delimiters are handled differently, and they are not interchangeable:

- **`<tag>`** is global markup. It is stripped for every model (`strip_tags`),
  with the inner text preserved.
- **`{instruction}`** is a delivery instruction. It is always extracted and
  removed from the text, then passed as `instruct` where the model's capability
  accepts it. When `voice` is `design`, the same value becomes the voice-design
  description instead.
- **`[tag]`** is model-native content. Some models speak `[Sigh]` or `[Laughter]`
  as vocal events, so it is left alone by default and stripped only where the
  manifest sets `strip_brackets: true`.

## Manual test mode: the test console

The server serves a self-contained test page at its root —
`http://127.0.0.1:8000/` by default (the `server.host` and `server.port` from
`server/config.yaml`). No build step, no separate frontend.

- **Model** — every registered model and alias.
- **Voice** — the selected model's addressable voices. **Fetch voices** loads
  them in selection order: the `design` target first when supported, then
  built-in presets, then the clone archive.
- **Load full catalogue** — asks the model itself for its complete built-in
  speaker list. Unlike the manifest-derived list, this may load the checkpoint
  into VRAM on the first call, so it is behind an explicit button.
- **Text** — the text to synthesize. With the `design` voice, describe the voice
  in `{curly braces}`.
- **Format** — `mp3`, `wav`, or `pcm`.
- **Synthesize** — plays the result in the browser and surfaces errors such as
  `400` (bad model/voice), `501` (model runtime not initialized, e.g. its venv
  is missing), and `503` (model degraded).
- **Cache panel and CUDA status** — refresh to see loaded models, cached clone
  prompts, disk usage, and CUDA health.

## Requirements

- **Windows 10/11.** The launcher scripts are `.bat` files and the documented
  paths are Windows paths. The Python code itself is portable, but the scripts
  are not.
- **NVIDIA GPU with CUDA.** The server process needs no GPU, but synthesis does.
  Each model's virtual environment needs a CUDA-enabled PyTorch build.
- **Python 3.10+** for the server venv and for each model venv.
- **FFmpeg on `PATH`** for `mp3` output (used by `pydub`).
- **Disk space** for model checkpoints (Hugging Face cache) and reference audio.
- **Optional: `uv`** for faster environment creation. Plain `venv` and `pip`
  work.
- **Optional: `faster-whisper`** (the `transcribe` extra) for auto-transcription
  of reference clips.

## Installation

### 1. Clone this repository

```
git clone https://github.com/lancerdl/tts-troupe
cd tts-troupe
```

### 2. Create the server environment

The server runs from its own virtual environment at the repo root. It has no
torch — models live in their own venvs.

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
`.venv\Scripts\python.exe`. The `transcribe` extra brings in `faster-whisper`
for the offline voice-library transcriber.

### 3. Install at least one model

Clone a model repository into `models/` and run its own setup. Each model owns
its runtime: it declares its own dependencies and installs them into **its own
virtual environment**, which the server never touches. Two worked examples are
in [Example models](#example-models).

### 4. Add the model's integration files

Copy [`model.yaml.template`](model.yaml.template) to `models/<repo>/model.yaml`
and edit it. Copy [`adapter.py.template`](adapter.py.template) to
`models/<repo>/adapter.py` and fill in the factory class — every model needs
one, even if it is only a few lines. See
[Model integration](#model-integration).

### 5. Generate the model's preset list

If the model has built-in preset voices, run `sync_presets.bat` while the server
is stopped. It asks the model for its speaker list and writes
`models/<repo>/presets.txt`, which the server reads at registration. Skip this
for models with no preset voices.

### 6. Register the model

`server/config.yaml` is auto-created from `server/config.yaml.example` on first
run and is gitignored. Add the model repo name:

```yaml
models:
  list: ["faster-qwen3-tts"]
```

### 7. Point at your voice library

```yaml
voices:
  voices_dir: "C:\\path\\to\\voices"
```

### 8. Start the server

```
start_tts_server.bat
```

This is the one-stop entry point: it launches the whole server, and the server
also runs the voice-library transcription pass on boot. You do **not** need to
run `transcribe_voices.bat` as well.

Or directly:

```
python -m server
```

Optional CLI overrides exist, e.g. `--port 8080` or `--voices-dir <path>` — see
`tts-troupe --help` or `python -m server --help`.

The server is ready when you see:

```
Application startup complete.
Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### 9. Verify

Open `http://127.0.0.1:8000/` and synthesize a line, or use `curl`:

```
curl -s http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3", "input": "Hello, world!", "voice": "Vivian", "response_format": "mp3"}' \
  --output hello.mp3
```

`model` and `voice` must match a model you have actually integrated and
registered. For the `faster-qwen3-tts` example, `"model": "qwen3"` and
`"voice": "Vivian"` work.

## Model integration

### Adding a model: the sequence

1. Clone the model repo into `models/<repo>/` and run its own setup.
2. Add `model.yaml` and `adapter.py` at the repo root.
3. If the model has built-in preset voices, run `sync_presets.bat` while the
   server is stopped to write `models/<repo>/presets.txt`.
4. Add the repo name to `models.list` in `server/config.yaml`.
5. Start or restart the server.

The sections below cover each file in detail.

### How a model plugs in

`models/` is a gitignored drop directory. Each subdirectory is an independent
git repository. You add the integration files at its root:

- **`model.yaml`** — required manifest: name, aliases, runtime, factory, and the
  capability mappings.
- **`adapter.py`** — the model host: the factory class that constructs the
  model, plus optional residency and aging hooks. Required, but usually small;
  only the parts the manifest cannot express need code.
- **`presets.txt`** — optional comma-separated list of built-in voices.

The server reads only the manifest at registration. Model code is never imported
or executed by the server process; it runs inside the model's own worker.

### `model.yaml`

The manifest declares:

- **Identity** — `name`, `aliases`, `source`.
- **Runtime** — `mode: subprocess` and the interpreter path (default
  `.venv/Scripts/python.exe`).
- **Setup** — the factory class and an optional config file passed to it.
- **Capabilities** — `voice_preset`, `voice_clone`, and `voice_create`, each
  mapped to a model call with argument and result specs, or implemented natively
  by the adapter.
- **Text handling** — `strip_brackets` for models with no native use of
  `[tag]` tokens.

The annotated [`model.yaml.template`](model.yaml.template) is the full
reference: every key with its required/optional status, default, and the
arg/result spec forms. Copy it to `model.yaml` in the model repo and edit.

### `adapter.py`

The adapter is the model **host**, not the mapping. The mapper lives in
`framework/` (`MappedAdapter`), which reads `model.yaml` and calls model methods
generically. Write `adapter.py` only for:

- the factory class named by `setup.factory`, constructed as
  `factory(config=<config dict>, model_dir=<Path>)`;
- `get_instance(key)` when a capability names an `instance:`;
- optional lifecycle hooks: `is_loaded`, `age_caches`, `age_checkpoints`, and
  `supported_speakers`;
- native synthesis or clone-prompt-cache hooks when the manifest grammar cannot
  express the call.

Everything else — task and argument shaping, structured kwargs, per-instruction
variants, result extraction — belongs in the manifest.

The skeleton is [`adapter.py.template`](adapter.py.template).

### `presets.txt`

A model's built-in voice list is not a manifest key. It lives in a
comma-separated sidecar, `models/<repo>/presets.txt`:

```
Vivian,Serena,Uncle_Fu,Dylan,Eric,Ryan,Aiden,Ono_Anna,Sohee
```

The server reads it at registration — a file read, no model code and no
checkpoint load. Generate it by asking the model, which runs
`supported_speakers()` in the model's own venv:

```
tts-troupe-sync-presets              # every repo under models/
tts-troupe-sync-presets ming qwen3   # repo dir, manifest name, or alias
```

Or double-click `sync_presets.bat` at the repo root, which runs the same command
and forwards any model names you pass to it.

This is a manual, one-off operation. Run it while the TTS server is stopped —
the server reads `presets.txt` only at registration, so restart it afterward. It
is never part of normal server startup.

You can also write the file by hand. Without it the model simply has no preset
voices: preset names fall through to the clone and design checks and then
`400`.

### Two installation shapes

- **An installable package** — the repo ships a `pyproject.toml` plus a setup
  script, installed with `uv pip install -e .` (falling back to `pip`). Its code
  is imported through that install.
- **A directory of loose modules** — the repo ships no packaging metadata, only
  `requirements.txt` and setup steps in its own README. Its code is imported
  from its directory at runtime.

Either shape is valid. The server assumes nothing about how a model is
initialized. If a model's venv is missing, the model still registers
(capabilities and routing work) and synthesis returns `501` with a message
directing you to the model's own setup instructions.

### Example models

Both are upstream projects you clone yourself; nothing is bundled with this
repo.

#### `faster-qwen3-tts` (installable package)

[`faster-qwen3-tts`](https://github.com/andimarafioti/faster-qwen3-tts) is an
installable package exposing the full range of Qwen3-TTS capabilities:

- **Voice cloning** (`Base` checkpoint) — zero-shot cloning from reference audio
  in your local voice library.
- **Preset speakers** (Qwen's `CustomVoice` checkpoint) — built-in voices
  (Vivian, Ryan, Serena, …) with live emotion and tone steering.
- **Voice design** (`VoiceDesign` checkpoint) — synthesize an entirely new voice
  from a natural-language description.

Clone it into `models/` and run its setup once:

```
git clone https://github.com/andimarafioti/faster-qwen3-tts models/faster-qwen3-tts
models\faster-qwen3-tts\setup_windows.bat   # creates models/faster-qwen3-tts/.venv
```

Add a `model.yaml` at the repo root (abridged here — see the annotated
[`model.yaml.template`](model.yaml.template) for the full shape) and register it
in `server/config.yaml`:

```yaml
name: faster-qwen3-tts
aliases: [qwen3]
runtime:
  mode: subprocess
  python: .venv/Scripts/python.exe
# setup: and capabilities: blocks omitted here; see model.yaml.template
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
| `CustomVoice` (Qwen's preset checkpoint) | ✅ | ✅ |
| `VoiceDesign` (voice design) | ❌ | ✅ |

> ⚠️ There is **no** `0.6B-VoiceDesign` release. If you see a 401 or "not a
> valid model identifier" error, check that `design_model` points to the 1.7B
> variant.

#### `Ming-3.0-tts` (directory of loose modules)

[`Ming-3.0-tts`](https://github.com/inclusionAI/Ming-3.0-tts) (0.5b checkpoint)
provides multilingual synthesis with native **paralinguistic inline tags**
(`[Sigh]`, `[Laughter]`, `[Cough]`, … spoken as vocal events), built-in premium
voices (`US_Emma`, `UK_George`, …) via the `IP` task, zero-shot cloning, and
free-form natural-language voice control.

It ships no packaging metadata — no `pyproject.toml`, no `setup.py` — so set up
its venv with the steps in its own README (`requirements.txt`); text
normalization stays off (`pynini` has no reliable Windows wheels). Its code is
imported from the directory at runtime, so a `model.yaml` at its root registers
it like any other model:

```yaml
name: Ming-3.0-tts
aliases: [ming]
runtime:
  mode: subprocess
  python: .venv/Scripts/python.exe
# setup: and capabilities: blocks omitted here; see model.yaml.template
```

```yaml
models:
  list: ["Ming-3.0-tts"]
```

Because that manifest leaves `strip_brackets` unset (default `false`), its
`[Sigh]`-style tags pass through and are spoken as vocal events.

## API reference

### `POST /v1/audio/speech`

OpenAI-compatible request body:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `input` | string | ✅ | Text to synthesize. Inline tags allowed: `[whisper]`, `[tone: angry]`, `{sad}`, `(sigh)`. A `{...}` directive drives voice design when `voice: "design"`, and on a cloned voice selects the model's control where its `instruct_map` declares one. |
| `voice` | string | ✅ | Resolved within the selected model: the reserved `design` target first, then its preset voices, then clone stems. Anything else is rejected with `400`. |
| `model` | string | ✅ | Must match a registered model name or a declared alias. Unknown models are rejected with `400`. |
| `response_format` | string | ❌ | `mp3`, `wav`, or `pcm`. When omitted, falls back to `server.response_format` from `server/config.yaml` (default `mp3`). |
| `speed` | float | ❌ | Accepted for compatibility (not yet applied). |

Returns audio bytes with the matching `Content-Type`.

### `GET /health`

Probes CUDA device health on demand and returns the service status, the
configured voice directory, and per-model cache status:

```json
{
  "status": "ok",
  "cuda_healthy": true,
  "presets": ["<preset>", "…"],
  "cloned_voices": ["AliceW", "Arnold"],
  "voices_dir": "drive:\\path\\to\\voices",
  "caches": {}
}
```

If CUDA is frozen or unresponsive (e.g. after PC sleep), `"status"` returns
`"degraded"` and `"cuda_healthy"` returns `false`.

### `GET /v1/voices`

Returns a structured JSON list of all available voices (design target, presets,
and clones), each tagged with its type. Pass `?model=<name>` to scope the list to
one model.

### `GET /v1/speakers`

Returns each model's **complete built-in speaker catalogue**, introspected live
from the model in its worker:
`{"catalogue": [{"model": "…", "speakers": [...]}]}`. Unlike `/v1/voices`, this
asks the model itself, so the **first call may load the checkpoint** into VRAM.
Pass `?model=<name>` to query a single model (unknown → `404`) so a fetch cannot
pull another model into memory.

`/v1/voices` is the *addressable* set — the manifest-declared presets plus your
clone archive, known without loading anything. `/v1/speakers` is what the
checkpoint natively knows, typically a superset of the manifest's presets.

## Voice library and `voices_dir`

Reference clones live in the folder configured by `voices.voices_dir` (default:
`voices` relative to the server working directory; configure it in
`server/config.yaml`).

```
AliceW.wav
AliceW.reference.txt        <- optional transcript (recommended)
Arnold.wav
...and so on
```

- The **file stem** (without `.wav`) is the voice name clients use, e.g.
  `"AliceW"`, `"Arnold"`.
- A companion transcript (`.reference.txt`) dramatically improves clone quality
  via ICL conditioning.
- Drop new `.wav` files in at any time — they are discovered on server start (or
  by re-running the transcriber).

### Auto-transcription

Transcription is part of the server boot, not a separate step: on startup the
server runs a `faster-whisper` pass over the configured `voices_dir`
(best-effort, non-fatal) and writes `<stem>.reference.txt` for any clip that
lacks a transcript. Launching the server with `start_tts_server.bat` therefore
covers it — you do **not** need to run `transcribe_voices.bat` as well. Clips
dropped into the archive are transcribed on the next boot.

**Validate the output of automated transcription for accuracy and best
results.**

The same pass is exposed as a standalone helper, for re-transcribing without
restarting the server:

```
python -m server.transcribe_voices
```

or double-click `transcribe_voices.bat`.

- Only `<Name>.reference.txt` counts as a transcript; plain `.txt` files are
  treated as user meta-data and ignored.
- Any `.wav` without a `.reference.txt` is transcribed and written as
  `<stem>.reference.txt`.
- The standalone helper also writes a `voices/voices.txt` export; the server's
  boot pass skips that artifact, and the server writes its own exports in the
  server root (see below).

### Server-root voice exports

After every startup the server writes two comma-separated lists next to the
server code (workspace root):

- `voices.txt` — all cloneable voices found in the configured archive.
- `<model_name>_voices.txt` — one per registered model, in selection order:
  `design` when the model supports voice design, then its preset voices, then
  the clone archive (the model's full addressable voice space, regardless of
  whether prompts have been converted into ICL tensors yet).

Paste `<model_name>_voices.txt` into SillyTavern's **Available Voices** field
after setting the matching model in SillyTavern's **Model** field.

## Configuration

All settings live in **`server/config.yaml`**. It is auto-created from
**`server/config.yaml.example`** on first run and is **gitignored**
(machine-specific). The precedence is:

```
CLI arguments  >  environment variables  >  server/config.yaml  >  built-in defaults
```

- `server/config.yaml.example` — committed, portable template. Edit values there
  when you want to change defaults for new installs.
- `server/config.yaml` — your local file. **This is the one you actually edit**
  for this machine. It is never tracked by git.
- If `server/config.yaml` is missing, it is recreated automatically from the
  example on next startup.

The template ships with workable defaults, so you normally change only a couple
of things:

- **`voices.voices_dir`** — point this at your own reference-audio folder
  (default `voices`). This is the one you will almost always change.
- **`models.list`** — the model repo names under `models/` to load.
- **Timeouts** — `server.synthesis_timeout_seconds`,
  `server.cold_start_timeout_seconds`, and `server.recovery_cooldown_seconds`
  bound worker calls on the wire; `memory_management.model_idle_timeout_minutes`
  and `memory_management.prompt_cache_ttl_minutes` govern VRAM and prompt-cache
  aging. Adjust to suit your hardware and usage.

Every setting can also be supplied as an environment variable — handy for
containers, CI, or a one-off run. You do **not** need them when you have a proper
`server/config.yaml`; when both are present the environment variables win (and
command-line flags win over both). The full list is in
[`ARCHITECTURE.md`](ARCHITECTURE.md#13-configuration--environment-variables).

## Troubleshooting

- **Port 8000 already in use** — another `python -m server` instance is running.
  Stop it (or run with `--port <other>`).
- **"is not a valid model identifier"** — the model isn't cached and can't be
  reached (or the repo ID is wrong). Verify the checkpoint exists (see the
  model's own docs, or the `faster-qwen3-tts` example in
  [Example models](#example-models)) and that `enable_prefetch` can reach
  `huggingface.co`.
- **Voice not resolving as a clone** — confirm the `.wav` is inside `voices_dir`
  and restart the server after adding it. The fast lookup runs at startup.
- **`faster-whisper` not installed** — install the `transcribe` extra before
  using `transcribe_voices` (see [Installation](#installation)).
- **`ffmpeg` missing for MP3** — install `ffmpeg` and ensure it is on `PATH`
  (used by `pydub`).
- **High variance with voice design** — a natural-language voice-design request
  samples a fresh vocal identity each time. For a stable character, generate one
  design, save it as a reference clip in `voices_dir`, and use that voice name
  for subsequent turns.
- **Synthesis returns 501** — the model runtime is not initialized (usually a
  missing virtual environment), or no loaded model supports the requested
  interaction. Run the model repo's own setup script.
- **Synthesis returns 503** — the model is degraded after a failed or timed-out
  call. Wait for the recovery cooldown, then retry.

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — for maintainers and AI agents: the wire
  protocol, worker lifecycle, cache ownership, the full adapter contract,
  configuration and environment-variable reference, logging markers, and known
  limitations.
- [`model.yaml.template`](model.yaml.template) — annotated manifest reference.
- [`adapter.py.template`](adapter.py.template) — adapter skeleton.
