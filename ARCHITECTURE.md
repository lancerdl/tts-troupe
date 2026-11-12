# Architecture

**Audience:** maintainers and AI agents working on this repository.
**Status:** reflects the implemented system as of 2026-11-09.

---

## 1. System overview

An OpenAI-compatible speech-synthesis server on Windows. Each TTS backend runs
in **its own Python virtual environment** as a subprocess, dispatched over
JSON-lines on stdio. No HTTP exists anywhere in the model layer.

```
 SillyTavern / test console (GET /) / any OpenAI-TTS client
        │  POST /v1/audio/speech  { model, voice, input, response_format }
        ▼
 server/server.py  (FastAPI, server/.venv — no torch)
        │  strict model resolution + voice routing + audio encoding
        ▼
 framework/model_manager.py  (ModelManager)
        │  capability routing, per-model locks, idle aging, health gates
        ▼
 framework/proxy_adapter.py  (ProxyAdapter)
        │  spawn, JSON-over-stdio framing, deadlines, retry ladder
        │  (the ONLY place the wire protocol exists)
        ▼
 framework/model_worker.py   (generic harness, executed BY the model's venv)
        ▼
 models/<repo>/adapter.py    (repo adapter — does not know it is proxied)
        ▼
 model library               (the model's own code + its checkpoints)
```

Every worker appears as **two processes** (a `uv` trampoline plus the real
interpreter, because the model venvs are uv-managed). This is expected, not a
leak; treat each pair as one worker.

## 2. Repository layout & dependency rules

| Path | Role | Dependencies |
| --- | --- | --- |
| `server/` | FastAPI app: endpoints, request routing, voice archive, audio encoding, config, logging | FastAPI, uvicorn, numpy, pydub/ffmpeg — **never torch** |
| `framework/` | Generic model layer: adapter protocol + YAML manifest interpreter (`adapter.py`), subprocess harness (`model_worker.py`), proxy (`proxy_adapter.py`), manager (`model_manager.py`), CUDA probe (`cuda_utils.py`), voice cache (`voice_cache.py`) | stdlib + numpy + torch only — **no FastAPI, no server imports** |
| `models/` | **Gitignored drop directory.** Independent model repos cloned here. The upstream repo supplies its own code, dependencies, and `.venv`; you add the integration files on top — a `model.yaml` manifest plus the adapter module it names | whatever the model needs (e.g. transformers 5.x) |
| `tests/` | GPU-free pytest suite (in-process fakes; subprocess shape without spawning) | pytest |

Dependency direction is one-way: `server → framework → (wire) → repo adapter`.
The repo adapter never imports server or framework modules; the generic layer
never imports a repo adapter. The sole integration surface this repo adds to a
model repo is the `runtime:` block in its `model.yaml` — a `git pull` inside
`models/<repo>/` can conflict on that hunk; accepted, re-apply it.

**Why subprocess isolation is mandatory:** models routinely pin dependency
versions that cannot share the server's interpreter — the server venv stays on
its own Python/transformers, and each model venv keeps its own pins (for
example, `transformers==5.15.1`, where 5.20.2 breaks a model).

## 3. Model integration via `model.yaml`

A model integrates by dropping its repo into `models/<name>/` with a
`model.yaml` at its root. Registration (`ModelManager.load`) reads only the
manifest — model code is never imported or executed by the server process.

The manifest shape — every key, its default, which are required, and the
arg/result spec forms — is documented in the annotated template
[`model.yaml.template`](model.yaml.template) at the repo root. Copy it to
`model.yaml` in the model repo and edit; it is the single source of truth.

**Manifest-only capabilities:** `capabilities_from_manifest()` derives the
capability view without importing model code, so a model whose venv has not
been set up yet still registers, advertises presets/aliases, and routes —
synthesis then returns **501** pointing at its setup script. Likewise
`speakers` catalogue queries and `presets_by_model` are manifest-derived;
only synthesis and the speaker catalogue itself touch the model.

## 4. Wire protocol (JSON over stdio)

Lockstep request/response, one outstanding command at a time — the manager
serializes per model, so there is no multiplexing.

- **Transport:** binary-mode pipes (`stdin`/`stdout`). Text mode would
  translate newlines on Windows and corrupt base64 payloads. Requests are one
  JSON line on stdin; responses are one JSON line on stdout carrying
  base64-encoded audio + sample rate, or an error object.
- **stdout is protocol-reserved:** the worker redirects `sys.stdout` to a
  stderr sink at startup, so no library `print` can corrupt the stream. All
  logs flow on stderr; the server re-tags them into `logs/<date>/`.
- **A failed synthesis must not kill the worker** — it returns an error
  object. Only a hang or crash ends the process (see lifecycle).
- **Spawn line** (built by `ProxyAdapter`):
  `models/<repo>/.venv/Scripts/python.exe framework/model_worker.py --repo models/<repo>`
  plus policy argv (`--model-idle-timeout`, `--prompt-ttl`, `--device`,
  `--cuda-health-timeout`, optional `--enable-prefetch`) and
  `PYTHONPATH=<repo root>` so `framework.*` imports inside the model
  interpreter. **TTL policy is server-side only** (argv); the model's
  `model.yaml` carries operational necessities only.
- **`enable_prefetch`** maps to worker-startup HF-cache verification
  (disk-only, best-effort; never downloads into VRAM).

| Command | Purpose |
| --- | --- |
| `ping` | Liveness + CUDA health probe executed inside the worker (tiered: zero-cost when no CUDA context exists; alloc+sync when one does). Returns `cuda_ok`. |
| `synthesize_default` | Preset-voice synthesis (text, voice, instruct). |
| `synthesize_clone` | Reference cloning (text, ABSOLUTE ref path, ref_text, instruct). The worker opens the file itself; audio bytes never cross the pipe. With `prompt_cache`, the harness intercepts: extract → cache → synthesize-with-prompt. |
| `synthesize_create` | Voice-design synthesis (text, description). |
| `speakers` | Complete built-in speaker catalogue from the loaded model (may trigger a checkpoint load on first call). |
| `cache_info` | Voice-cache visibility: memory entries, persisted files, disk bytes. Accurate even while unloaded (disk view). |
| `age` | Run the aging pass now (per-checkpoint idle sweep + voice TTL sweep). Policy-free: the worker applies the settings it received at spawn. |
| `shutdown` | Release + flush + exit. |

**`ping` has four roles:** (1) process liveness (timeout ⇒ hung or dead →
recovery path); (2) the CUDA health probe, run inside the worker where the
context lives; (3) the manager's health gate (`check_ready` /
`ensure_cuda_healthy` map onto it — pre-synthesis with cooldown, forced for
`/health`, and before aging passes); (4) the recovery decision point — a hung
ping is detected and killed from OUTSIDE the corrupted CUDA context, which
in-process recovery can never achieve.

## 5. Lifecycle

- **Lazy spawn:** the worker process exists only after the first request that
  needs the model. Cold start measured at 15–30 s (spawn + checkpoint load +
  CUDA-graph warmup); warm synthesis ~1 s.
- **Deadlines:** synthesis 60 s, spawn/cold-start 300 s, recovery cooldown
  60 s (configurable via `ModelSettings`, which flows to the worker at spawn).
- **Single-retry ladder:** a failed or timed-out wire command kills the
  worker, respawns once, retries the command; a second failure marks the
  adapter **degraded** for the cooldown window. Requests during degradation
  return **503** immediately instead of stacking on a dead pipe.
- **EOF = worker exit:** a dead worker is detected by pipe EOF and respawned
  on the next request; a hang is detected by ping/synthesis timeout and
  killed.
- **Idle aging:** a background cleaner thread (30 s tick) runs the CUDA health
  gate, then (a) unloads models idle past `model_idle_timeout` minutes (for a
  proxy: `shutdown`; the adapter stays registered for respawn) and (b) calls
  each adapter's `age_caches()` (voice TTL + per-checkpoint sweeps inside the
  worker). Aging is skipped while CUDA is unresponsive.
- **Shutdown:** idempotent; stops the cleaner, sends `shutdown` to every
  worker (process death frees all VRAM including the CUDA context), marks the
  manager closed. Also registered via `atexit`.

## 6. Request routing (strict, model-pinned)

`POST /v1/audio/speech` requires a `model` that resolves to exactly one
registered adapter — canonical name first, then declared aliases, both
case-insensitive (`ModelManager.resolve_model`). Unresolvable ⇒ `400` with
the available names. There is **no** cross-model voice fallback.

Voice resolution happens **within the pinned model** (per-voice checkpoint
selection; immune to name collisions across models). Text conditioning runs
in order: global `<tag>` stripping (`strip_tags`, content preserved) →
per-model `[tag]` stripping (`strip_brackets`, whole token removed, only for
models with no native use) → `{instruction}` extraction (marker always
removed; the value is passed as `instruct` and reaches the model only if the
capability's `args` mapping binds it) → the empty-input silence check:

1. **Design target first** — `voice` equal to the reserved `custom`
   (case-insensitive) → `synthesize_create`, with the extracted `{instruction}`
   value as the description. Requires `voice_create` and a non-empty
   directive; otherwise `400`.
2. **Preset voices second** — case-insensitive match against that model's
   manifest `presets` → `synthesize_default` (the adapter picks its
   preset checkpoint).
3. **Clone voices third** — case-insensitive match against the voice archive
   stems, only if the model supports `voice_clone` → `synthesize_clone`
   (the clone checkpoint; reference audio + transcript).
4. Otherwise `400` ("voice not available for model"). An unmatched voice
   string is never read as a description.

| Condition | HTTP |
| --- | --- |
| Unknown model | 400 |
| Voice unresolved for the pinned model | 400 |
| Unsupported format (mp3 / wav / pcm only) | 400 |
| Model venv absent (`ModelNotInitializedError`) | 501 |
| Adapter degraded / CUDA unresponsive (`ModelNotReadyError`) | 503 |
| Synthesis failure (`SynthesisError`) | 500 |

## 7. Voice archive pipeline

The cloneable-voice archive is a plain directory of reference clips,
configured by `voices.voices_dir` in `server/config.yaml` (precedence:
`--voices-dir` argv → `TTS_VOICES_DIR` env → yaml → `voices` default, with a
legacy `server/voices` fallback). Each voice is `<Name>.wav` plus an optional
transcript.

Boot order (FastAPI lifespan):

1. **Transcription pass** (best-effort, non-fatal): `transcribe_voice_library`
   transcribes every `.wav` that lacks a `<Name>.reference.txt`, via
   faster-whisper (`base.en`, CUDA float16 → CPU int8 fallback), writing
   `<Name>.reference.txt`. A no-op when the archive is fully transcribed;
   failures never block startup.
2. **Scan** (`scan_voice_library`): every `.wav` (recursive) becomes a
   cloneable voice; the transcript is `<Name>.reference.txt` — the only
   accepted source. Plain `.txt` files are user meta-data and are never read
   as transcripts.
3. **Exports to the workspace root**:
   - `voices.txt` — all cloneable archive voices (comma-separated).
   - `<model_name>_voices.txt` — one per registered model, in selection
     order: `custom` when the model declares `voice_create`, then its manifest
     presets, then the clone archive (the model's full addressable voice
     space, regardless of ICL-conversion state).

The scan result feeds request routing (clone voices) and the console's voice
picker. The scan runs once at startup; new clips require a restart (the
transcript re-read, by contrast, happens per request, so transcript edits
apply live).

## 8. Voice cache ownership (ICL prompts)

Cloned-voice prompts are **harness-owned** when a manifest declares
`prompt_cache`: on a clone miss, `model_worker` calls the adapter's
`extract` hook, stores the returned tensors via `framework/voice_cache.py`,
and thereafter serves clones through the `synthesize` hook with the cached
prompt (`ref_audio`/`ref_text` are still passed through). The repo adapter
knows nothing about caching.

`PersistentPromptCache` properties:

- **Keying:** content-hash fingerprint — sha256 of the reference `.wav` plus
  the ref_text — so renaming or re-recording a voice invalidates naturally.
  Keys are versioned (`adapter` + `format` versions from the manifest), so a
  format change forces re-extraction.
- **Two tiers:** a TTL memory mirror (fast path, torch tensors on GPU/CPU)
  over an LRU-indexed disk store of `<sha256>.pt` files (atomic writes;
  corrupt files are detected and regenerated).
- **Aging:** per-checkpoint idle sweeps; dropping a checkpoint (Base⇒cache)
  releases its memory entries while the `.pt` files survive worker restarts.
- **Visibility:** `cache_info` over the wire and `cache_status()` in
  `/health` expose memory entries, persisted voices, and disk bytes.

## 9. Concurrency & process model

- **Manager:** one `threading.Lock` per adapter serializes synthesis; the
  manager lock protects registration/lookup maps. `_last_accessed` drives
  idle aging.
- **Workers are single-flight** (lockstep protocol); concurrent requests for
  the same model queue on the adapter lock.
- **Cleaner thread** (daemon, 30 s tick): CUDA gate → model aging → cache
  aging. Skips passes while CUDA is unresponsive.
- **Logging:** `setup_server_logging` writes console output plus date-routed
  files under `logs/<date>/`; worker stderr lines are re-tagged with a
  `worker.<model>` prefix. Midnight-safe date routing.

## 10. Endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/audio/speech` | Synthesis (OpenAI-compatible). `model` required. |
| `GET /v1/voices` | Ordered flat list (design target, presets, clones) + `presets_by_model` + `design_by_model`; `?model=` → `{custom, cloned}` with `custom` listing the design target first when supported. |
| `GET /v1/speakers` | Full built-in speaker catalogue; `?model=` scopes to one adapter (only that checkpoint may load). |
| `GET /health` | CUDA status, presets, cloned voices, per-model voice-cache status. |
| `GET /` | Self-contained test console (static page). |

## 11. Notes for future agents

- **Do not add torch to the server venv**; do not add FastAPI to `framework/`.
  The split is what makes model isolation work.
- **`models/` is disposable.** Deleting a repo (or its `.venv`) leaves the
  server functional — the model registers from its manifest and returns 501
  until its setup script runs. Re-cloning only loses local edits such as the
  `runtime:` block (re-apply it).
- **Checkpoints** (HuggingFace): each model repo documents the checkpoints it
  needs — cloning, presets, and voice design may be separate checkpoints or one
  and the same. `enable_prefetch` verifies them in the local cache at startup.
  A model may also map several native tasks onto the three capabilities (see
  the `faster-qwen3-tts` and `Ming-3.0-tts` examples in the README).
- **transformers pins:** model venvs must keep their own versions — check each
  repo's `requirements.txt`; the server venv's versions are independent.
- **Tests:** `python -m pytest tests -q` from the server venv — GPU-free;
  subprocess behavior is exercised via in-process fakes shaped like the proxy.
- **Timing expectations:** cold synthesis 15–30 s (spawn + load + CUDA-graph
  warmup); warm ~1 s; first clone of a voice adds prompt extraction (~1 s).
- **Deferred, deliberately:** streaming audio chunks (the wire protocol keeps
  a framed-binary upgrade path); a VRAM pre-check or 503-retry for the Stable
  Diffusion coexistence boundary (that boundary belongs to the SD side — TTS
  can age itself out of VRAM but cannot evict SD's memory); a queueing rework
  (lock-based serialization is adequate for single-user load; the
  `queue.Queue` + worker-thread change is mechanical if bursts appear).
- **Acceptance checks worth re-running:** a clone-voice synthesis returns
  audio while the log shows the worker spawned from
  `models/<repo>/.venv`, and the server process imports no torch;
  VRAM returns progressively under `nvidia-smi` as the server idles (stale
  checkpoints at the timeout, CUDA context after full idle); a hung ping after
  Windows sleep recovers via worker restart; editing a `.reference.txt`
  changes the fingerprint and re-clones without a server restart.
- **The three text delimiters are not interchangeable.** `<tag>` is global
  markup, stripped for every model (`strip_tags`). `[tag]` is model-native
  content — e.g. some models speak `[Sigh]`/`[Laughter]` as vocal events — so it
  is stripped only where the manifest sets `strip_brackets`. `{...}` is a
  delivery instruction: always extracted, always removed from the text, and
  bound only if the dispatched capability's `args` mapping names `instruct`.
  When `voice` is the reserved `custom` target, that same extracted value is
  used as the voice-design description instead of delivery steering.
  Never route instructions through `[..]`; that silently deletes native
  content on models that speak it.
- **On the clone path the instruction is resolved through `instruct_map`.**
  A declared map translates client terms to model-native labels (`{sad}` →
  悲伤); a miss is dropped (or passed raw, per `instruct_on_miss`), so an
  unrecognized directive degrades to a plain clone rather than a bad control.
  Without `accepts_instruct`, the instruction is discarded and logged, exactly
  as for any other capability.
- **An unbound instruction is logged, not raised.** `MappedAdapter._dispatch`
  emits an INFO line naming the model, the capability, and the discarded
  value when a non-empty `instruct` arrives at a capability that does not
  bind it. If steering seems to have no effect, check
  `logs/<date>/<model>.log` before assuming a bug.
- **A model may have reachable tasks that are not mapped.** If the model's
  native API exposes task or control modes beyond the three capabilities (for
  example a dedicated emotion task), those stay unreachable until the manifest
  maps them. Binding one means a new adapter method plus a capability entry in
  that repo — not here.
- **How the repo's code is located.** An *installable package* is imported
  through the editable install its venv carries (`__editable__.*.pth` +
  finder). A *loose-module repo* has no packaging metadata at all — no
  `pyproject.toml`, no `setup.py` — so `_import_module_from_dir` adds its
  directory to `sys.path` at spawn time, resolved from the `--repo` argument
  the worker is started with. Either shape is valid for the next model.
