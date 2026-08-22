# subtitle-factory

A one-command subtitle pipeline for a video or audio file: extract audio, take or fetch a transcript, drop the segments too short to be readable, render ASS subtitles with a CJK fallback style, optionally burn or embed them, and write QA frames and a validation report.

## What it does / why

Subtitling a finished video is a chain of small ffmpeg invocations and format conversions, and the failure modes are boring: a one-second cue nobody can read, a Korean character that renders as a box because the font has no glyph, a burn-in that silently produced nothing. This runs the chain once and writes down what each stage did.

Stages, in order. Each records `ok`, `skipped` with a reason, or an error, into the report:

1. **Extract audio** — 16 kHz mono WAV via ffmpeg.
2. **Transcribe** — only with `--run-transcribe`, and only through an external CLI you supply (see below). Skipped otherwise.
3. **Load transcript** — from `--transcript-json`, `--transcript-text`, the transcription stage, or `--placeholder-text`.
4. **Filter segments** — by duration (see the rule below).
5. **Translate** — only with `--translate`. Pluggable; without `--translator-command` it writes a stub and copies the original text through.
6. **Generate ASS** — with a primary style and a separate fallback style for a different CJK font.
7. **Burn / embed** — `--burn` hard-renders into the picture; `--embed` adds a soft subtitle track.
8. **QA** — still frames spread across the runtime plus a contact sheet, so you can see the subtitles rather than trust that they are there.

### Segment filtering rule

| Duration | Result |
| --- | --- |
| ≥ 10 s | Kept |
| 5–10 s | Kept |
| 3–5 s | Kept only if the text is substantive |
| < 3 s | Dropped, unless `--keep-short` |

"Substantive" is a conservative text heuristic: enough Latin words, enough CJK/Korean/Japanese characters, or enough normalised content length. Everything dropped is written to `segments.dropped.json`, so nothing disappears silently.

## Requirements

- Python 3.10 or newer. Standard library only.
- `ffmpeg` on `PATH`. Every media stage shells out to it.
- For **burning**: the fonts named by `--font` and `--fallback-font` must be installed on the machine, because libass resolves them by name at burn time. The defaults are `Noto Sans CJK TC` and `Noto Sans CJK KR`. A missing font does not fail the burn — it renders nothing or renders boxes. Check the QA contact sheet.
- For **transcription only**: an external transcription CLI, plus whatever credentials it needs.

### The transcription CLI is not bundled

`--run-transcribe` shells out to a separate program. **This repository does not ship one**, because the one it was written against is third-party Apache-2.0 code from an unrelated package. Point it at your own with `--transcribe-cli`, or the `TRANSCRIBE_CLI` environment variable.

Whatever you point it at is invoked as:

```
<python> <transcribe-cli> <audio.wav> --model <model> --response-format <text|json|diarized_json> --out-dir <dir> [--language <lang>]
```

and is expected to leave its result at `<dir>/<audio-stem>.transcript.txt` for `text`, or `<dir>/<audio-stem>.transcript.json` otherwise. If the file is not there afterwards, the stage is recorded as producing nothing and the run continues.

**You do not need any of this to use the tool.** `--transcript-json` accepts a timed transcript from any source — faster-whisper, whisper.cpp, a vendor API, a hand-written file — and that is the better path for production timing anyway.

## Install

```
git clone <repo-url>
cd subtitle-factory
python subtitle_factory.py --help
```

## Usage

From an existing timed transcript:

```powershell
python subtitle_factory.py --input input.mp4 --transcript-json transcript.json
```

From an SRT or plain-text transcript:

```powershell
python subtitle_factory.py --input input.mp4 --transcript-text transcript.srt
```

Accepted JSON shapes:

- `[{"start": 0.0, "end": 6.2, "text": "..."}]`
- `{"segments": [...]}`
- `{"utterances": [...]}`
- `{"text": "..."}` — untimed fallback; becomes one cue spanning the whole file

Dry run with no transcript at all, to check audio extraction, font fallback and the QA sheet:

```powershell
python subtitle_factory.py --input input.mp4 --placeholder-text "first cue|second cue|third cue"
```

Burn or embed:

```powershell
python subtitle_factory.py --input input.mp4 --transcript-json transcript.json --burn
python subtitle_factory.py --input input.mp4 --transcript-json transcript.json --embed
```

Live transcription, if you have supplied a CLI:

```powershell
$env:TRANSCRIBE_CLI = "C:\path\to\your_transcriber.py"
python subtitle_factory.py --input input.mp4 --language zh --run-transcribe
```

Translation, with your own translator:

```powershell
python subtitle_factory.py `
  --input input.mp4 `
  --transcript-json transcript.json `
  --translate `
  --target-language ko `
  --translator-command "python translate_segments.py --in {input} --out {output} --lang {language}"
```

Your command receives `{input}` (a JSON segment list), `{output}` (where the translated JSON is expected) and `{language}`.

### Selected flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--input` | required | Video or audio file. |
| `--out-dir` | `<repo>/output` | Parent for job folders. Override with `SUBTITLE_FACTORY_OUT`. |
| `--job-name` | input stem + timestamp | Fixed job folder name. |
| `--keep-short` | off | Keep segments under 3 s. |
| `--font` / `--fallback-font` | `Noto Sans CJK TC` / `Noto Sans CJK KR` | ASS style fonts. |
| `--font-size` | derived from the video height | Override the computed size. |
| `--skip-qa` / `--qa-frames N` | off / `6` | Skip QA rendering, or cap the number of stills. |
| `--language` | — | Hint passed to transcription and recorded in the report. |

## Output

One job folder per run:

| File | Contents |
| --- | --- |
| `audio_16k_mono.wav` | The extracted audio. |
| `segments.filtered.json` | The cues that survived filtering. |
| `segments.dropped.json` | The ones that did not, with the reason. |
| `subtitles.ass` | Two styles — the primary and the CJK fallback — sized to the video's resolution. |
| `video.burned.mp4` | Only with `--burn`. |
| `video.embedded.mp4` | Only with `--embed`. |
| `qa/frames/*.jpg` | Stills spread across the runtime, with subtitles composited. |
| `qa/contact_sheet.jpg` | Those stills as one image. |
| `validation_report.json` / `.md` | Every stage, its status, and every output path. |

## Limitations

- **Burning cannot verify itself.** libass renders missing glyphs as nothing or as boxes and ffmpeg still exits `0`. The QA contact sheet exists precisely because of this — look at it.
- **A plain-text transcript with no timings becomes one cue** spanning the whole file. Use timed JSON, or SRT-style text where the timestamps are preserved.
- **The translation stage is a hook, not a translator.** Without `--translator-command` it copies the source text through and says so in the report.
- **The 3–5 second "substantive" test is a heuristic** tuned for mixed Latin/CJK text. It will occasionally drop a short line you wanted.
- **ASS styling is fixed** beyond font, size and the fallback style: position, outline, shadow and margins are computed from the video height and are not exposed as flags.
- **The fallback style is a style, not per-glyph fallback.** ASS has no automatic per-character font substitution here; a cue mixing scripts still renders in one font.
- **Everything shells out to ffmpeg**, so an ffmpeg build without libass cannot burn, and one without the right encoders cannot embed.
- **No tests.**

## License

MIT. See [LICENSE](LICENSE).
