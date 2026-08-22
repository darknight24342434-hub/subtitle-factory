#!/usr/bin/env python3
"""End-to-end subtitle factory pipeline.

Stages:
- ffmpeg audio extraction
- optional transcription through an external CLI supplied by the caller
- segment filtering by readable-duration rules
- optional pluggable translation
- ASS subtitle generation
- optional hard burn or soft embed
- QA frames and contact sheet
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"
DEFAULT_TRANSCRIBE_FORMAT = "text"
DEFAULT_FALLBACK_FONTS = [
    "Noto Sans CJK TC",
    "Noto Sans CJK KR",
    "Microsoft JhengHei",
    "Malgun Gothic",
    "Arial Unicode MS",
    "Arial",
]
TIMESTAMP_RE = re.compile(
    r"(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2}(?:[,.]\d{1,3})?)"
)
SRT_RANGE_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}(?:[,.]\d{1,3})?)\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}(?:[,.]\d{1,3})?)"
)


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = ""
    keep: bool = True
    decision: str = ""
    reason: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def default_output_root() -> Path:
    env_path = os.getenv("SUBTITLE_FACTORY_OUT")
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parent / "output"


def default_transcribe_cli() -> Path:
    env_path = os.getenv("TRANSCRIBE_CLI")
    if env_path:
        return Path(env_path)
    return Path.home() / ".codex" / "skills" / "transcribe" / "scripts" / "transcribe_diarize.py"


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found on PATH")
    return ffmpeg


def run_command(
    args: list[str],
    report: dict[str, Any],
    stage: str,
    *,
    check: bool = True,
    cwd: Path | None = None,
    shell: bool = False,
) -> subprocess.CompletedProcess[str]:
    started = time.time()
    command_for_report: str | list[str] = args if not shell else " ".join(args)
    entry: dict[str, Any] = {
        "stage": stage,
        "command": command_for_report,
        "cwd": str(cwd) if cwd else None,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    report.setdefault("commands", []).append(entry)
    completed = subprocess.run(
        args if not shell else command_for_report,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=shell,
    )
    entry.update(
        {
            "returncode": completed.returncode,
            "elapsed_seconds": round(time.time() - started, 3),
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"{stage} failed with exit code {completed.returncode}: "
            f"{completed.stderr[-1000:] or completed.stdout[-1000:]}"
        )
    return completed


def parse_duration(raw: str) -> float | None:
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", raw)
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def probe_media(input_path: Path, report: dict[str, Any]) -> dict[str, Any]:
    completed = run_command(
        ["ffmpeg", "-hide_banner", "-i", str(input_path)],
        report,
        "probe",
        check=False,
    )
    raw = completed.stdout + completed.stderr
    video_match = re.search(r"Video:.*?,\s*(\d{2,5})x(\d{2,5})", raw)
    audio_match = re.search(r"Audio:\s*([^,\n]+)", raw)
    info = {
        "duration_seconds": parse_duration(raw),
        "has_video": "Video:" in raw,
        "has_audio": "Audio:" in raw,
        "width": int(video_match.group(1)) if video_match else None,
        "height": int(video_match.group(2)) if video_match else None,
        "audio_codec": audio_match.group(1).strip() if audio_match else None,
    }
    report["probe"] = info
    return info


def extract_audio(input_path: Path, job_dir: Path, report: dict[str, Any]) -> Path | None:
    out_path = job_dir / "audio_16k_mono.wav"
    try:
        run_command(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(out_path),
            ],
            report,
            "extract_audio",
        )
    except RuntimeError as exc:
        report.setdefault("stages", []).append(
            {"stage": "extract_audio", "status": "failed", "message": str(exc)}
        )
        raise
    report.setdefault("stages", []).append(
        {"stage": "extract_audio", "status": "ok", "output": str(out_path)}
    )
    return out_path


def run_transcribe_if_requested(
    args: argparse.Namespace,
    audio_path: Path | None,
    job_dir: Path,
    report: dict[str, Any],
) -> Path | None:
    if not args.run_transcribe:
        report.setdefault("stages", []).append(
            {"stage": "transcribe", "status": "skipped", "reason": "--run-transcribe not set"}
        )
        return None
    if audio_path is None:
        report.setdefault("stages", []).append(
            {"stage": "transcribe", "status": "skipped", "reason": "audio extraction unavailable"}
        )
        return None
    if not os.getenv("OPENAI_API_KEY"):
        report.setdefault("stages", []).append(
            {
                "stage": "transcribe",
                "status": "skipped",
                "reason": "OPENAI_API_KEY is not set; set it locally before live transcription",
            }
        )
        return None

    transcribe_cli = Path(args.transcribe_cli)
    if not transcribe_cli.exists():
        raise RuntimeError(f"Transcribe CLI not found: {transcribe_cli}")

    transcribe_dir = job_dir / "transcribe"
    transcribe_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(transcribe_cli),
        str(audio_path),
        "--model",
        args.transcribe_model,
        "--response-format",
        args.transcribe_response_format,
        "--out-dir",
        str(transcribe_dir),
    ]
    if args.language:
        cmd.extend(["--language", args.language])
    run_command(cmd, report, "transcribe")
    ext = "txt" if args.transcribe_response_format == "text" else "json"
    transcript_path = transcribe_dir / f"{audio_path.stem}.transcript.{ext}"
    report.setdefault("stages", []).append(
        {"stage": "transcribe", "status": "ok", "output": str(transcript_path)}
    )
    return transcript_path if transcript_path.exists() else None


def parse_timestamp(value: str) -> float:
    match = TIMESTAMP_RE.fullmatch(value.strip())
    if not match:
        raise ValueError(f"Invalid timestamp: {value}")
    seconds = match.group("s").replace(",", ".")
    return int(match.group("h")) * 3600 + int(match.group("m")) * 60 + float(seconds)


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if TIMESTAMP_RE.fullmatch(text):
        return parse_timestamp(text)
    try:
        return float(text)
    except ValueError:
        return None


def text_from_item(item: dict[str, Any]) -> str:
    for key in ("text", "transcript", "content", "sentence"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def segment_from_item(item: dict[str, Any]) -> Segment | None:
    start = coerce_float(
        item.get("start")
        if "start" in item
        else item.get("start_time", item.get("begin", item.get("from")))
    )
    end = coerce_float(
        item.get("end") if "end" in item else item.get("end_time", item.get("stop", item.get("to")))
    )
    text = text_from_item(item)
    if start is None or end is None or not text:
        return None
    speaker = str(item.get("speaker", item.get("speaker_id", ""))).strip()
    return Segment(start=start, end=end, text=text, speaker=speaker)


def parse_json_transcript(path: Path, duration: float | None) -> list[Segment]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        candidates = data
    elif isinstance(data, dict):
        for key in ("segments", "utterances", "items", "results"):
            if isinstance(data.get(key), list):
                candidates = data[key]
                break
        else:
            text = text_from_item(data)
            if text:
                return [Segment(start=0.0, end=duration or 10.0, text=text)]
            candidates = []
    else:
        candidates = []

    segments: list[Segment] = []
    for item in candidates:
        if isinstance(item, dict):
            segment = segment_from_item(item)
            if segment:
                segments.append(segment)
    return segments


def parse_srt_text(raw: str) -> list[Segment]:
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    segments: list[Segment] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        range_match = SRT_RANGE_RE.search(line)
        if not range_match:
            index += 1
            continue
        start = parse_timestamp(range_match.group("start"))
        end = parse_timestamp(range_match.group("end"))
        index += 1
        text_lines: list[str] = []
        while index < len(lines) and lines[index].strip():
            text_lines.append(lines[index].strip())
            index += 1
        text = " ".join(text_lines).strip()
        if text:
            segments.append(Segment(start=start, end=end, text=text))
    return segments


def parse_plain_text(path: Path, duration: float | None) -> list[Segment]:
    raw = path.read_text(encoding="utf-8-sig")
    srt_segments = parse_srt_text(raw)
    if srt_segments:
        return srt_segments
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return []
    text = " ".join(lines)
    return [Segment(start=0.0, end=duration or 10.0, text=text)]


def make_placeholder_segments(text: str, duration: float | None) -> list[Segment]:
    parts = [part.strip() for part in re.split(r"\s*\|\s*", text) if part.strip()]
    if not parts:
        return []
    total = max(duration or 12.0, float(len(parts)) * 4.0)
    usable = max(1.0, total - 1.0)
    slot = usable / len(parts)
    segments: list[Segment] = []
    cursor = 0.5
    for idx, part in enumerate(parts):
        end = total - 0.5 if idx == len(parts) - 1 else min(total - 0.5, cursor + slot - 0.2)
        segments.append(Segment(start=round(cursor, 3), end=round(max(cursor + 3.0, end), 3), text=part))
        cursor = end + 0.2
    return segments


def load_segments(
    args: argparse.Namespace,
    live_transcript: Path | None,
    duration: float | None,
    report: dict[str, Any],
) -> tuple[list[Segment], str | None]:
    source: Path | None = None
    source_kind: str | None = None
    if args.transcript_json:
        source = Path(args.transcript_json)
        source_kind = "json"
    elif args.transcript_text:
        source = Path(args.transcript_text)
        source_kind = "text"
    elif live_transcript:
        source = live_transcript
        source_kind = "json" if live_transcript.suffix.lower() == ".json" else "text"

    if source:
        if not source.exists():
            raise RuntimeError(f"Transcript not found: {source}")
        segments = (
            parse_json_transcript(source, duration)
            if source_kind == "json"
            else parse_plain_text(source, duration)
        )
        report.setdefault("stages", []).append(
            {
                "stage": "load_transcript",
                "status": "ok",
                "source": str(source),
                "source_kind": source_kind,
                "segments": len(segments),
            }
        )
        return segments, str(source)

    if args.placeholder_text:
        segments = make_placeholder_segments(args.placeholder_text, duration)
        report.setdefault("stages", []).append(
            {
                "stage": "load_transcript",
                "status": "ok",
                "source": "placeholder",
                "segments": len(segments),
            }
        )
        return segments, "placeholder"

    report.setdefault("stages", []).append(
        {
            "stage": "load_transcript",
            "status": "skipped",
            "reason": "no transcript source and no placeholder text",
        }
    )
    return [], None


def is_substantive(text: str) -> bool:
    normalized = re.sub(r"<[^>]+>", " ", text)
    normalized = re.sub(r"[^\w\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]+", " ", normalized)
    normalized = normalized.strip().lower()
    if not normalized:
        return False
    filler = {"um", "uh", "hmm", "mmm", "ah", "oh", "okay", "ok"}
    if normalized in filler:
        return False
    latin_words = re.findall(r"[a-z0-9]+", normalized)
    cjk_chars = re.findall(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", normalized)
    return len(latin_words) >= 3 or len(cjk_chars) >= 6 or len(normalized) >= 12


def filter_segments(
    segments: list[Segment],
    report: dict[str, Any],
    *,
    keep_short: bool,
    duration: float | None,
) -> tuple[list[Segment], list[Segment]]:
    kept: list[Segment] = []
    dropped: list[Segment] = []
    for segment in segments:
        if duration is not None:
            segment.start = max(0.0, min(segment.start, duration))
            segment.end = max(0.0, min(segment.end, duration))
        if segment.end <= segment.start:
            segment.keep = False
            segment.decision = "drop"
            segment.reason = "invalid or zero-length timing"
        elif segment.duration >= 10.0:
            segment.keep = True
            segment.decision = "keep"
            segment.reason = "spoken segment >=10s"
        elif segment.duration >= 5.0:
            segment.keep = True
            segment.decision = "keep"
            segment.reason = "spoken segment 5-10s"
        elif segment.duration >= 3.0:
            segment.keep = is_substantive(segment.text)
            segment.decision = "keep" if segment.keep else "drop"
            segment.reason = "3-5s substantive" if segment.keep else "3-5s not substantive"
        elif keep_short:
            segment.keep = True
            segment.decision = "keep"
            segment.reason = "<3s kept by --keep-short"
        else:
            segment.keep = False
            segment.decision = "drop"
            segment.reason = "<3s dropped by default"
        (kept if segment.keep else dropped).append(segment)

    report.setdefault("stages", []).append(
        {
            "stage": "filter_segments",
            "status": "ok",
            "input_segments": len(segments),
            "kept": len(kept),
            "dropped": len(dropped),
        }
    )
    return kept, dropped


def write_segments(job_dir: Path, kept: list[Segment], dropped: list[Segment]) -> tuple[Path, Path]:
    kept_path = job_dir / "segments.filtered.json"
    dropped_path = job_dir / "segments.dropped.json"
    kept_path.write_text(
        json.dumps([asdict(segment) for segment in kept], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    dropped_path.write_text(
        json.dumps([asdict(segment) for segment in dropped], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return kept_path, dropped_path


def translate_segments(
    args: argparse.Namespace,
    segments: list[Segment],
    job_dir: Path,
    report: dict[str, Any],
) -> list[Segment]:
    if not args.translate:
        report.setdefault("stages", []).append(
            {"stage": "translate", "status": "skipped", "reason": "--translate not set"}
        )
        return segments

    input_path = job_dir / "translate_input.json"
    output_path = job_dir / "translate_output.json"
    input_path.write_text(
        json.dumps([asdict(segment) for segment in segments], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.translator_command:
        command = args.translator_command.format(
            input=str(input_path),
            output=str(output_path),
            language=args.target_language or "",
        )
        run_command([command], report, "translate", cwd=job_dir, shell=True)
        translated = parse_json_transcript(output_path, None)
        report.setdefault("stages", []).append(
            {
                "stage": "translate",
                "status": "ok",
                "mode": "translator_command",
                "output": str(output_path),
                "segments": len(translated),
            }
        )
        return translated

    output_path.write_text(input_path.read_text(encoding="utf-8"), encoding="utf-8")
    report.setdefault("stages", []).append(
        {
            "stage": "translate",
            "status": "stubbed",
            "reason": "no --translator-command or translator API configured; copied original text",
            "output": str(output_path),
        }
    )
    return segments


def ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    centiseconds = int(round(seconds * 100))
    cs = centiseconds % 100
    total_seconds = centiseconds // 100
    s = total_seconds % 60
    total_minutes = total_seconds // 60
    m = total_minutes % 60
    h = total_minutes // 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def ass_escape(text: str) -> str:
    text = text.replace("{", "(").replace("}", ")")
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("\\n", "\\N")


def wrap_subtitle(text: str, width: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= width:
        return text
    if " " in text:
        return "\\N".join(textwrap.wrap(text, width=width, break_long_words=False))
    return "\\N".join(text[idx : idx + width] for idx in range(0, len(text), width))


def write_ass(
    segments: list[Segment],
    job_dir: Path,
    media_info: dict[str, Any],
    args: argparse.Namespace,
    report: dict[str, Any],
) -> Path | None:
    if not segments:
        report.setdefault("stages", []).append(
            {"stage": "ass_generation", "status": "skipped", "reason": "no kept segments"}
        )
        return None
    width = media_info.get("width") or 1920
    height = media_info.get("height") or 1080
    font_name = args.font or DEFAULT_FALLBACK_FONTS[0]
    fallback_font = args.fallback_font or DEFAULT_FALLBACK_FONTS[1]
    font_size = args.font_size or max(32, min(58, int(height * 0.058)))
    margin_v = max(36, int(height * 0.075))
    wrap_width = max(18, min(34, int(width / max(font_size, 1) * 1.35)))
    out_path = job_dir / "subtitles.ass"

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
            "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
            "MarginL, MarginR, MarginV, Encoding"
        ),
        (
            f"Style: Default,{font_name},{font_size},&H00FFFFFF,&H000000FF,"
            f"&H00111111,&H99000000,0,0,0,0,100,100,0,0,1,2.4,0,2,"
            f"80,80,{margin_v},1"
        ),
        (
            f"Style: FallbackKR,{fallback_font},{font_size},&H00FFFFFF,&H000000FF,"
            f"&H00111111,&H99000000,0,0,0,0,100,100,0,0,1,2.4,0,2,"
            f"80,80,{margin_v},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for segment in segments:
        speaker = ass_escape(segment.speaker)
        text = wrap_subtitle(ass_escape(segment.text), wrap_width)
        lines.append(
            f"Dialogue: 0,{ass_time(segment.start)},{ass_time(segment.end)},Default,"
            f"{speaker},0,0,0,,{text}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report.setdefault("stages", []).append(
        {
            "stage": "ass_generation",
            "status": "ok",
            "output": str(out_path),
            "font": font_name,
            "fallback_font": fallback_font,
            "segments": len(segments),
        }
    )
    return out_path


def burn_subtitles(
    input_path: Path,
    ass_path: Path | None,
    job_dir: Path,
    report: dict[str, Any],
) -> Path | None:
    if ass_path is None:
        report.setdefault("stages", []).append(
            {"stage": "burn", "status": "skipped", "reason": "ASS file unavailable"}
        )
        return None
    out_path = job_dir / "video.burned.mp4"
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vf",
            "ass=subtitles.ass",
            "-c:a",
            "copy",
            str(out_path),
        ],
        report,
        "burn",
        cwd=job_dir,
    )
    report.setdefault("stages", []).append(
        {"stage": "burn", "status": "ok", "output": str(out_path)}
    )
    return out_path


def embed_subtitles(
    input_path: Path,
    ass_path: Path | None,
    job_dir: Path,
    report: dict[str, Any],
    language: str | None,
) -> Path | None:
    if ass_path is None:
        report.setdefault("stages", []).append(
            {"stage": "embed", "status": "skipped", "reason": "ASS file unavailable"}
        )
        return None
    out_path = job_dir / "video.embedded.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-i",
        str(ass_path),
        "-map",
        "0",
        "-map",
        "1:0",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-c:s",
        "mov_text",
    ]
    if language:
        cmd.extend(["-metadata:s:s:0", f"language={language}"])
    cmd.append(str(out_path))
    run_command(cmd, report, "embed")
    report.setdefault("stages", []).append(
        {"stage": "embed", "status": "ok", "output": str(out_path)}
    )
    return out_path


def qa_times(segments: list[Segment], duration: float | None, max_frames: int) -> list[float]:
    times = [segment.start + segment.duration / 2 for segment in segments if segment.duration > 0]
    if not times and duration:
        times = [duration * 0.25, duration * 0.5, duration * 0.75]
    cleaned: list[float] = []
    for value in times:
        if duration:
            value = min(max(value, 0.1), max(0.1, duration - 0.1))
        if not any(abs(value - existing) < 0.5 for existing in cleaned):
            cleaned.append(value)
    return cleaned[:max_frames]


def run_qa(
    input_path: Path,
    ass_path: Path | None,
    segments: list[Segment],
    job_dir: Path,
    duration: float | None,
    report: dict[str, Any],
    max_frames: int,
) -> dict[str, Any]:
    qa_dir = job_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = qa_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    vf_overlay = ["-vf", "ass=subtitles.ass"] if ass_path else []
    frame_paths: list[str] = []
    for idx, seconds in enumerate(qa_times(segments, duration, max_frames), start=1):
        frame_path = frame_dir / f"frame_{idx:02d}_{seconds:.2f}s.jpg"
        cmd = ["ffmpeg", "-y", "-ss", f"{seconds:.3f}", "-i", str(input_path)]
        cmd.extend(vf_overlay)
        cmd.extend(["-frames:v", "1", "-q:v", "2", str(frame_path)])
        run_command(cmd, report, "qa_frame", cwd=job_dir)
        frame_paths.append(str(frame_path))

    contact_sheet = qa_dir / "contact_sheet.jpg"
    vf_parts = []
    if ass_path:
        vf_parts.append("ass=subtitles.ass")
    vf_parts.extend(["fps=1/5", "scale=320:-1", "tile=3x2"])
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vf",
            ",".join(vf_parts),
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(contact_sheet),
        ],
        report,
        "qa_contact_sheet",
        cwd=job_dir,
    )
    result = {"frames": frame_paths, "contact_sheet": str(contact_sheet)}
    report.setdefault("stages", []).append({"stage": "qa", "status": "ok", **result})
    return result


def write_reports(report: dict[str, Any], job_dir: Path) -> tuple[Path, Path]:
    json_path = job_dir / "validation_report.json"
    md_path = job_dir / "validation_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Subtitle Factory Validation Report",
        "",
        f"- Input: `{report.get('input')}`",
        f"- Job directory: `{report.get('job_dir')}`",
        f"- Created: {report.get('created_at')}",
        f"- OPENAI_API_KEY: {report.get('environment', {}).get('openai_api_key')}",
        "",
        "## Stages",
    ]
    for stage in report.get("stages", []):
        status = stage.get("status")
        name = stage.get("stage")
        reason = stage.get("reason") or stage.get("message") or ""
        suffix = f" - {reason}" if reason else ""
        lines.append(f"- {name}: {status}{suffix}")
    outputs = report.get("outputs", {})
    if outputs:
        lines.extend(["", "## Outputs"])
        for key, value in outputs.items():
            if value:
                lines.append(f"- {key}: `{value}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build subtitles from media and transcripts.")
    parser.add_argument("--input", required=True, help="Input video or audio path")
    parser.add_argument(
        "--out-dir",
        default=str(default_output_root()),
        help="Output root or job directory (default: sibling 03.產出/subtitle_factory)",
    )
    parser.add_argument("--language", help="Optional language hint for transcription and metadata")
    parser.add_argument("--run-transcribe", action="store_true", help="Call bundled OpenAI transcription CLI")
    parser.add_argument("--transcript-json", help="Timed transcript JSON with segments")
    parser.add_argument("--transcript-text", help="Transcript TXT/SRT; SRT timestamps are preserved")
    parser.add_argument("--burn", action="store_true", help="Hard-burn ASS subtitles into MP4")
    parser.add_argument("--embed", action="store_true", help="Soft-embed subtitles into MP4")
    parser.add_argument("--keep-short", action="store_true", help="Keep <3s subtitle segments")
    parser.add_argument(
        "--placeholder-text",
        help="Validation-only text. Separate multiple timed placeholder cues with |",
    )
    parser.add_argument("--translate", action="store_true", help="Run translation stage")
    parser.add_argument("--target-language", help="Target language for translation stage")
    parser.add_argument(
        "--translator-command",
        help=(
            "Command template for external translator. Available placeholders: "
            "{input}, {output}, {language}"
        ),
    )
    parser.add_argument("--font", help="ASS primary font (default: Noto Sans CJK TC)")
    parser.add_argument("--fallback-font", help="ASS fallback style font (default: Noto Sans CJK KR)")
    parser.add_argument("--font-size", type=int, help="ASS font size override")
    parser.add_argument("--skip-qa", action="store_true", help="Skip QA frames and contact sheet")
    parser.add_argument("--qa-frames", type=int, default=6, help="Maximum QA still frames")
    parser.add_argument(
        "--transcribe-cli",
        default=str(default_transcribe_cli()),
        help="Path to an external transcription CLI. Not bundled with this repo; "
             "see the README for the command contract it must satisfy. "
             "Can also be set with the TRANSCRIBE_CLI environment variable.",
    )
    parser.add_argument(
        "--transcribe-model",
        default=DEFAULT_TRANSCRIBE_MODEL,
        help=f"Transcription model (default: {DEFAULT_TRANSCRIBE_MODEL})",
    )
    parser.add_argument(
        "--transcribe-response-format",
        default=DEFAULT_TRANSCRIBE_FORMAT,
        choices=["text", "json", "diarized_json"],
        help=f"Transcription response format (default: {DEFAULT_TRANSCRIBE_FORMAT})",
    )
    parser.add_argument(
        "--job-name",
        help="Optional fixed job folder name under --out-dir; default uses input stem and timestamp",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    transcript_sources = [args.transcript_json, args.transcript_text]
    if sum(1 for item in transcript_sources if item) > 1:
        raise SystemExit("Use only one of --transcript-json or --transcript-text")
    if args.transcribe_response_format == "diarized_json" and "transcribe-diarize" not in args.transcribe_model:
        raise SystemExit("diarized_json requires a diarization transcription model")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    ensure_ffmpeg()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    out_root = Path(args.out_dir).expanduser().resolve()
    if args.job_name:
        job_dir = out_root / args.job_name
    else:
        job_dir = out_root / f"{input_path.stem}_{timestamp_slug()}"
    job_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(input_path),
        "job_dir": str(job_dir),
        "environment": {
            "openai_api_key": "present" if os.getenv("OPENAI_API_KEY") else "missing",
            "transcribe_cli": str(args.transcribe_cli),
            "ffmpeg": shutil.which("ffmpeg"),
        },
        "config": {
            "language": args.language,
            "run_transcribe": args.run_transcribe,
            "burn": args.burn,
            "embed": args.embed,
            "translate": args.translate,
            "target_language": args.target_language,
            "keep_short": args.keep_short,
            "transcribe_model": args.transcribe_model,
            "transcribe_response_format": args.transcribe_response_format,
        },
        "stages": [],
        "commands": [],
        "outputs": {},
    }

    try:
        media_info = probe_media(input_path, report)
        audio_path = extract_audio(input_path, job_dir, report) if media_info.get("has_audio") else None
        if audio_path is None:
            report.setdefault("stages", []).append(
                {"stage": "extract_audio", "status": "skipped", "reason": "input has no audio stream"}
            )
        live_transcript = run_transcribe_if_requested(args, audio_path, job_dir, report)
        segments, transcript_source = load_segments(
            args, live_transcript, media_info.get("duration_seconds"), report
        )
        kept, dropped = filter_segments(
            segments,
            report,
            keep_short=args.keep_short,
            duration=media_info.get("duration_seconds"),
        )
        kept_path, dropped_path = write_segments(job_dir, kept, dropped)
        report["outputs"]["filtered_segments"] = str(kept_path)
        report["outputs"]["dropped_segments"] = str(dropped_path)
        report["transcript_source"] = transcript_source

        final_segments = translate_segments(args, kept, job_dir, report)
        ass_path = write_ass(final_segments, job_dir, media_info, args, report)
        report["outputs"]["ass"] = str(ass_path) if ass_path else None

        burned_path = burn_subtitles(input_path, ass_path, job_dir, report) if args.burn else None
        if not args.burn:
            report.setdefault("stages", []).append(
                {"stage": "burn", "status": "skipped", "reason": "--burn not set"}
            )
        embedded_path = embed_subtitles(input_path, ass_path, job_dir, report, args.language) if args.embed else None
        if not args.embed:
            report.setdefault("stages", []).append(
                {"stage": "embed", "status": "skipped", "reason": "--embed not set"}
            )
        report["outputs"]["burned_video"] = str(burned_path) if burned_path else None
        report["outputs"]["embedded_video"] = str(embedded_path) if embedded_path else None

        if args.skip_qa:
            report.setdefault("stages", []).append(
                {"stage": "qa", "status": "skipped", "reason": "--skip-qa set"}
            )
        else:
            qa_source = burned_path if burned_path else input_path
            qa = run_qa(
                qa_source,
                None if burned_path else ass_path,
                final_segments,
                job_dir,
                media_info.get("duration_seconds"),
                report,
                max(1, args.qa_frames),
            )
            report["outputs"]["qa_contact_sheet"] = qa.get("contact_sheet")
            report["outputs"]["qa_frames"] = qa.get("frames")
    finally:
        json_path, md_path = write_reports(report, job_dir)
        print(f"Wrote report: {json_path}")
        print(f"Wrote report: {md_path}")

    print(f"Job directory: {job_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
