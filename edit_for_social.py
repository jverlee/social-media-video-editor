#!/usr/bin/env python3
"""Edit a video for social media: transcribe subtitles, auto-crop to the
speaker(s), and re-encode to mp4.

Usage:
    ./edit_for_social.py input.mp4 [-o output.mp4] [--aspect 9:16]
                         [--no-subs] [--whisper-model base]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Belt-and-suspenders against the libomp double-load crash when PyTorch
# (via whisper) and OpenCV both bring their own OpenMP runtime on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("missing python deps. install with:\n  pip install opencv-python numpy openai-whisper Pillow")


def check_deps() -> None:
    for cmd in ("ffmpeg", "ffprobe"):
        if not shutil.which(cmd):
            sys.exit(f"missing dependency: {cmd} (install with: brew install ffmpeg)")


def probe(path: Path) -> dict:
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of", "json", str(path),
    ])
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "fps": float(num) / float(den),
        "duration": float(s.get("duration", 0) or 0),
    }


def ensure_whisper_model(model_name: str) -> str:
    """Return a local path to the whisper .pt file.

    Whisper's built-in downloader uses urllib, which on Python framework
    builds fails behind corporate MITM proxies that inject a self-signed
    root CA. curl uses the macOS Keychain trust store, which usually has
    those roots, so we download via curl and hand the path to whisper.
    """
    import whisper

    if model_name not in whisper._MODELS:
        return model_name
    url = whisper._MODELS[model_name]
    cache_dir = Path.home() / ".cache" / "whisper"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / Path(url).name
    if target.exists() and target.stat().st_size > 0:
        return str(target)
    if not shutil.which("curl"):
        sys.exit("need curl to download whisper model behind your network's SSL setup")
    print(f"  downloading whisper '{model_name}' model → {target}")
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        subprocess.check_call(["curl", "-L", "--fail", "--progress-bar", "-o", str(tmp), url])
        tmp.replace(target)
    except subprocess.CalledProcessError as e:
        tmp.unlink(missing_ok=True)
        sys.exit(f"failed to download whisper model: {e}")
    return str(target)


Word = tuple[str, float, float]            # (text, start, end)
Segment = tuple[float, float, list[Word]]  # (start, end, words)


def transcribe(video: Path, model_name: str, initial_prompt: str | None = None) -> list[Segment]:
    try:
        import whisper
    except ImportError:
        sys.exit("openai-whisper not installed. run: pip install openai-whisper")
    model_path = ensure_whisper_model(model_name)
    model = whisper.load_model(model_path)
    # word_timestamps=True asks whisper to align each token to a time range so
    # we can highlight the word currently being spoken. initial_prompt biases
    # the decoder toward listed proper nouns / jargon.
    result = model.transcribe(
        str(video), verbose=False, word_timestamps=True,
        initial_prompt=initial_prompt,
    )
    segs: list[Segment] = []
    for s in result["segments"]:
        words: list[Word] = []
        for w in s.get("words", []) or []:
            wt = w["word"].strip()
            if wt:
                words.append((wt, float(w["start"]), float(w["end"])))
        if words:
            segs.append((float(s["start"]), float(s["end"]), words))
    return segs


_CAPTION_HEADER_RE = re.compile(r"^\[\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*\]\s*$")


def review_captions(segments: list[Segment]) -> list[Segment]:
    """Open the transcript in $EDITOR so the user can confirm or edit it.

    The file shows one segment per block with a `[start - end]` marker; the user
    edits the text below each marker. On save, segments whose text is unchanged
    keep whisper's per-word timings exactly; edited ones get their words spread
    evenly across the segment's time range.
    """
    if not segments:
        return segments
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
    if not shutil.which(editor.split()[0]):
        print(f"  $EDITOR ({editor!r}) not found — skipping caption review")
        return segments

    lines = [
        "# Review captions. Edit text below each [start - end] marker, then save & quit.",
        "# Lines starting with # are ignored. Blank a segment's text to drop it.",
        "# Pass --no-review-captions to skip this step.",
        "",
    ]
    for (s_start, s_end, words) in segments:
        text = " ".join(w[0] for w in words)
        lines.append(f"[{s_start:.2f} - {s_end:.2f}]")
        lines.append(text)
        lines.append("")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="captions-",
    ) as tf:
        tf.write("\n".join(lines))
        path = tf.name

    print(f"opening captions in {editor}…")
    try:
        # Use a shell so $EDITOR values like "code -w" or "subl -w" work.
        subprocess.check_call(f'{editor} "{path}"', shell=True)
    except subprocess.CalledProcessError as e:
        os.unlink(path)
        sys.exit(f"editor exited with error: {e}")

    with open(path) as f:
        edited = f.read()
    os.unlink(path)
    return _parse_reviewed_captions(edited, segments)


def _parse_reviewed_captions(text: str, originals: list[Segment]) -> list[Segment]:
    # Key by the printed time range so we can detect "user didn't touch this
    # segment" and preserve the original per-word timings.
    orig_by_range: dict[tuple[str, str], list[Word]] = {
        (f"{s:.2f}", f"{e:.2f}"): words for (s, e, words) in originals
    }
    segs: list[Segment] = []
    cur: tuple[str, str] | None = None
    cur_lines: list[str] = []

    def flush() -> None:
        nonlocal cur, cur_lines
        if cur is None:
            return
        s_start = float(cur[0])
        s_end = float(cur[1])
        body = " ".join(ln.strip() for ln in cur_lines).strip()
        cur_lines = []
        key = cur
        cur = None
        if not body:
            return
        toks = body.split()
        if not toks:
            return
        orig_words = orig_by_range.get(key)
        if orig_words is not None and body == " ".join(w[0] for w in orig_words):
            segs.append((s_start, s_end, orig_words))
            return
        # Spread tokens evenly across the segment range.
        span = (s_end - s_start) / len(toks) if s_end > s_start else 0.01
        words: list[Word] = [
            (tok, s_start + i * span, s_start + (i + 1) * span)
            for i, tok in enumerate(toks)
        ]
        segs.append((s_start, s_end, words))

    for ln in text.splitlines():
        if ln.lstrip().startswith("#"):
            continue
        m = _CAPTION_HEADER_RE.match(ln.strip())
        if m:
            flush()
            cur = (m.group(1), m.group(2))
            continue
        if cur is not None:
            cur_lines.append(ln)
    flush()
    return segs


def load_openrouter_key() -> str | None:
    """Return the OpenRouter API key from $OPENROUTER_API_KEY or .openrouter."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip() or None
    for p in (Path.cwd() / ".openrouter", Path(__file__).resolve().parent / ".openrouter"):
        if p.exists():
            txt = p.read_text().strip()
            if txt:
                return txt
    return None


def _openrouter_chat(api_key: str, model: str, messages: list[dict], timeout: float = 60.0) -> str:
    """POST to OpenRouter's chat-completions endpoint and return the message text.

    Uses curl (not urllib) because Python framework builds on macOS often hit
    SSL cert-verify errors behind MITM proxies — the same reason whisper model
    downloads in this script also shell out to curl.
    """
    if not shutil.which("curl"):
        raise RuntimeError("curl not found; install it or set up Python certs")
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    })
    proc = subprocess.run(
        [
            "curl", "-sS", "--fail-with-body",
            "--max-time", str(int(timeout)),
            "-X", "POST",
            "-H", f"Authorization: Bearer {api_key}",
            "-H", "Content-Type: application/json",
            "-H", "HTTP-Referer: https://github.com/local/edit_for_social",
            "-H", "X-Title: edit_for_social",
            "--data-binary", "@-",
            "https://openrouter.ai/api/v1/chat/completions",
        ],
        input=payload,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"OpenRouter request failed (curl rc={proc.returncode}): {proc.stdout[:400] or proc.stderr[:400]}")
    try:
        body = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"OpenRouter returned non-JSON: {proc.stdout[:400]}") from e
    if "choices" not in body:
        raise RuntimeError(f"OpenRouter error: {body}")
    return body["choices"][0]["message"]["content"]


def _segments_for_prompt(segments: list[Segment]) -> str:
    """One line per segment: `[seconds] text`, suitable for an LLM prompt."""
    lines = []
    for (s_start, _s_end, words) in segments:
        text = " ".join(w[0] for w in words)
        lines.append(f"[{s_start:.2f}] {text}")
    return "\n".join(lines)


def _extract_json_object(text: str) -> dict | None:
    """Pull the first {...} JSON object out of an LLM response (handles fences/prose)."""
    text = text.strip()
    if text.startswith("```"):
        # Strip the code fence.
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def detect_emphasis_llm(
    segments: list[Segment],
    api_key: str,
    model: str,
    max_per_min: float,
    min_spacing_s: float,
    video_duration: float,
) -> list[tuple[float, float, str]]:
    """Ask an LLM via OpenRouter to pick emphasis moments worth a zoom punch.

    Returns a list of (start_s, end_s, reason) snapped to word boundaries and
    de-conflicted by min_spacing_s.
    """
    if not segments:
        return []
    transcript = _segments_for_prompt(segments)
    minutes = max(0.5, video_duration / 60.0)
    soft_cap = max(0, int(round(minutes * max_per_min)))
    sys_prompt = (
        "You are an expert short-form video editor reviewing a transcript of a "
        "spoken video. You pick a small number of moments where a quick zoom-in "
        "would punch up an important point."
    )
    user_prompt = f"""Identify moments in this transcript where a zoom-in would emphasize a meaningful point. The zoom will be held until the end of the sentence the speaker is currently saying, so pick the moment the emphasis starts.

Pick phrases that are:
- A key insight, takeaway, or surprising claim
- An emphatic delivery — the speaker is clearly hitting a point
- An important name, number, or product detail worth highlighting

Do NOT pick:
- Generic intros, outros, pleasantries, or filler
- Adjacent moments — leave at least {min_spacing_s:.0f} seconds between picks

Return at most {soft_cap} picks for this {video_duration:.0f}s clip. Zero is fine if nothing stands out.

Return JSON only, no prose, in this exact shape:
{{"moments": [{{"start": <seconds>, "end": <seconds>, "phrase": "<words>", "why": "<short reason>"}}]}}

Transcript (one segment per line, with start time in seconds):
{transcript}
"""
    raw = _openrouter_chat(api_key, model, [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ])
    parsed = _extract_json_object(raw)
    if not parsed or "moments" not in parsed:
        print(f"  emphasis: LLM returned no usable JSON (got {raw[:160]!r})")
        return []
    moments_raw = parsed.get("moments") or []
    # All words flattened — used for snapping to nearest word boundaries.
    all_words = [w for (_, _, ws) in segments for w in ws]
    picked: list[tuple[float, float, str]] = []
    for m in moments_raw:
        try:
            s = float(m["start"])
            e = float(m["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if e <= s:
            continue
        # Snap start to the nearest word start, then extend end to the end of
        # the sentence the speaker is in — the first word at/after the LLM's end
        # whose text ends in sentence-final punctuation. This keeps the zoom
        # held until the thought finishes instead of popping out mid-phrase.
        if all_words:
            s_word = min(all_words, key=lambda w: abs(w[1] - s))
            s = s_word[1]
            sentence_end: float | None = None
            for (txt, _ws, we) in all_words:
                if we < e:
                    continue
                if txt.rstrip("\"'”’)]}").endswith((".", "!", "?")):
                    sentence_end = we
                    break
            if sentence_end is not None:
                e = sentence_end
            else:
                e_word = min(all_words, key=lambda w: abs(w[2] - e))
                e = e_word[2]
            e = max(e, s + 0.6)
        # Cap absolute length so a runaway sentence (or LLM mistake) can't hold
        # the zoom forever.
        e = min(e, s + 8.0)
        if e - s < 0.6:
            continue
        picked.append((s, e, str(m.get("why") or m.get("phrase") or "")))
    picked.sort(key=lambda m: m[0])
    # Enforce min spacing: keep the earliest, drop anything that starts within
    # min_spacing_s of the previous moment's end.
    final: list[tuple[float, float, str]] = []
    for mt in picked:
        if final and mt[0] - final[-1][1] < min_spacing_s:
            continue
        final.append(mt)
    if soft_cap and len(final) > soft_cap:
        final = final[:soft_cap]
    return final


def _emphasis_envelope(n: int, ramp_f: int, max_zoom: float) -> np.ndarray:
    """Cosine-eased zoom envelope over n frames, peaking at max_zoom."""
    env = np.ones(n, dtype=np.float32)
    for t in range(n):
        if t < ramp_f:
            u = 0.5 - 0.5 * math.cos(math.pi * t / ramp_f)
        elif t > n - ramp_f - 1:
            k = max(0, n - 1 - t)
            u = 0.5 - 0.5 * math.cos(math.pi * k / ramp_f)
        else:
            u = 1.0
        env[t] = 1.0 + (max_zoom - 1.0) * u
    return env


def build_zoom_per_panel(
    moments: list[tuple[float, float, str]],
    panel_tracks_per_frame: list[list[int]],
    speaker_per_frame: np.ndarray,
    total_frames: int,
    fps: float,
    max_zoom: float = 1.15,
    ramp_s: float = 0.18,
) -> list[list[float]] | None:
    """For each emphasis moment, zoom only the panel that's showing the dominant
    active speaker during that window. Returns a per-frame list of per-panel
    zoom factors (parallel to panels_per_frame), or None if there's nothing to do.
    """
    if not moments or total_frames <= 0:
        return None
    zooms: list[list[float]] = [
        [1.0] * len(panel_tracks_per_frame[i]) for i in range(total_frames)
    ]
    ramp_f = max(1, int(round(ramp_s * fps)))
    for (s, e, _why) in moments:
        sf = max(0, int(round(s * fps)))
        ef = min(total_frames, int(round(e * fps)))
        if ef <= sf:
            continue
        # Dominant active speaker (track index) during this window.
        window = speaker_per_frame[sf:ef]
        valid = window[window >= 0]
        if len(valid) > 0:
            vals, counts = np.unique(valid, return_counts=True)
            dominant = int(vals[np.argmax(counts)])
        else:
            dominant = -1
        env = _emphasis_envelope(ef - sf, ramp_f, max_zoom)
        for k, i in enumerate(range(sf, ef)):
            ptracks = panel_tracks_per_frame[i]
            if not ptracks:
                continue
            # Find the panel showing the dominant speaker; if not visible (or
            # unknown), fall back to panel 0 so we still get a punch effect.
            target_p = 0
            for pi, tr in enumerate(ptracks):
                if tr == dominant and dominant >= 0:
                    target_p = pi
                    break
            zooms[i][target_p] = max(zooms[i][target_p], float(env[k]))
    return zooms


def speech_bounds_from_segments(segments: list[Segment]) -> tuple[float, float] | None:
    """First word start and last word end across all segments."""
    words = [w for (_, _, ws) in segments for w in ws]
    if not words:
        return None
    return words[0][1], words[-1][2]


def speech_bounds_silencedetect(
    video: Path, duration: float, noise_db: float = -30.0, min_silence: float = 0.3,
) -> tuple[float, float] | None:
    """Use ffmpeg's silencedetect to find where audio activity starts and ends.

    Returns (first_speech_s, last_speech_s) or None if the file has no audio.
    """
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(video),
            "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
            "-f", "null", "-",
        ],
        capture_output=True, text=True,
    )
    out = proc.stderr
    if "Audio:" not in out:
        return None
    silences: list[tuple[float, float | None]] = []
    cur_start: float | None = None
    for line in out.splitlines():
        m = re.search(r"silence_start:\s*(-?[\d.]+)", line)
        if m:
            cur_start = max(0.0, float(m.group(1)))
            continue
        m = re.search(r"silence_end:\s*([\d.]+)", line)
        if m and cur_start is not None:
            silences.append((cur_start, float(m.group(1))))
            cur_start = None
    if cur_start is not None:
        silences.append((cur_start, None))

    first = 0.0
    last = duration
    if silences and silences[0][0] <= 0.1 and silences[0][1] is not None:
        first = silences[0][1]
    if silences and silences[-1][1] is None:
        last = silences[-1][0]
    if last <= first:
        return None
    return first, last


def shift_segments(segments: list[Segment], offset: float, new_duration: float) -> list[Segment]:
    """Subtract offset from all timestamps; drop anything outside [0, new_duration]."""
    if offset <= 0:
        return segments
    out: list[Segment] = []
    for (s_start, s_end, words) in segments:
        new_words: list[Word] = []
        for (t, ws, we) in words:
            nws = ws - offset
            nwe = we - offset
            if nwe <= 0 or nws >= new_duration:
                continue
            new_words.append((t, max(0.0, nws), min(new_duration, nwe)))
        if new_words:
            out.append((
                max(0.0, s_start - offset),
                min(new_duration, s_end - offset),
                new_words,
            ))
    return out


def trim_source(inp: Path, out: Path, start: float, duration: float) -> None:
    """Re-encode inp[start : start+duration] to out. Re-encoding keeps the cut
    frame-accurate; stream-copy would snap to the previous keyframe."""
    subprocess.check_call([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-i", str(inp),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out),
    ])


def load_replacements(args, input_path: Path) -> list[tuple[str, str]]:
    """Collect (old, new) pairs from CLI flags + auto-loaded files.

    File format: one 'old=new' per line. Blank lines and '#' comments allowed.
    Auto-loaded: <input>.replacements.txt if present in the same directory.
    """
    pairs: list[tuple[str, str]] = []
    files: list[Path] = []
    auto = input_path.with_suffix(input_path.suffix + ".replacements.txt")
    if auto.exists():
        files.append(auto)
    if args.replacements_file:
        files.append(Path(args.replacements_file).expanduser().resolve())
    for f in files:
        for raw in f.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                old, new = line.split("=", 1)
                pairs.append((old.strip(), new.strip()))
    for r in args.replace or []:
        if "=" in r:
            old, new = r.split("=", 1)
            pairs.append((old.strip(), new.strip()))
    return pairs


def apply_replacements(segments: list[Segment], pairs: list[tuple[str, str]]) -> list[Segment]:
    """Apply case-insensitive whole-word replacements to each word's text.

    Word timings are preserved exactly — only the displayed text changes.
    """
    if not pairs:
        return segments
    # Pre-compile patterns for speed. \b…\b matches at word boundaries so
    # trailing punctuation like 'Myr.' / 'Myr,' is still substituted.
    compiled = [(re.compile(r"\b" + re.escape(old) + r"\b", re.IGNORECASE), new) for old, new in pairs]
    new_segs: list[Segment] = []
    for (s_start, s_end, words) in segments:
        new_words: list[Word] = []
        for (text, ws, we) in words:
            t = text
            for pat, new in compiled:
                t = pat.sub(new, t)
            new_words.append((t, ws, we))
        new_segs.append((s_start, s_end, new_words))
    return new_segs


MOUTH_ROI_SIZE = (24, 16)  # downsampled mouth-region crops used for motion scoring


@dataclass
class FaceDet:
    frame: int          # source frame index
    x: int              # bbox top-left x
    y: int              # bbox top-left y
    w: int              # bbox width
    h: int              # bbox height
    mouth: np.ndarray   # downsampled grayscale of lower face — for lip-motion scoring


@dataclass
class Panel:
    """A region cut from the source frame and placed in the output canvas."""
    src_x: int
    src_y: int
    src_w: int
    src_h: int
    dst_x: int
    dst_y: int
    dst_w: int
    dst_h: int


@dataclass
class Track:
    cx: float           # average face center x
    cy: float           # average face center y
    avg_w: float        # average bbox width
    avg_h: float        # average bbox height
    by_frame: dict      # frame_idx -> FaceDet
    # Tile rectangle in source coords — learned from Zoom's green active-speaker
    # border when this track is speaking, or estimated from face size otherwise.
    tile_x: int = 0
    tile_y: int = 0
    tile_w: int = 0
    tile_h: int = 0
    # Final crop rectangle inside the tile (with safety inset to hide the border).
    crop_x: int = 0
    crop_y: int = 0
    crop_w: int = 0
    crop_h: int = 0


def detect_green_border(frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find the Zoom 'active speaker' lime-green border in a frame.

    Returns the largest border-like rectangle (x, y, w, h) or None if no
    plausible border is present. We require:
      - reasonable size (>= 200x150 px)
      - rectangular-ish aspect (0.4 .. 3.0)
      - low fill ratio inside the bounding box (it's an outline, not a solid).
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([40, 80, 80]), np.array([80, 255, 255]))
    if cv2.countNonZero(mask) == 0:
        return None
    # Connect any pixel gaps along the border.
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 200 or h < 150:
            continue
        ar = w / float(h)
        if ar < 0.4 or ar > 3.0:
            continue
        # A real border is mostly hollow; foliage/wallpaper is a solid blob.
        rect_area = w * h
        fill_ratio = cv2.countNonZero(mask[y:y + h, x:x + w]) / float(rect_area)
        if fill_ratio > 0.5:
            continue
        if rect_area > best_area:
            best_area = rect_area
            best = (int(x), int(y), int(w), int(h))
    return best


def detect_faces(
    video_path: Path, sample_every: int
) -> tuple[list[FaceDet], dict[int, tuple[int, int, int, int]], int]:
    """Detect faces and the Zoom active-speaker border every Nth frame.

    Returns (faces, borders_by_frame, total_frames). borders_by_frame maps a
    sample frame index → (x, y, w, h) of the active green border when present.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        sys.exit("could not load face cascade")

    cap = cv2.VideoCapture(str(video_path))
    faces: list[FaceDet] = []
    borders: dict[int, tuple[int, int, int, int]] = {}
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            for (x, y, w, h) in dets:
                my0 = int(y + h * 0.55)
                my1 = int(y + h)
                mx0 = int(x + w * 0.15)
                mx1 = int(x + w * 0.85)
                if my1 > my0 + 4 and mx1 > mx0 + 4:
                    mouth = gray[my0:my1, mx0:mx1]
                    mouth_small = cv2.resize(mouth, MOUTH_ROI_SIZE)
                    faces.append(FaceDet(idx, int(x), int(y), int(w), int(h), mouth_small))
            border = detect_green_border(frame)
            if border is not None:
                borders[idx] = border
        idx += 1
    cap.release()
    return faces, borders, idx


def cluster_tracks(faces: list[FaceDet], src_w: int, src_h: int) -> list[Track]:
    """Greedy spatial clustering of face detections into person-tracks.

    Two detections are the same person if their centers are within ~15% of
    the smaller source dimension. Good enough for Zoom-style fixed tiles.
    """
    if not faces:
        return []
    thresh = min(src_w, src_h) * 0.15
    thresh_sq = thresh * thresh
    tracks: list[Track] = []
    for f in faces:
        fcx = f.x + f.w / 2.0
        fcy = f.y + f.h / 2.0
        best_i = -1
        best_d = float("inf")
        for i, t in enumerate(tracks):
            dx = t.cx - fcx
            dy = t.cy - fcy
            d = dx * dx + dy * dy
            if d < best_d:
                best_d = d
                best_i = i
        if best_i >= 0 and best_d < thresh_sq:
            t = tracks[best_i]
            n = len(t.by_frame)
            t.cx = (t.cx * n + fcx) / (n + 1)
            t.cy = (t.cy * n + fcy) / (n + 1)
            t.avg_w = (t.avg_w * n + f.w) / (n + 1)
            t.avg_h = (t.avg_h * n + f.h) / (n + 1)
            t.by_frame[f.frame] = f
        else:
            tracks.append(Track(cx=fcx, cy=fcy, avg_w=float(f.w), avg_h=float(f.h), by_frame={f.frame: f}))
    # Drop spurious tracks. A real participant is visible in most sample frames;
    # ghosts (paintings, partial detections) appear far less often.
    tracks.sort(key=lambda t: len(t.by_frame), reverse=True)
    if tracks:
        top_count = len(tracks[0].by_frame)
        cutoff = max(30, int(top_count * 0.3))
        tracks = [t for t in tracks if len(t.by_frame) >= cutoff]
    return tracks[:4]  # at most 4 participants in the final cut


def attach_tile_boxes(
    tracks: list[Track],
    borders: dict[int, tuple[int, int, int, int]],
    src_w: int,
    src_h: int,
) -> dict[int, int]:
    """Use the Zoom active-speaker borders to determine each track's tile bbox
    and which track owns the border in each sample frame.

    Returns a mapping sample_frame → track index of the active speaker. A track's
    tile is the median border bbox of samples where that border encloses the
    track's face center.
    """
    border_track: dict[int, int] = {}
    track_borders: list[list[tuple[int, int, int, int]]] = [[] for _ in tracks]
    for frame_idx, (bx, by, bw, bh) in borders.items():
        # Pick the track whose mean face center sits inside the border. If
        # several do, prefer the closest to border center.
        bcx, bcy = bx + bw / 2.0, by + bh / 2.0
        best_i = -1
        best_d = float("inf")
        for i, t in enumerate(tracks):
            if not (bx <= t.cx <= bx + bw and by <= t.cy <= by + bh):
                continue
            d = (t.cx - bcx) ** 2 + (t.cy - bcy) ** 2
            if d < best_d:
                best_d = d
                best_i = i
        if best_i >= 0:
            border_track[frame_idx] = best_i
            track_borders[best_i].append((bx, by, bw, bh))

    for i, t in enumerate(tracks):
        if track_borders[i]:
            xs = [b[0] for b in track_borders[i]]
            ys = [b[1] for b in track_borders[i]]
            ws = [b[2] for b in track_borders[i]]
            hs = [b[3] for b in track_borders[i]]
            t.tile_x = int(np.median(xs))
            t.tile_y = int(np.median(ys))
            t.tile_w = int(np.median(ws))
            t.tile_h = int(np.median(hs))
        else:
            # Fall back to a face-relative tile estimate.
            tw = min(src_w, int(t.avg_w * 3.5))
            th = min(src_h, int(t.avg_h * 2.4))
            t.tile_w = tw
            t.tile_h = th
            t.tile_x = max(0, min(src_w - tw, int(t.cx - tw / 2)))
            t.tile_y = max(0, min(src_h - th, int(t.cy - th / 2)))
    return border_track


def compute_track_framing(
    tracks: list[Track], src_w: int, src_h: int, aspect: tuple[int, int]
) -> None:
    """Compute the final per-tile crop rectangle — the largest output-aspect
    window that fits *inside* the tile bounds, inset to hide the green border."""
    aw, ah = aspect
    for t in tracks:
        inset = max(6, min(t.tile_w, t.tile_h) // 40)  # ~2.5% of tile
        tx = t.tile_x + inset
        ty = t.tile_y + inset
        tw = max(40, t.tile_w - 2 * inset)
        th = max(40, t.tile_h - 2 * inset)

        # Largest aw:ah rectangle that fits inside (tw, th)
        if th * aw <= tw * ah:
            ch = th
            cw = int(round(th * aw / ah))
        else:
            cw = tw
            ch = int(round(tw * ah / aw))
        cw -= cw % 2
        ch -= ch % 2

        # Center horizontally on the face; vertically pin the face to about 40%
        # from the top of the crop for nicer composition.
        cx = int(round(t.cx - cw / 2.0))
        cy = int(round(t.cy - ch * 0.40))
        cx = max(tx, min(tx + tw - cw, cx))
        cy = max(ty, min(ty + th - ch, cy))
        cx = max(0, min(src_w - cw, cx))
        cy = max(0, min(src_h - ch, cy))
        t.crop_x, t.crop_y, t.crop_w, t.crop_h = cx, cy, cw, ch


def compute_speaker_timeline(
    tracks: list[Track],
    border_track: dict[int, int],
    total_frames: int,
    sample_every: int,
    fps: float,
    min_dwell_seconds: float,
) -> np.ndarray:
    """Return a per-frame array of track indices indicating the active speaker.

    Primary signal: Zoom's green active-speaker border (when present). Fallback:
    per-track lip-motion score. Hysteresis prevents rapid cuts.
    """
    K = len(tracks)
    if K == 0:
        return np.zeros(total_frames, dtype=int)
    if K == 1:
        return np.zeros(total_frames, dtype=int)

    sample_frames = sorted({fr for t in tracks for fr in t.by_frame})
    S = len(sample_frames)

    # --- Fallback: lip-motion scoring per track per sample ---
    motion = np.zeros((S, K), dtype=np.float32)
    prev_roi: list[np.ndarray | None] = [None] * K
    last_motion = np.zeros(K, dtype=np.float32)
    for s, fr in enumerate(sample_frames):
        for k in range(K):
            f = tracks[k].by_frame.get(fr)
            if f is None:
                motion[s, k] = last_motion[k] * 0.7
            else:
                if prev_roi[k] is not None:
                    diff = np.abs(f.mouth.astype(np.int16) - prev_roi[k].astype(np.int16))
                    motion[s, k] = float(diff.mean())
                prev_roi[k] = f.mouth
            last_motion[k] = motion[s, k]
    smooth_samples = max(3, int(round(0.6 * fps / sample_every)))
    if smooth_samples > 1:
        kernel = np.ones(smooth_samples, dtype=np.float32) / smooth_samples
        for k in range(K):
            motion[:, k] = np.convolve(motion[:, k], kernel, mode="same")

    # --- Decide per sample, preferring the border signal ---
    min_dwell_samples = max(2, int(round(min_dwell_seconds * fps / sample_every)))
    active_per_sample = np.zeros(S, dtype=int)
    cur = border_track.get(sample_frames[0], int(np.argmax(motion[0])) if S > 0 else 0)
    cur_since = 0
    for s, fr in enumerate(sample_frames):
        if fr in border_track:
            winner = border_track[fr]
        else:
            winner = int(np.argmax(motion[s]))
            if motion[s, winner] < 1e-4:
                winner = cur
        if winner == cur:
            cur_since += 1
        elif cur_since >= min_dwell_samples:
            cur = winner
            cur_since = 1
        else:
            cur_since += 1
        active_per_sample[s] = cur

    per_frame = np.zeros(total_frames, dtype=int)
    for s, fr in enumerate(sample_frames):
        end = sample_frames[s + 1] if s + 1 < S else total_frames
        per_frame[fr:end] = active_per_sample[s]
    if sample_frames:
        per_frame[:sample_frames[0]] = active_per_sample[0]
    return per_frame


def _speaker_order(border_track: dict[int, int]) -> list[int]:
    """Return track indices in the order they first received an active border."""
    seen: list[int] = []
    seen_set: set[int] = set()
    for fr in sorted(border_track):
        k = border_track[fr]
        if k not in seen_set:
            seen.append(k)
            seen_set.add(k)
    return seen


def _panel_for_track(
    t: Track, target_w: int, target_h: int, src_w: int, src_h: int,
    dst_x: int, dst_y: int,
) -> Panel:
    """Build a Panel that crops the largest target-aspect rectangle from inside
    this track's tile and places it at (dst_x, dst_y) sized (target_w, target_h)."""
    inset = max(6, min(t.tile_w, t.tile_h) // 40)
    tx = t.tile_x + inset
    ty = t.tile_y + inset
    tw = max(40, t.tile_w - 2 * inset)
    th = max(40, t.tile_h - 2 * inset)
    # Largest target_w:target_h rectangle inside (tw, th).
    if th * target_w <= tw * target_h:
        ch = th
        cw = int(round(th * target_w / target_h))
    else:
        cw = tw
        ch = int(round(tw * target_h / target_w))
    cw -= cw % 2
    ch -= ch % 2
    sx = int(round(t.cx - cw / 2.0))
    sy = int(round(t.cy - ch * 0.40))
    sx = max(tx, min(tx + tw - cw, sx))
    sy = max(ty, min(ty + th - ch, sy))
    sx = max(0, min(src_w - cw, sx))
    sy = max(0, min(src_h - ch, sy))
    return Panel(sx, sy, cw, ch, dst_x, dst_y, target_w, target_h)


def build_layout(
    faces: list[FaceDet],
    borders: dict[int, tuple[int, int, int, int]],
    total_frames: int,
    src_w: int,
    src_h: int,
    fps: float,
    sample_every: int,
    smooth_seconds: float,
    min_dwell_seconds: float,
    target_w: int,
    target_h: int,
    aspect: tuple[int, int],
    layout_mode: str,                 # "auto" | "single" | "stack"
) -> tuple[list[list[Panel]], list[list[int]], np.ndarray, list[Track], str]:
    """Return (panels_per_frame, panel_tracks_per_frame, speaker_per_frame, tracks, description).

    - panels_per_frame[i] is the list of Panels to composite for output frame i.
    - panel_tracks_per_frame[i][p] is the track index displayed in panel p, or
      -1 if no track is associated (center-crop fallback).
    - speaker_per_frame[i] is the dominant active-speaker track index for frame
      i, or -1 if unknown.
    """
    aw, ah = aspect
    nf = max(1, total_frames)

    if total_frames <= 0 or not faces:
        cw, ch = _fit_aspect(src_w, src_h, aw, ah)
        x0 = (src_w - cw) // 2
        y0 = (src_h - ch) // 2
        panel = Panel(x0, y0, cw, ch, 0, 0, target_w, target_h)
        return (
            [[panel]] * nf,
            [[-1]] * nf,
            np.full(nf, -1, dtype=int),
            [],
            "center crop (no faces)",
        )

    tracks = cluster_tracks(faces, src_w, src_h)
    if not tracks:
        cw, ch = _fit_aspect(src_w, src_h, aw, ah)
        x0 = (src_w - cw) // 2
        y0 = (src_h - ch) // 2
        panel = Panel(x0, y0, cw, ch, 0, 0, target_w, target_h)
        return (
            [[panel]] * total_frames,
            [[-1]] * total_frames,
            np.full(total_frames, -1, dtype=int),
            [],
            "center crop (no tracks)",
        )

    border_track = attach_tile_boxes(tracks, borders, src_w, src_h)
    compute_track_framing(tracks, src_w, src_h, aspect)

    speakers = _speaker_order(border_track)

    # Decide layout
    use_stack = (
        layout_mode == "stack"
        or (layout_mode == "auto" and len(speakers) >= 2)
    )

    if use_stack and len(speakers) >= 2:
        top_t = tracks[speakers[0]]
        bot_t = tracks[speakers[1]]
        half_h = target_h // 2
        half_h -= half_h % 2
        top_panel = _panel_for_track(top_t, target_w, half_h, src_w, src_h, 0, 0)
        bot_panel = _panel_for_track(bot_t, target_w, half_h, src_w, src_h, 0, half_h)
        panels = [top_panel, bot_panel]
        # Both panels are shown every frame; map their track ids so the zoom
        # builder can pick which one to enlarge during an emphasis moment.
        panel_tracks = [speakers[0], speakers[1]]
        active = compute_speaker_timeline(
            tracks, border_track, total_frames, sample_every, fps, min_dwell_seconds
        )
        desc = (
            f"stack: top track {speakers[0]} {top_panel.src_w}x{top_panel.src_h}, "
            f"bottom track {speakers[1]} {bot_panel.src_w}x{bot_panel.src_h}; "
            f"{len(tracks) - 2} other track(s) excluded"
        )
        return (
            [panels] * total_frames,
            [panel_tracks] * total_frames,
            active,
            tracks,
            desc,
        )

    # Single-person OR explicit single layout with multiple tracks: cut between
    # them based on active speaker.
    if len(tracks) == 1:
        t = tracks[0]
        sample_frames = sorted(t.by_frame)
        idxs = np.array(sample_frames, dtype=float)
        cxs = np.array([t.by_frame[i].x + t.by_frame[i].w / 2.0 for i in sample_frames])
        cys = np.array([t.by_frame[i].y + t.by_frame[i].h / 2.0 for i in sample_frames])
        all_idx = np.arange(total_frames)
        cx = np.interp(all_idx, idxs, cxs)
        cy = np.interp(all_idx, idxs, cys)
        smooth_n = max(1, int(round(smooth_seconds * fps)))
        if smooth_n > 1:
            k = np.ones(smooth_n) / smooth_n
            pad = smooth_n // 2
            cx = np.convolve(np.pad(cx, pad, mode="edge"), k, mode="valid")[:total_frames]
            cy = np.convolve(np.pad(cy, pad, mode="edge"), k, mode="valid")[:total_frames]
        w, h = t.crop_w, t.crop_h
        tx0, ty0 = t.tile_x, t.tile_y
        tx1, ty1 = t.tile_x + t.tile_w, t.tile_y + t.tile_h
        x = np.round(cx - w / 2.0).astype(int)
        y = np.round(cy - h * 0.40).astype(int)
        np.clip(x, tx0, max(tx0, tx1 - w), out=x)
        np.clip(y, ty0, max(ty0, ty1 - h), out=y)
        np.clip(x, 0, src_w - w, out=x)
        np.clip(y, 0, src_h - h, out=y)
        panels_per_frame = [
            [Panel(int(x[i]), int(y[i]), w, h, 0, 0, target_w, target_h)]
            for i in range(total_frames)
        ]
        return (
            panels_per_frame,
            [[0]] * total_frames,
            np.zeros(total_frames, dtype=int),
            tracks,
            f"single follow, crop {w}x{h}",
        )

    # Multi-track single-panel: pick active speaker per frame.
    active = compute_speaker_timeline(
        tracks, border_track, total_frames, sample_every, fps, min_dwell_seconds
    )
    panels_per_frame = [
        [Panel(tracks[a].crop_x, tracks[a].crop_y, tracks[a].crop_w, tracks[a].crop_h,
               0, 0, target_w, target_h)]
        for a in active
    ]
    panel_tracks_per_frame = [[int(a)] for a in active]
    return (
        panels_per_frame,
        panel_tracks_per_frame,
        active,
        tracks,
        f"single panel, cuts between {len(tracks)} tracks",
    )


def _fit_aspect(src_w: int, src_h: int, aw: int, ah: int) -> tuple[int, int]:
    """Largest (w, h) of aspect aw:ah that fits inside src_w x src_h, even dims."""
    h = src_h
    w = int(round(h * aw / ah))
    if w > src_w:
        w = src_w
        h = int(round(w * ah / aw))
    return (w - w % 2, h - h % 2)


def pick_font(size: int) -> ImageFont.FreeTypeFont:
    for p in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


def wrap_words(
    words: list[Word],
    font: ImageFont.FreeTypeFont,
    max_width: int,
    draw: ImageDraw.ImageDraw,
) -> list[list[Word]]:
    """Greedy-wrap a list of timed words into lines that fit within max_width."""
    if not words:
        return []
    lines: list[list[Word]] = []
    cur_line: list[Word] = [words[0]]
    cur_text = words[0][0]
    for w in words[1:]:
        trial = cur_text + " " + w[0]
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width:
            cur_line.append(w)
            cur_text = trial
        else:
            lines.append(cur_line)
            cur_line = [w]
            cur_text = w[0]
    lines.append(cur_line)
    return lines


def _zoom_panel(p: Panel, zoom: float, src_w: int, src_h: int) -> Panel:
    """Shrink a panel's source rect around its center by `zoom`, clamped to the
    source frame. dst rect is unchanged, so the panel ends up enlarged."""
    if zoom <= 1.0001:
        return p
    new_w = max(2, int(p.src_w / zoom))
    new_h = max(2, int(p.src_h / zoom))
    new_w -= new_w % 2
    new_h -= new_h % 2
    new_x = p.src_x + (p.src_w - new_w) // 2
    new_y = p.src_y + (p.src_h - new_h) // 2
    new_x = max(0, min(src_w - new_w, new_x))
    new_y = max(0, min(src_h - new_h, new_y))
    return Panel(new_x, new_y, new_w, new_h, p.dst_x, p.dst_y, p.dst_w, p.dst_h)


def composite_with_subs(
    video_path: Path,
    out_path: Path,
    panels_per_frame: list[list[Panel]],
    target_w: int,
    target_h: int,
    src_w: int,
    src_h: int,
    fps: float,
    segments: list[Segment],
    zoom_per_frame: list[list[float]] | None = None,
    intro_frames: int = 0,
    outro_frames: int = 0,
) -> None:
    """Composite each frame's panels onto a target_w x target_h canvas, draw
    subtitles, and stream to ffmpeg as raw BGR24."""
    font_size = max(22, target_h // 28)
    font = pick_font(font_size)
    line_h = font_size + 6
    stroke_w = max(2, font_size // 12)
    stroke_w_active = stroke_w + 2  # thicker outline behind the current word
    color_spoken = (255, 255, 255)       # already said — white
    color_active = (255, 221, 0)         # being said — bold saturated yellow
    color_upcoming = (170, 170, 170)     # not yet said — light gray
    # Background pill behind the text — sits over the seam between speakers.
    bg_alpha = 160                      # 0..255 — ~63% opacity
    bg_pad_x = max(20, font_size // 2)
    bg_pad_y = max(10, font_size // 3)
    bg_radius = max(12, font_size // 2)

    dummy = Image.new("RGB", (target_w, target_h))
    dummy_draw = ImageDraw.Draw(dummy)
    space_w = int(round(dummy_draw.textlength(" ", font=font)))

    # Pre-wrap each segment, and pre-measure each word so we don't redo work per frame.
    @dataclass
    class WrappedWord:
        text: str
        start: float
        end: float
        width: int

    MAX_LINES = 2  # never show more than this many lines at once

    wrapped: list[tuple[float, float, list[list[WrappedWord]], list[WrappedWord]]] = []
    for (s_start, s_end, words) in segments:
        lines_of_words = wrap_words(words, font, int(target_w * 0.9), dummy_draw)
        all_lines: list[list[WrappedWord]] = []
        for line in lines_of_words:
            row: list[WrappedWord] = []
            for (wt, ws, we) in line:
                bbox = dummy_draw.textbbox((0, 0), wt, font=font)
                row.append(WrappedWord(wt, ws, we, bbox[2] - bbox[0]))
            all_lines.append(row)

        # Split into chunks of at most MAX_LINES lines. Each chunk becomes its
        # own display segment so we never have 3+ lines visible at once.
        for i in range(0, len(all_lines), MAX_LINES):
            chunk = all_lines[i:i + MAX_LINES]
            flat = [w for line in chunk for w in line]
            if not flat:
                continue
            chunk_start = flat[0].start
            if i + MAX_LINES < len(all_lines):
                # End right when the next chunk's first word starts so there's
                # no visual overlap.
                chunk_end = all_lines[i + MAX_LINES][0].start
            else:
                chunk_end = max(flat[-1].end, s_end)
            wrapped.append((chunk_start, chunk_end, chunk, flat))

    # Pre-compute swoosh axis once. We overlay the swoosh during the first
    # `intro_frames` and last `outro_frames` of the composited output so the
    # underlying video keeps playing while the animation passes over it.
    swoosh_u_grid: np.ndarray | None = None
    swoosh_u_max: float = 0.0
    if intro_frames > 0 or outro_frames > 0:
        angle = math.radians(_SWOOSH_ANGLE_DEG)
        tan_a = math.tan(angle)
        swoosh_u_max = target_w + target_h * tan_a
        xs = np.arange(target_w, dtype=np.float32)
        ys = np.arange(target_h, dtype=np.float32)
        swoosh_u_grid = xs[None, :] + ys[:, None] * tan_a

    ff = subprocess.Popen(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{target_w}x{target_h}", "-r", f"{fps}",
            "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
    )
    assert ff.stdin is not None

    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    n = len(panels_per_frame)
    seg_i = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            j = idx if idx < n else n - 1
            panels = panels_per_frame[j]
            if zoom_per_frame is not None and j < len(zoom_per_frame):
                zs = zoom_per_frame[j]
                if any(z > 1.0001 for z in zs):
                    panels = [
                        _zoom_panel(p, zs[pi] if pi < len(zs) else 1.0, src_w, src_h)
                        for pi, p in enumerate(panels)
                    ]
            if len(panels) == 1 and panels[0].dst_x == 0 and panels[0].dst_y == 0 \
                    and panels[0].dst_w == target_w and panels[0].dst_h == target_h:
                # Fast path for single full-canvas panel.
                p = panels[0]
                region = frame[p.src_y:p.src_y + p.src_h, p.src_x:p.src_x + p.src_w]
                if region.shape[0] != target_h or region.shape[1] != target_w:
                    crop = cv2.resize(region, (target_w, target_h), interpolation=cv2.INTER_AREA)
                else:
                    crop = region
            else:
                crop = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                for p in panels:
                    region = frame[p.src_y:p.src_y + p.src_h, p.src_x:p.src_x + p.src_w]
                    resized = cv2.resize(region, (p.dst_w, p.dst_h), interpolation=cv2.INTER_AREA)
                    crop[p.dst_y:p.dst_y + p.dst_h, p.dst_x:p.dst_x + p.dst_w] = resized

            if wrapped:
                t = idx / fps
                while seg_i < len(wrapped) and t > wrapped[seg_i][1]:
                    seg_i += 1
                if seg_i < len(wrapped) and wrapped[seg_i][0] <= t <= wrapped[seg_i][1]:
                    _, _, line_data, flat = wrapped[seg_i]
                    # Active word = the last word whose start time is <= t.
                    # That way the highlight advances on each word boundary
                    # and the last word stays highlighted to segment end.
                    active = -1
                    for i, w in enumerate(flat):
                        if t >= w.start:
                            active = i
                        else:
                            break

                    total_h = line_h * len(line_data)
                    row_widths = [
                        sum(w.width for w in row) + space_w * (len(row) - 1)
                        for row in line_data
                    ]
                    max_row_w = max(row_widths) if row_widths else 0
                    # Vertically center the text block on the canvas — that puts
                    # it across the seam between the two stacked speakers.
                    y_start = (target_h - total_h) // 2

                    # Compose the semi-transparent rounded background.
                    bg_x0 = (target_w - max_row_w) // 2 - bg_pad_x
                    bg_x1 = (target_w + max_row_w) // 2 + bg_pad_x
                    bg_y0 = y_start - bg_pad_y
                    bg_y1 = y_start + total_h + bg_pad_y
                    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).convert("RGBA")
                    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                    od = ImageDraw.Draw(overlay)
                    od.rounded_rectangle(
                        [bg_x0, bg_y0, bg_x1, bg_y1],
                        radius=bg_radius,
                        fill=(0, 0, 0, bg_alpha),
                    )
                    img = Image.alpha_composite(img, overlay)
                    draw = ImageDraw.Draw(img)

                    y = y_start
                    word_i = 0
                    for row in line_data:
                        row_w = sum(w.width for w in row) + space_w * (len(row) - 1)
                        x = (target_w - row_w) // 2
                        for j, w in enumerate(row):
                            if word_i == active:
                                fill = color_active
                                sw = stroke_w_active
                            elif word_i < active:
                                fill = color_spoken
                                sw = stroke_w
                            else:
                                fill = color_upcoming
                                sw = stroke_w
                            draw.text(
                                (x, y), w.text, font=font,
                                fill=fill,
                                stroke_width=sw,
                                stroke_fill=(0, 0, 0),
                            )
                            x += w.width + (space_w if j < len(row) - 1 else 0)
                            word_i += 1
                        y += line_h
                    crop = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

            if swoosh_u_grid is not None:
                # Overlay the swoosh on top of the playing video. Intro covers
                # the first `intro_frames` frames (curtain pulling off to reveal
                # the live scene); outro covers the last `outro_frames` frames
                # (curtain sweeping in over the live scene to end on dark).
                if intro_frames > 0 and idx < intro_frames:
                    p = idx / max(1, intro_frames - 1)
                    crop = _render_swoosh_frame(
                        swoosh_u_grid, swoosh_u_max, target_w, target_h,
                        _ease_in_out_cubic(p), crop, "intro",
                    )
                elif outro_frames > 0 and idx >= n - outro_frames:
                    k = idx - (n - outro_frames)
                    p = k / max(1, outro_frames - 1)
                    crop = _render_swoosh_frame(
                        swoosh_u_grid, swoosh_u_max, target_w, target_h,
                        _ease_in_out_cubic(p), crop, "outro",
                    )

            ff.stdin.write(np.ascontiguousarray(crop).tobytes())
            idx += 1
    finally:
        cap.release()
        ff.stdin.close()
        rc = ff.wait()
    if rc != 0:
        sys.exit(f"ffmpeg crop encode failed with code {rc}")


_SWOOSH_DARK_BGR: tuple[int, int, int] = (28, 14, 8)          # near-black, cool deep-navy tint
_SWOOSH_STRIPES_BGR: list[tuple[tuple[int, int, int], float]] = [
    ((230, 110, 30), 0.18),   # deep cobalt   (#1E6EE6)
    ((248, 189, 56), 0.17),   # vivid sky     (#38BDF8)
    ((253, 222, 150), 0.18),  # pale ice-blue (#96DEFD)
]
_SWOOSH_ANGLE_DEG = 18.0
_SWOOSH_GAP_FRAC = 0.018


def _ease_in_out_cubic(p: float) -> float:
    if p < 0.5:
        return 4 * p * p * p
    return 1 - ((-2 * p + 2) ** 3) / 2


def _render_swoosh_frame(
    u_grid: np.ndarray, u_max: float, w: int, h: int,
    p: float, ref_frame: np.ndarray, kind: str,
) -> np.ndarray:
    """Render one swoosh frame.

    The wipe is parameterized along a tilted axis u = x + y*tan(angle). Stripes
    sit centered on `u_wave`. Pixels past the wipe (in the direction of motion)
    show `ref_frame` — for intro that's the revealed first frame, for outro the
    not-yet-covered last frame. Pixels the wipe has passed show the dark color.
    """
    gap = _SWOOSH_GAP_FRAC * u_max
    total_stripe_w = sum(f for _, f in _SWOOSH_STRIPES_BGR) * u_max + gap * (len(_SWOOSH_STRIPES_BGR) - 1)
    half = total_stripe_w / 2.0
    buffer = 0.06 * u_max

    if kind == "intro":
        # Wave moves from large u (right) to small u (left). Covered side (u <
        # wave - half) shows dark; uncovered side shows ref_frame (the reveal).
        start_u = u_max + half + buffer
        end_u = -half - buffer
    else:
        # Outro: wave moves left → right, covering the frame with dark behind it.
        start_u = -half - buffer
        end_u = u_max + half + buffer
    u_wave = start_u + (end_u - start_u) * p

    result = ref_frame.copy()
    covered_mask = u_grid < (u_wave - half)
    result[covered_mask] = _SWOOSH_DARK_BGR

    edge_soft = max(1.5, u_max * 0.003)
    result_f = result.astype(np.float32)
    cur_right = u_wave + half
    for color, frac in _SWOOSH_STRIPES_BGR:
        sw = frac * u_max
        cur_left = cur_right - sw
        d_right = cur_right - u_grid
        d_left = u_grid - cur_left
        alpha = np.minimum(
            np.clip(d_right / edge_soft, 0.0, 1.0),
            np.clip(d_left / edge_soft, 0.0, 1.0),
        )
        if alpha.max() > 0:
            alpha = alpha[..., None]
            color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
            result_f = result_f * (1.0 - alpha) + color_arr * alpha
        cur_right = cur_left - gap

    return np.clip(result_f, 0, 255).astype(np.uint8)


def compute_target(src_w: int, src_h: int, aspect: str) -> tuple[int, int, tuple[int, int]]:
    """Choose an output frame size for the requested aspect ratio.

    We pick a sensible canvas size — tall enough for the social-media platform
    but not larger than needed. The actual source region used per frame is
    determined separately by build_trajectory() and resized to fit here.
    """
    aw, ah = (int(x) for x in aspect.split(":"))
    target_h = min(src_h, 1080)
    target_w = int(round(target_h * aw / ah))
    target_w -= target_w % 2
    target_h -= target_h % 2
    return target_w, target_h, (aw, ah)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="input video file")
    ap.add_argument("-o", "--output", help="output mp4 (default: <input>_social.mp4)")
    ap.add_argument("--aspect", default="9:16", help="target aspect ratio W:H (default 9:16)")
    ap.add_argument("--no-subs", action="store_true", help="skip subtitle generation")
    ap.add_argument("--whisper-model", default="base", help="whisper model size (tiny/base/small/medium/large)")
    ap.add_argument("--sample-every", type=int, default=5, help="run face detection on every Nth frame (default 5)")
    ap.add_argument("--smooth-seconds", type=float, default=1.5, help="single-speaker trajectory smoothing window (default 1.5s)")
    ap.add_argument("--min-dwell", type=float, default=1.5, help="minimum seconds before cutting to a new speaker (default 1.5s)")
    ap.add_argument(
        "--layout", choices=("auto", "single", "stack"), default="auto",
        help="auto: stack two speakers if 2+ detected; single: one panel, cuts between speakers; stack: top/bottom split (default auto)",
    )
    ap.add_argument(
        "--vocab", default=None,
        help="proper nouns / jargon to bias Whisper toward, e.g. 'Brian Myri, Matthew Fenton, Aprize'",
    )
    ap.add_argument(
        "--replace", action="append", default=[],
        help="post-transcription replacement 'old=new' (repeatable). Word boundaries + case-insensitive.",
    )
    ap.add_argument(
        "--replacements-file", default=None,
        help="path to a file of 'old=new' lines. <input>.replacements.txt is auto-loaded if it exists.",
    )
    ap.add_argument(
        "--review-captions", action=argparse.BooleanOptionalAction, default=True,
        help="open the transcript in $EDITOR (or $VISUAL) to confirm/edit before rendering (default: on)",
    )
    ap.add_argument(
        "--trim-silence", action=argparse.BooleanOptionalAction, default=True,
        help="trim leading/trailing silence so the cut starts/ends near the first/last spoken word (default: on)",
    )
    ap.add_argument(
        "--lead-pad", type=float, default=0.15,
        help="seconds of audio to keep before the first spoken word (default 0.15)",
    )
    ap.add_argument(
        "--tail-pad", type=float, default=0.30,
        help="seconds of audio to keep after the last spoken word (default 0.30)",
    )
    ap.add_argument(
        "--emphasis", action=argparse.BooleanOptionalAction, default=True,
        help="use an LLM to pick emphasis moments and add quick zoom-in punches (default: on if API key is configured)",
    )
    ap.add_argument(
        "--emphasis-model", default="anthropic/claude-sonnet-4.5",
        help="OpenRouter model id for emphasis picks (default anthropic/claude-sonnet-4.5)",
    )
    ap.add_argument(
        "--emphasis-max-per-min", type=float, default=2.0,
        help="soft cap on zoom punches per minute (default 2.0)",
    )
    ap.add_argument(
        "--emphasis-min-spacing", type=float, default=8.0,
        help="minimum seconds between zoom punches (default 8.0)",
    )
    ap.add_argument(
        "--emphasis-zoom", type=float, default=1.15,
        help="peak zoom factor for emphasis punches (default 1.15 = +15%%)",
    )
    ap.add_argument(
        "--intro", action=argparse.BooleanOptionalAction, default=True,
        help="prepend a colored swoosh intro that reveals the first frame (default: on)",
    )
    ap.add_argument(
        "--outro", action=argparse.BooleanOptionalAction, default=True,
        help="append a colored swoosh outro that wipes to black (default: on)",
    )
    ap.add_argument(
        "--intro-duration", type=float, default=1.1,
        help="intro swoosh sweep length in seconds (default 1.1)",
    )
    ap.add_argument(
        "--outro-duration", type=float, default=0.85,
        help="outro swoosh sweep length in seconds (default 0.85)",
    )
    args = ap.parse_args()

    check_deps()
    inp = Path(args.input).expanduser().resolve()
    if not inp.exists():
        sys.exit(f"no such file: {inp}")
    out = Path(args.output).expanduser().resolve() if args.output else inp.with_name(inp.stem + "_social.mp4")

    info = probe(inp)
    src_w, src_h, fps, duration = info["width"], info["height"], info["fps"], info["duration"]
    target_w, target_h, aspect_wh = compute_target(src_w, src_h, args.aspect)
    print(f"source {src_w}x{src_h} @ {fps:.2f}fps  →  target {target_w}x{target_h} ({args.aspect})")

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)

        # Transcribe first (when subs are on) so we can use word timestamps as
        # the speech boundaries for trimming. Whisper sees the *original* audio
        # — trimming would only confuse it.
        segments: list[Segment] = []
        if not args.no_subs:
            print(f"transcribing with whisper '{args.whisper_model}'…")
            segments = transcribe(inp, args.whisper_model, initial_prompt=args.vocab)
            replacements = load_replacements(args, inp)
            if replacements:
                segments = apply_replacements(segments, replacements)
                print(f"  {len(segments)} subtitle segments ({len(replacements)} replacement(s) applied)")
            else:
                print(f"  {len(segments)} subtitle segments")
            if args.review_captions and sys.stdin.isatty():
                segments = review_captions(segments)
                print(f"  {len(segments)} subtitle segments after review")
            elif args.review_captions:
                print("  stdin is not a TTY — skipping caption review")

        # Decide on a trim window. Prefer ffmpeg's silencedetect — whisper tends
        # to stretch the first word's start back to the segment start (sometimes
        # all the way to 0.0), so its word timings are unreliable for trimming.
        # Fall back to whisper bounds only if silencedetect finds nothing.
        work_inp = inp
        if args.trim_silence and duration > 0:
            bounds = speech_bounds_silencedetect(inp, duration)
            if bounds is None and segments:
                bounds = speech_bounds_from_segments(segments)
            if bounds is not None:
                first, last = bounds
                trim_start = max(0.0, first - args.lead_pad)
                trim_end = min(duration, last + args.tail_pad)
                new_dur = trim_end - trim_start
                lead_gap = trim_start
                tail_gap = duration - trim_end
                print(
                    f"  speech bounds: {first:.2f}s – {last:.2f}s "
                    f"(lead silence {lead_gap:.2f}s, tail silence {tail_gap:.2f}s)"
                )
                # Only bother trimming if we'd cut more than ~200ms from either end.
                if new_dur > 0.5 and (lead_gap > 0.2 or tail_gap > 0.2):
                    print(
                        f"trimming silence: {trim_start:.2f}s – {trim_end:.2f}s "
                        f"(was 0.00s – {duration:.2f}s)"
                    )
                    trimmed = tdp / "trimmed.mp4"
                    trim_source(inp, trimmed, trim_start, new_dur)
                    work_inp = trimmed
                    segments = shift_segments(segments, trim_start, new_dur)
                else:
                    print("no significant leading/trailing silence to trim")
            else:
                print("could not detect speech boundaries; skipping silence trim")

        print(f"detecting faces (every {args.sample_every} frames)…")
        faces, borders, total = detect_faces(work_inp, args.sample_every)
        sampled = (total + args.sample_every - 1) // args.sample_every
        print(f"  {len(faces)} face detections, {len(borders)} active-speaker borders across {sampled} sampled frames")

        panels_per_frame, panel_tracks_per_frame, speaker_per_frame, tracks, layout_desc = build_layout(
            faces, borders, total, src_w, src_h, fps,
            args.sample_every, args.smooth_seconds, args.min_dwell,
            target_w, target_h, aspect_wh, args.layout,
        )
        print(f"  {len(tracks)} face tracks → layout: {layout_desc}")

        zoom_per_frame: list[list[float]] | None = None
        if args.emphasis and segments:
            api_key = load_openrouter_key()
            if not api_key:
                print("  emphasis: no OpenRouter API key found ($OPENROUTER_API_KEY or .openrouter) — skipping")
            else:
                # `total` is the trimmed-file frame count and `segments` is in
                # the trimmed timeline, so picks land on the right frames.
                trimmed_duration = total / fps if fps > 0 else 0.0
                print(f"  emphasis: asking {args.emphasis_model} for zoom moments…")
                try:
                    moments = detect_emphasis_llm(
                        segments,
                        api_key,
                        args.emphasis_model,
                        args.emphasis_max_per_min,
                        args.emphasis_min_spacing,
                        trimmed_duration,
                    )
                except RuntimeError as e:
                    print(f"  emphasis: {e} — skipping")
                    moments = []
                for (s, e, why) in moments:
                    print(f"    {s:6.2f}s – {e:6.2f}s  {why}")
                if moments:
                    zoom_per_frame = build_zoom_per_panel(
                        moments, panel_tracks_per_frame, speaker_per_frame,
                        total, fps, max_zoom=args.emphasis_zoom,
                    )

        # The swoosh is baked on top of the playing video for the first
        # `intro_frames` and last `outro_frames` of the composited stream, so
        # the underlying scene keeps moving while the animation passes over it.
        # We cap each bookend at ~25% of the total to leave most of the clip
        # un-overlaid even on short videos.
        intro_frames = 0
        outro_frames = 0
        cap_frames = max(1, total // 4)
        if args.intro:
            intro_frames = min(cap_frames, int(round(args.intro_duration * fps)))
        if args.outro:
            outro_frames = min(cap_frames, int(round(args.outro_duration * fps)))
        if intro_frames > 0:
            print(f"  intro swoosh: {intro_frames} frames ({intro_frames / fps:.2f}s)")
        if outro_frames > 0:
            print(f"  outro swoosh: {outro_frames} frames ({outro_frames / fps:.2f}s)")

        print("compositing + drawing subtitles…")
        cropped = tdp / "cropped.mp4"
        composite_with_subs(
            work_inp, cropped, panels_per_frame,
            target_w, target_h, src_w, src_h, fps, segments,
            zoom_per_frame=zoom_per_frame,
            intro_frames=intro_frames, outro_frames=outro_frames,
        )

        print("muxing audio…")
        subprocess.check_call([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats",
            "-i", str(cropped),
            "-i", str(work_inp),
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            "-shortest",
            str(out),
        ])

    print(f"done → {out}")


if __name__ == "__main__":
    main()
