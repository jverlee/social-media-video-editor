# social-media-video-editor

A script that takes a video and edits it for social media:

- Auto-generates word-level subtitles (Whisper) with karaoke-style highlight on the current word
- Detects the active Zoom speaker tile (green border) and either crops to that speaker or stacks the top two speakers vertically
- Outputs an mp4 at the requested aspect ratio (default 9:16)

## Setup

```sh
brew install ffmpeg
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```sh
./edit_for_social.py input.mov                            # → input_social.mp4 (9:16)
./edit_for_social.py input.mov -o reel.mp4
./edit_for_social.py input.mov --aspect 1:1               # square
./edit_for_social.py input.mov --no-subs                  # skip transcription
./edit_for_social.py input.mov --whisper-model small      # better subs, slower
./edit_for_social.py input.mov --layout single            # one panel, cuts between speakers
./edit_for_social.py input.mov --layout stack             # force two-speaker stack
./edit_for_social.py input.mov --vocab "Brian Myri, Aprize"
./edit_for_social.py input.mov --replace "Myr=Myri" --replace "Fenton's=Fenton"
```

Flags:

- `--aspect W:H` — target aspect ratio (default `9:16`)
- `--layout` — `auto` (default — stack if 2+ speakers detected), `single`, or `stack`
- `--whisper-model` — `tiny`/`base`/`small`/`medium`/`large` (default `base`)
- `--vocab "name1, name2, ..."` — biases Whisper toward proper nouns
- `--replace "old=new"` — post-transcription whole-word, case-insensitive replacement (repeatable)
- `--replacements-file PATH` — file of `old=new` lines; `<input>.replacements.txt` is auto-loaded if present
- `--no-subs` — skip subtitle generation
- `--sample-every N` — face-detect every Nth frame (default 5)
- `--smooth-seconds` — single-speaker smoothing window (default 1.5s)
- `--min-dwell` — minimum seconds before cutting to a new speaker (default 1.5s)
- `--no-intro` / `--no-outro` — disable the colored swoosh bookends
- `--intro-duration` / `--outro-duration` — sweep length in seconds (default 1.1 / 0.85)

## Fixing caption errors

For one or two names Whisper consistently misses, the cheapest fix is to add them to `--vocab`. Whisper biases its decoder toward listed terms — usually catches the right spelling after one shot.

For everything else (apostrophes, brand names that resist biasing), use `--replace`. To avoid retyping every run, drop a `<video>.replacements.txt` next to your input:

```
# aprize-video.mp4.replacements.txt
Myr = Myri
Fenton's = Fenton
nonprofit = non-profit
```

Lines starting with `#` are comments. Matches are case-insensitive and respect word boundaries (so `Myr` matches `Myr.` and `Myr,` but not `Myrold`). Replacements preserve word-level timings — the karaoke highlight still lines up.

## How it works

1. **Detect faces** every Nth frame (OpenCV Haar cascade) and the Zoom **active-speaker green border**, then cluster faces into per-person tracks and assign each border to the track it encloses.
2. **Transcribe** with Whisper using `word_timestamps=True` and the optional `--vocab` bias, then apply replacements.
3. **Build the layout**:
   - `auto`/`stack` with 2+ speakers → top/bottom split, top = first speaker to receive a border, bottom = second. Other tiles are dropped.
   - Otherwise single panel that either smooth-follows the lone face or cuts between speakers using the green-border signal (with `--min-dwell` hysteresis).
4. **Composite** each frame: cut each Panel's source rect, resize to its destination on a fixed canvas, draw subtitles with PIL (white = spoken, bold yellow = current word, light gray = upcoming).
5. **Mux** original audio with ffmpeg (`libx264`, `yuv420p`, `+faststart`).
