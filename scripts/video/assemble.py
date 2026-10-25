"""ffmpeg helpers: clips from slides and recordings, captions (ASS burn-in + SRT), voice-over mix."""
from __future__ import annotations

import hashlib
import re
import subprocess
import textwrap
from pathlib import Path

from PIL import ImageFont

from slides import BG, FONT_DIR

FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
BG_HEX = "0x" + BG.lstrip("#")
CAP_SIZE = 42
MAX_CAP_CHARS = 62


def run(cmd, **kw):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if r.returncode:
        raise RuntimeError(f"{' '.join(map(str, cmd))[:300]}\n{r.stderr[-2000:]}")
    return r


def probe_duration(path) -> float:
    r = run([FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path])
    return float(r.stdout.strip())


ENC = ["-c:v", "libx264", "-crf", "13", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-r", "30", "-an"]


def _fade(dur: float) -> str:
    return (f"fade=t=in:st=0:d=0.3:color={BG_HEX},"
            f"fade=t=out:st={max(dur - 0.3, 0):.3f}:d=0.3:color={BG_HEX}")


def still_clip(png: Path, dur: float, out: Path):
    run([FFMPEG, "-y", "-loglevel", "error", "-loop", "1", "-framerate", "30", "-i", png, "-t", f"{dur:.3f}",
         "-vf", "format=yuv420p", *ENC, out])


def slide_segment(pngs: list[Path], starts: list[float], dur: float, out: Path):
    """Concatenate one still per view, then fade the whole segment in and out."""
    tmp = []
    bounds = list(starts[1:]) + [dur]
    for i, (p, a, b) in enumerate(zip(pngs, starts, bounds)):
        c = out.with_name(f"{out.stem}_c{i}.mp4")
        still_clip(p, b - a, c)
        tmp.append(c)
    raw = out.with_name(out.stem + "_raw.mp4")
    concat(tmp, raw)
    run([FFMPEG, "-y", "-loglevel", "error", "-i", raw, "-vf", _fade(dur), *ENC, out])


def live_segment(webm: Path, t_start: float, rec_dur: float, dur: float, out: Path):
    """Trim a Playwright recording to [t_start, t_start + rec_dur], 1080p30; if captions need more time,
    hold the last frame until `dur`. Short fades in and out."""
    pad = f",tpad=stop_mode=clone:stop_duration={dur - rec_dur + 0.05:.3f}" if dur > rec_dur else ""
    vf = f"fps=30,scale=1920:1080:flags=lanczos,setsar=1{pad},{_fade(dur)}"
    run([FFMPEG, "-y", "-loglevel", "error", "-i", webm, "-ss", f"{t_start:.3f}", "-t", f"{rec_dur:.3f}", "-vf", vf,
         "-t", f"{dur:.3f}", *ENC, out])


def concat(clips: list[Path], out: Path):
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{Path(c).resolve()}'\n" for c in clips))
    run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out])


# ------------------------------------------------------------------------------------- captions
def wrap_caption(text: str) -> list[str]:
    lines = textwrap.wrap(text, MAX_CAP_CHARS, break_long_words=False)
    if len(lines) > 2:
        raise ValueError(f"caption needs {len(lines)} lines (max 2): {text!r}")
    if len(lines) == 2:   # balance the two lines
        w = textwrap.wrap(text, max(len(text) // 2 + 6, 20), break_long_words=False)
        if len(w) == 2 and max(map(len, w)) <= MAX_CAP_CHARS:
            lines = w
    return lines


def _ts_ass(t):
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _ts_srt(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_srt(caps, path: Path):
    out = []
    for i, c in enumerate(caps, 1):
        out.append(f"{i}\n{_ts_srt(c['start'])} --> {_ts_srt(c['end'])}\n" + "\n".join(wrap_caption(c["text"])) + "\n")
    path.write_text("\n".join(out), encoding="utf-8")


def write_ass(caps, path: Path, size=(1920, 1080)):
    """One dark box per caption (drawn vector) plus centered white text, so two lines share a single box."""
    fnt = ImageFont.truetype(str(FONT_DIR / "DejaVuSans.ttf"), CAP_SIZE)
    asc, desc = fnt.getmetrics()
    lh = asc + desc
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {size[0]}
PlayResY: {size[1]}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,DejaVu Sans,{CAP_SIZE},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,5,40,40,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []
    for c in caps:
        lines = wrap_caption(c["text"])
        tw = max(fnt.getlength(l) for l in lines)
        bw, bh = tw + 76, lh * len(lines) + 34
        cx, cy = size[0] / 2, size[1] - 34 - bh / 2 - 6
        x0, y0 = cx - bw / 2, cy - bh / 2
        r = 14
        shape = (f"m {r} 0 l {bw - r:.0f} 0 b {bw:.0f} 0 {bw:.0f} 0 {bw:.0f} {r} l {bw:.0f} {bh - r:.0f} "
                 f"b {bw:.0f} {bh:.0f} {bw:.0f} {bh:.0f} {bw - r:.0f} {bh:.0f} l {r} {bh:.0f} "
                 f"b 0 {bh:.0f} 0 {bh:.0f} 0 {bh - r:.0f} l 0 {r} b 0 0 0 0 {r} 0")
        a, b = _ts_ass(c["start"]), _ts_ass(c["end"])
        ev.append(f"Dialogue: 0,{a},{b},Cap,,0,0,0,,{{\\an7\\pos({x0:.0f},{y0:.0f})\\bord0\\shad0\\1c&H101010&\\1a&H30&\\p1}}{shape}{{\\p0}}")
        ev.append(f"Dialogue: 1,{a},{b},Cap,,0,0,0,,{{\\an5\\pos({cx:.0f},{cy + 1:.0f})\\fad(120,120)}}" + "\\N".join(lines))
    path.write_text(head + "\n".join(ev) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------------- voice
def speakable(text: str) -> str:
    t = text
    t = t.replace("10⁻⁴", "10 to the minus 4").replace("R²", "R squared")
    t = re.sub(r"\bN-(\d|k)", r"N minus \1", t)
    t = t.replace("−", "minus ").replace("×", " times").replace("Δ", "delta").replace("…", "")
    t = re.sub(r"(\d)–(\d)", r"\1 to \2", t)
    t = re.sub(r"\s\+\s", " plus ", t)
    t = re.sub(r"(^|[\s(])\+(\d)", r"\1plus \2", t)
    t = re.sub(r"(\d) ms\b", r"\1 milliseconds", t)
    t = re.sub(r"\bMW\b", "megawatts", t)
    t = t.replace("GOC", "G O C").replace("SHA-256", "S H A 256").replace("GENCO-", "GENCO ")
    return t


def synth(text: str, voice: str, rate: int, cache: Path) -> Path:
    key = hashlib.sha1(f"{voice}|{rate}|{text}".encode()).hexdigest()[:16]
    wav = cache / f"{key}.wav"
    if not wav.exists():
        aiff = cache / f"{key}.aiff"
        run(["say", "-v", voice, "-r", rate, "-o", aiff, text])
        run([FFMPEG, "-y", "-loglevel", "error", "-i", aiff, "-ar", "44100", "-ac", "2", wav])
        aiff.unlink()
    return wav


def mix_voice(caps, total: float, voice: str, rate: int, cache: Path, out: Path):
    cache.mkdir(parents=True, exist_ok=True)
    clips = []
    for c in caps:
        w = synth(c.get("say") or speakable(c["text"]), voice, rate, cache)
        d = probe_duration(w)
        room = c["end"] - c["start"]
        if d > room + 0.3:
            print(f"  warning: voice {d:.1f}s exceeds caption window {room:.1f}s at {c['start']:.1f}s: {c['text'][:50]!r}")
        clips.append((w, c["start"] + 0.12))
    inputs = ["-f", "lavfi", "-t", f"{total:.3f}", "-i", "anullsrc=r=44100:cl=stereo"]
    for w, _ in clips:
        inputs += ["-i", str(w)]
    fl = "".join(f"[{i + 1}:a]adelay={int(s * 1000)}|{int(s * 1000)}[a{i}];" for i, (_, s) in enumerate(clips))
    fl += "[0:a]" + "".join(f"[a{i}]" for i in range(len(clips))) + f"amix=inputs={len(clips) + 1}:normalize=0:duration=first,alimiter=limit=0.95[m]"
    run([FFMPEG, "-y", "-loglevel", "error", *inputs, "-filter_complex", fl, "-map", "[m]", "-c:a", "aac", "-b:a", "160k", out])
    return out


def silence(total: float, out: Path):
    run([FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-t", f"{total:.3f}", "-i", "anullsrc=r=44100:cl=stereo",
         "-c:a", "aac", "-b:a", "96k", out])
    return out


def burn(video: Path, audio: Path, ass: Path, out: Path):
    vf = f"subtitles={ass.name}:fontsdir={FONT_DIR}"
    run([FFMPEG, "-y", "-loglevel", "error", "-i", video.resolve(), "-i", audio.resolve(), "-vf", vf,
         "-c:v", "libx264", "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "copy",
         "-movflags", "+faststart", "-shortest", out.resolve()], cwd=ass.parent)
