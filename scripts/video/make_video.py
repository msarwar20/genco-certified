#!/usr/bin/env python
"""Build the GENCO challenge demo video from docs/video/storyboard.yaml.

    source .venv/bin/activate
    caffeinate -i python scripts/video/make_video.py [--voice] [--end-links "..."] [--reuse-live]

Outputs (dist/video/): genco_entry_captions.mp4 (+ .srt), and with --voice genco_entry_voice.mp4.
Intermediate files stay in dist/video/work/. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import assemble as A  # noqa: E402
import slides as S    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", action="store_true", help="also write genco_entry_voice.mp4 (macOS `say`)")
    ap.add_argument("--end-links", default="{{LINKS}}", help="text shown on the end card under 'Code, data, ...'")
    ap.add_argument("--storyboard", default=str(ROOT / "docs/video/storyboard.yaml"))
    ap.add_argument("--out", default=str(ROOT / "dist/video"))
    ap.add_argument("--reuse-live", action="store_true", help="reuse recordings in work/ instead of re-recording")
    args = ap.parse_args()

    sb = yaml.safe_load(Path(args.storyboard).read_text())
    meta, segs = sb["meta"], sb["segments"]
    out = Path(args.out)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)

    # 1. live recordings -------------------------------------------------------------------------
    live = [s for s in segs if s["type"] == "live"]
    if args.reuse_live and all((work / f"{s['id']}.json").exists() for s in live):
        logs = {s["id"]: json.loads((work / f"{s['id']}.json").read_text()) for s in live}
        print("reusing live recordings")
    else:
        import live as L
        print("recording the console ...")
        logs = L.record_all(ROOT, live, meta, work)

    # 2. clips + timeline ------------------------------------------------------------------------
    # Caption windows follow the spoken length of each caption (macOS `say`), so the captions-only and
    # the voiced video share one timeline. `dur` in the storyboard is a minimum reading time.
    tts = work / "tts"
    tts.mkdir(exist_ok=True)

    def spoken(c):
        return A.probe_duration(A.synth(c.get("say") or A.speakable(c["text"]), meta["voice"], meta["voice_rate"], tts))

    clips, caps, t = [], [], 0.0
    for seg in segs:
        sid = seg["id"]
        clip = work / f"seg_{sid}.mp4"
        is_slide = seg["type"] == "slide"
        if is_slide:
            ev, base = {}, 0.0
        else:
            lg = logs[sid]
            ev = {e["name"]: e["t"] for e in lg["events"] if e["kind"] == "start"}
            base = lg["t_start"]
        rel_caps, prev_end = [], None
        for c in seg.get("captions", []):
            A.wrap_caption(c["text"])    # validates the 2-line limit
            start = c["at"] if "at" in c else ev[c["after"]] - base + c.get("offset", 0.3)
            if prev_end is not None:
                start = max(start, prev_end + 0.3)
            end = start + max(c["dur"], spoken(c) + 0.45)
            rel_caps.append((start, end, c))
            prev_end = end
        need = (prev_end or 0.0) + 0.9
        if is_slide:
            card = seg["card"]
            dur = max(float(seg["duration"]), need)
            pngs = S.render(card, ROOT, work / f"card_{sid}", args.end_links)
            starts = [0.0]
            if card["kind"] == "figure":
                starts = [0.0 if i == 0 else (rel_caps[v["caption"]][0] - 0.3 if "caption" in v else v["at"])
                          for i, v in enumerate(card["views"])]
            A.slide_segment(pngs, starts, dur, clip)
        else:
            dur = max(lg["duration"], need)
            A.live_segment(Path(lg["video"]), lg["t_start"], lg["duration"], dur, clip)
        clips.append(clip)
        for start, end, c in rel_caps:
            caps.append({"start": t + start, "end": t + end, "text": c["text"], "say": c.get("say")})
        print(f"  {sid:10s} {t:7.1f} s  +{dur:5.1f} s")
        t += dur
    total = t

    # 3. assemble ----------------------------------------------------------------------------------
    A.concat(clips, work / "all.mp4")
    A.write_srt(caps, out / "genco_entry_captions.srt")
    A.write_ass(caps, work / "captions.ass")
    A.burn(work / "all.mp4", A.silence(total, work / "silence.m4a"), work / "captions.ass", out / "genco_entry_captions.mp4")
    print(f"wrote {out / 'genco_entry_captions.mp4'}  ({A.probe_duration(out / 'genco_entry_captions.mp4'):.1f} s)")
    if args.voice:
        voice = A.mix_voice(caps, total, meta["voice"], meta["voice_rate"], tts, work / "voice.m4a")
        A.burn(work / "all.mp4", voice, work / "captions.ass", out / "genco_entry_voice.mp4")
        print(f"wrote {out / 'genco_entry_voice.mp4'}")
    write_readme(out)


def write_readme(out: Path):
    rows = []
    for n in ["genco_entry_captions.mp4", "genco_entry_captions.srt", "genco_entry_voice.mp4"]:
        p = out / n
        if p.exists():
            d = f", {A.probe_duration(p):.0f} s" if p.suffix == ".mp4" else ""
            rows.append(f"- `{n}`: {p.stat().st_size / 1e6:.1f} MB{d}")
    nl = "\n"
    (out / "README.md").write_text(f"""# Demo video

Regenerate everything (records the live console, renders slides, burns captions):

```
source .venv/bin/activate
caffeinate -i python scripts/video/make_video.py [--voice] [--end-links "https://..."] [--reuse-live]
```

- Storyboard (segments, captions, live actions): `docs/video/storyboard.yaml`. Numbers in captions are quoted from `docs/POST.md`.
- `--voice` synthesizes each caption with macOS `say` (voice and rate in the storyboard `meta`) and also writes the voiced MP4.
- `--end-links` replaces the placeholder `{{{{LINKS}}}}` on the end card.
- `--reuse-live` skips re-recording the console and reuses `work/*.webm` and the event logs (`work/*.json`).
- Needs: ffmpeg with libass, Playwright Chromium, PyYAML, Pillow, and the console's own dependencies. The console is started on a free localhost port and stopped afterwards.

Outputs in this folder:
{nl.join(rows)}

`work/` holds intermediates (card PNGs, recordings with timestamped action logs, ASS captions, TTS cache). Nothing is uploaded or published by the script.
""")


if __name__ == "__main__":
    main()
