"""Record scripted sessions on the GENCO Operator Console (space/app.py) with Playwright.

Headless Chromium has no visible pointer, so the page gets an injected cursor arrow that follows
mouse events, a ring around the element about to be clicked or discussed, and smooth scrolling.
Every action logs its wall-clock time (as seconds into the video) so captions can follow events.
"""
from __future__ import annotations

import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

VIEW = {"width": 1920, "height": 1080}

INIT_JS = r"""
(() => {
  const mk = () => {
    if (document.getElementById('__cur')) return document.getElementById('__cur');
    const c = document.createElement('div');
    c.id = '__cur';
    c.style.cssText = 'position:fixed;left:-60px;top:-60px;width:30px;height:30px;z-index:2147483647;pointer-events:none;';
    c.innerHTML = '<svg width="30" height="30" viewBox="0 0 24 24"><path d="M3 2 L3 19 L8 14.5 L11.5 22 L14.5 20.6 L11 13.2 L18 13 Z" fill="#0b0b0b" stroke="#ffffff" stroke-width="1.6" stroke-linejoin="round"/></svg>';
    document.documentElement.appendChild(c);
    return c;
  };
  document.addEventListener('mousemove', e => { const c = mk(); c.style.left = (e.clientX - 3) + 'px'; c.style.top = (e.clientY - 2) + 'px'; }, true);
  document.addEventListener('mousedown', e => {
    const r = document.createElement('div');
    r.style.cssText = `position:fixed;left:${e.clientX-18}px;top:${e.clientY-18}px;width:36px;height:36px;border-radius:50%;border:3px solid #eb6834;z-index:2147483646;pointer-events:none;transition:all .45s ease-out;opacity:.95`;
    document.documentElement.appendChild(r);
    requestAnimationFrame(() => { r.style.transform = 'scale(1.9)'; r.style.opacity = '0'; });
    setTimeout(() => r.remove(), 600);
  }, true);
  const pad = () => { const st = document.createElement('style'); st.textContent = 'body::after{content:"";display:block;height:560px}'; document.documentElement.appendChild(st); };
  if (document.documentElement) pad(); else document.addEventListener('DOMContentLoaded', pad);
  window.__ring = (x, y, w, h, ms) => {
    const d = document.createElement('div');
    d.className = '__ring';
    d.style.cssText = `position:fixed;left:${x-6}px;top:${y-6}px;width:${w+12}px;height:${h+12}px;border:4px solid #eb6834;border-radius:12px;box-shadow:0 0 0 6px rgba(235,104,52,.18);z-index:2147483645;pointer-events:none;opacity:0;transition:opacity .25s`;
    document.documentElement.appendChild(d);
    requestAnimationFrame(() => d.style.opacity = '1');
    setTimeout(() => { d.style.opacity = '0'; setTimeout(() => d.remove(), 300); }, ms);
  };
  window.__smoothScroll = (y, ms) => new Promise(res => {
    const y0 = window.scrollY, t0 = performance.now();
    const ease = t => t < .5 ? 2*t*t : 1 - Math.pow(-2*t + 2, 2) / 2;
    const f = now => { const t = Math.min(1, (now - t0) / ms); window.scrollTo(0, y0 + (y - y0) * ease(t)); t < 1 ? requestAnimationFrame(f) : res(); };
    requestAnimationFrame(f);
  });
})();
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """Runs space/app.py on a free localhost port."""

    def __init__(self, space_dir: Path, log: Path):
        self.port = free_port()
        env = dict(os.environ, GRADIO_SERVER_PORT=str(self.port), GRADIO_SERVER_NAME="127.0.0.1",
                   GRADIO_ANALYTICS_ENABLED="False", PYTHONUNBUFFERED="1")
        self.log = open(log, "w")
        self.proc = subprocess.Popen([sys.executable, "app.py"], cwd=space_dir, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        self.url = f"http://127.0.0.1:{self.port}/"

    def wait(self, timeout=240):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("space/app.py exited early; see the server log")
            try:
                if urllib.request.urlopen(self.url, timeout=2).status == 200:
                    return
            except Exception:
                time.sleep(1)
        raise TimeoutError("console did not come up")

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


def ease(t):
    return 2 * t * t if t < .5 else 1 - (-2 * t + 2) ** 2 / 2


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


class Session:
    def __init__(self, page, t0: float):
        self.page, self.t0, self.events = page, t0, []
        self.mx, self.my = 960.0, 540.0

    def now(self):
        return time.time() - self.t0

    def log(self, name, kind="start"):
        self.events.append({"name": name, "kind": kind, "t": round(self.now(), 3)})

    # ---- pointer -------------------------------------------------------------------------------
    def move(self, x, y, dur=0.7):
        n = max(6, int(dur * 30))
        x0, y0 = self.mx, self.my
        for i in range(1, n + 1):
            e = ease(i / n)
            self.page.mouse.move(x0 + (x - x0) * e, y0 + (y - y0) * e)
            time.sleep(dur / n)
        self.mx, self.my = x, y

    def ring(self, box, ms=600):
        self.page.evaluate("([x,y,w,h,ms]) => window.__ring(x,y,w,h,ms)", [box["x"], box["y"], box["width"], box["height"], ms])

    def click(self, loc, name=None):
        loc.scroll_into_view_if_needed()
        box = loc.bounding_box()
        self.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, 0.8)
        self.ring(box, 700)
        time.sleep(0.45)
        self.page.mouse.down()
        time.sleep(0.08)
        self.page.mouse.up()

    # ---- named targets ------------------------------------------------------------------------
    def block_box(self, el_loc):
        h = el_loc.evaluate_handle("e => e.closest('.block') || e")
        return h.as_element().bounding_box()

    def target_box(self, name):
        p = self.page
        if name == "grid_map":
            return self.block_box(p.locator(".js-plotly-plot").nth(0))
        if name == "convergence":
            return self.block_box(p.locator(".js-plotly-plot").nth(1))
        if name == "decision":
            h = p.get_by_role("heading", name="Decision", exact=True).first
            return h.evaluate_handle("e => e.closest('[data-testid=markdown]')").as_element().bounding_box()
        if name == "table":
            return self.block_box(p.locator(".table-wrap:visible").first)
        if name == "batch_policy_rows":
            a = p.get_by_text("Sequential policy, GENCO prior", exact=True).first.bounding_box()
            b = p.get_by_text("Sequential policy, base-case prior", exact=True).first.bounding_box()
            tb = p.locator(".table-wrap:visible").first.bounding_box()
            y0, y1 = a["y"] - 10, b["y"] + b["height"] + 10
            return {"x": tb["x"], "y": y0, "width": tb["width"], "height": y1 - y0}
        if name == "batch_note":
            return p.get_by_text(re.compile(r"bundled N-\d scenarios")).last.bounding_box()
        raise KeyError(name)

    def plot_point(self, plot_idx, trace_idx, point_idx):
        js = """([pi, ti, k]) => {
            const g = document.querySelectorAll('.js-plotly-plot')[pi];
            const t = g.querySelectorAll('.scatterlayer .trace')[ti];
            const r = t.querySelectorAll('path.point')[k].getBoundingClientRect();
            return [r.x + r.width / 2, r.y + r.height / 2];
        }"""
        return self.page.evaluate(js, [plot_idx, trace_idx, point_idx])

    def worst_bus(self):
        js = """() => {
            const g = document.querySelectorAll('.js-plotly-plot')[0];
            const ti = g.data.findIndex(t => t.name === 'bus');
            const col = Array.from(g._fullData[ti].marker.color);
            let k = 0; col.forEach((v, i) => { if (v > col[k]) k = i; });
            const t = g.querySelectorAll('.scatterlayer .trace')[ti];
            const r = t.querySelectorAll('path.point')[k].getBoundingClientRect();
            return [r.x + r.width / 2, r.y + r.height / 2, k];
        }"""
        return self.page.evaluate(js)

    # ---- actions ------------------------------------------------------------------------------
    def run(self, act: dict):
        p, do = self.page, act["do"]
        name = act.get("name") or do
        self.log(name)
        if do == "wait":
            time.sleep(act["s"])
        elif do == "mark":
            pass
        elif do == "radio":
            sel = p.locator(f'label:visible:has-text("{act["label"]}")')
            self.click(sel.last if act.get("last") else sel.first)
            time.sleep(1.2)
        elif do == "tab":
            self.click(p.get_by_role("tab", name=act["label"]))
            time.sleep(0.6)
        elif do == "button":
            self.click(p.get_by_role("button", name=act["label"]))
            time.sleep(0.5)
        elif do == "scenario":
            inp = p.get_by_label("Scenario")
            self.click(inp)
            time.sleep(0.7)
            opt = p.locator(f'ul[role=listbox] li:has-text("{act["match"]}")')
            opt.scroll_into_view_if_needed()
            time.sleep(0.5)
            label = opt.inner_text()
            self.click(opt)
            m = re.search(r"out: (.*?) · GENCO", label)
            if m:   # wait until the decision card shows this outage
                p.wait_for_function("s => document.body.innerText.includes('Outage: ' + s)", arg=m.group(1), timeout=60000)
            time.sleep(1.0)
        elif do == "scroll":
            p.evaluate("([y, ms]) => window.__smoothScroll(y, ms)", [act["y"], int(act["dur"] * 1000)])
            time.sleep(0.3)
        elif do == "highlight":
            box = self.target_box(act["target"])
            self.ring(box, int(act["s"] * 1000))
            time.sleep(act["s"])
        elif do == "park":
            self.move(act["x"], act["y"], act.get("dur", 0.8))
            time.sleep(act.get("s", 0.2))
        elif do == "hover_bus":
            x, y, _ = self.worst_bus()
            self.move(x, y, 0.9)
            time.sleep(act["s"])
        elif do == "hover_point":
            x, y = self.plot_point(1, act["trace"], act["index"])
            self.move(x, y, 0.9)
            time.sleep(act["s"])
        else:
            raise ValueError(do)
        self.log(name, "end")


def warm_up(browser, url, order_inspect, order_batch):
    """Compute the console's engine caches once, off camera (first views take several seconds)."""
    pg = browser.new_page(viewport=VIEW)
    pg.goto(url)
    pg.wait_for_selector(".js-plotly-plot", timeout=240000)
    for o in sorted({order_inspect, order_batch}):
        pg.locator(f'label:visible:has-text("{o}")').first.click()
        pg.wait_for_function("() => document.querySelectorAll('.js-plotly-plot').length >= 2", timeout=240000)
        time.sleep(4)
    pg.get_by_role("tab", name="Screen a whole set").click()
    pg.locator(f'label:visible:has-text("{order_batch}")').last.click()
    pg.get_by_role("button", name="Run the screening policy on all bundled scenarios").click()
    pg.get_by_text("Sequential policy, GENCO prior", exact=True).first.wait_for(timeout=300000)
    pg.close()


def record_segment(browser, url, seg, out_dir: Path):
    ctx = browser.new_context(viewport=VIEW, device_scale_factor=1, record_video_dir=str(out_dir),
                              record_video_size=VIEW)
    ctx.add_init_script(INIT_JS)
    page = ctx.new_page()
    t0 = time.time()
    s = Session(page, t0)
    page.goto(url)
    page.wait_for_selector(".js-plotly-plot", timeout=240000)
    page.wait_for_function("() => document.querySelectorAll('.js-plotly-plot').length >= 2", timeout=240000)
    page.mouse.move(960, 540)          # park the cursor mid-screen
    for a in seg["actions"]:
        s.run(a)
    text = norm(page.evaluate("() => document.body.innerText"))
    missing = [e for e in seg.get("expect", []) if norm(e) not in text]
    page.close()
    video = Path(page.video.path())
    ctx.close()
    if missing:
        raise AssertionError(f"segment {seg['id']}: expected text not on screen: {missing}")
    marks = {e["name"]: e["t"] for e in s.events if e["kind"] == "start"}
    t_start = marks["start"]
    t_end = max(e["t"] for e in s.events) + 0.8
    log = {"video": str(video), "events": s.events, "t_start": t_start, "t_end": t_end, "duration": t_end - t_start}
    (out_dir / f"{seg['id']}.json").write_text(json.dumps(log, indent=1))
    final = out_dir / f"{seg['id']}.webm"
    video.replace(final)
    log["video"] = str(final)
    (out_dir / f"{seg['id']}.json").write_text(json.dumps(log, indent=1))
    return log


def record_all(root: Path, segments: list[dict], meta: dict, work: Path) -> dict:
    """Start the console, record every live segment, stop the console. Returns {segment id: log}."""
    srv = Server(root / "space", work / "server.log")
    logs = {}
    try:
        srv.wait()
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            warm_up(browser, srv.url, meta["console_order"], meta["batch_order"])
            for seg in segments:
                logs[seg["id"]] = record_segment(browser, srv.url, seg, work)
            browser.close()
    finally:
        srv.stop()
    for stray in work.glob("page@*.webm"):
        stray.unlink()
    return logs
