"""Headless-browser check of the inline SVG figures (Playwright + Chromium).

Screenshots every figure in light/dark, desktop/mobile, its final (reduced-motion) state and
a few mid-animation frames; checks accessibility hooks, console errors and mobile overflow.

Usage: python build/check_figures.py [screenshot_dir]
Env:   WB_DIST=/path/to/dist   (default: dist/)
       CHROMIUM=/path/to/chrome (optional; else Playwright's bundled browser)
       FIG_FILE=1               also open the pages over file://
"""

from __future__ import annotations

import functools
import http.server
import os
import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
DIST = Path(os.environ["WB_DIST"]).resolve() if os.environ.get("WB_DIST") else ROOT / "dist"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / ".screenshots" / "figures")
OUT.mkdir(parents=True, exist_ok=True)

PAGES = ["01-tensors.html", "04-loss-functions.html"]
FRAMES = {  # seconds into the animation to capture
    "broadcast-align": [1.0, 3.2, 8.6],
    "broadcast-stretch": [1.2, 3.2, 5.0],
    "broadcast-keepdims": [1.2, 4.7, 7.2],
    "broadcast-pairwise": [1.9, 4.9, 5.6],
}
VIEWPORTS = {"desktop": (1440, 1000), "mobile": (390, 844)}

results: list[tuple[bool, str]] = []


def ok(cond, msg):
    results.append((bool(cond), msg))
    print(("  ✓ " if cond else "  ✗ ") + msg)


def serve() -> str:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    handler = functools.partial(Quiet, directory=str(DIST))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{httpd.server_address[1]}/"


REVEAL = """() => {
  document.querySelectorAll('details.prereq').forEach(d => d.open = true);
  document.querySelectorAll('figure.fig').forEach(f => {
    const panel = f.closest('[role=tabpanel]');
    if (panel && panel.hidden) {
      const skip = panel.closest('.exercise').querySelector('[data-predict-skip]');
      if (skip) skip.click();
      const tab = document.getElementById(panel.getAttribute('aria-labelledby'));
      if (tab) tab.click();
    }
  });
}"""

SEEK = """([name, t]) => {
  const fig = document.querySelector(`figure[data-fig="${name}"]`);
  if (fig.__fx) fig.__fx.played = true;  // keep scroll-into-view from restarting it
  fig.classList.add('fx-on', 'is-playing');
  fig.getBoundingClientRect();
  for (const a of fig.getAnimations({subtree: true})) { a.pause(); a.currentTime = t * 1000; }
}"""

MIN_FONT = """(name) => {
  const svg = document.querySelector(`figure[data-fig="${name}"] svg`);
  const scale = svg.getBoundingClientRect().width / svg.viewBox.baseVal.width;
  let min = 1e9;
  svg.querySelectorAll('text').forEach(t => { min = Math.min(min, parseFloat(t.getAttribute('font-size')) * scale); });
  return [min, svg.getBoundingClientRect().width];
}"""


def shoot(page, name, path):
    page.locator(f'figure[data-fig="{name}"]').first.screenshot(path=str(path))


def run(browser, base: str, label: str, shots: bool):
    for vp, (w, h) in VIEWPORTS.items():
        for motion in ("reduce", "no-preference"):
            ctx = browser.new_context(viewport={"width": w, "height": h}, device_scale_factor=2 if shots else 1,
                                      reduced_motion=motion)
            page = ctx.new_page()
            errors: list[str] = []
            page.on("console", lambda m: m.type == "error" and errors.append(m.text))
            page.on("pageerror", lambda e: errors.append(str(e)))
            for pg in PAGES:
                for theme in ("light", "dark"):
                    page.goto(base + pg)
                    page.wait_for_selector("html[data-ready='1']")
                    page.evaluate(f"document.documentElement.dataset.theme = '{theme}'")
                    page.evaluate("document.fonts.ready")
                    page.evaluate(REVEAL)
                    names = page.eval_on_selector_all("figure.fig", "fs => fs.map(f => f.dataset.fig)")
                    tag = f"{label}-{vp}-{motion}-{theme}-{pg[:2]}"
                    if theme == "light":
                        ok(len(names) > 0, f"{tag}: figures present {names}")
                        for n in names:
                            a11y = page.evaluate("""(n) => { const s = document.querySelector(`figure[data-fig="${n}"] svg`);
                                return s.getAttribute('role') === 'img' && !!s.querySelector('title') && !!s.querySelector('desc'); }""", n)
                            ok(a11y, f"{tag}: {n} has role=img + title + desc")
                            mf, wd = page.evaluate(MIN_FONT, n)
                            ok(mf >= (10.5 if vp == "mobile" else 12), f"{tag}: {n} min text {mf:.1f}px at width {wd:.0f}px")
                            btn = page.locator(f'figure[data-fig="{n}"] .fig-replay')
                            if motion == "reduce":
                                ok(btn.count() == 0 or btn.is_hidden(), f"{tag}: {n} replay hidden under reduced motion")
                                ok(not page.evaluate(f"document.querySelector('figure[data-fig=\"{n}\"]').classList.contains('fx-on')"),
                                   f"{tag}: {n} not animated under reduced motion")
                        if vp == "mobile":
                            sw = page.evaluate("document.documentElement.scrollWidth")
                            ok(sw <= w, f"{tag}: no horizontal overflow (scrollWidth {sw})")
                    if not shots:
                        continue
                    for n in names:
                        if motion == "reduce":
                            shoot(page, n, OUT / f"{n}-{vp}-{theme}-final.png")
                        elif vp == "desktop" or theme == "light":
                            for t in FRAMES.get(n, []):
                                page.evaluate(SEEK, [n, t])
                                shoot(page, n, OUT / f"{n}-{vp}-{theme}-t{t:.1f}.png")
                    if motion == "no-preference" and theme == "light" and pg == PAGES[0]:
                        # natural play: scroll a figure into view, check it starts and Replay works
                        n = "broadcast-stretch"
                        fig = page.locator(f'figure[data-fig="{n}"]')
                        fig.scroll_into_view_if_needed()
                        page.wait_for_timeout(600)
                        ok("is-playing" in (fig.get_attribute("class") or ""), f"{tag}: {n} starts playing when scrolled into view")
                        fig.locator(".fig-replay").click()
                        page.wait_for_timeout(300)
                        running = page.evaluate(f"""() => document.querySelector('figure[data-fig="{n}"]').getAnimations({{subtree:true}})
                            .filter(a => a.playState === 'running').length""")
                        ok(running > 0, f"{tag}: Replay restarts animations ({running} running)")
            ok(not errors, f"{label}-{vp}-{motion}: no console errors {errors[:3]}")
            ctx.close()


def main() -> int:
    launch = {"executable_path": os.environ["CHROMIUM"]} if os.environ.get("CHROMIUM") else {}
    with sync_playwright() as p:
        browser = p.chromium.launch(**launch)
        run(browser, serve(), "http", shots=True)
        if os.environ.get("FIG_FILE"):
            run(browser, DIST.as_uri() + "/", "file", shots=False)
        browser.close()
    bad = [m for c, m in results if not c]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
