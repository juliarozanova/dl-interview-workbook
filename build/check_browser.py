"""Headless-browser smoke test of the built site (Playwright + Chromium).

Usage: python build/check_browser.py [screenshot_dir]
Env:   CHROMIUM=/path/to/chrome   (optional; else Playwright's bundled browser)
"""

from __future__ import annotations

import functools
import http.server
import re
import os
import sys
import threading
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / ".screenshots")
OUT.mkdir(parents=True, exist_ok=True)
CHAPTER = "04-loss-functions.html"
CH13 = "13-attention.html"
PY_TIMEOUT = 150_000  # first Pyodide boot downloads ~10 MB


def solution(ch: str, mod: str) -> str:
    return (ROOT / "companion" / "solutions" / ch / f"{mod}.py").read_text().split("\n", 1)[1]


def run_tests(page, ex, timeout=PY_TIMEOUT, via_keyboard=False):
    if via_keyboard:
        ex.locator(".cm-content").click()
        page.keyboard.press("Control+Enter")
    else:
        ex.locator("[data-run]").click()
    expect(ex.locator("[data-run-status]")).to_have_class(re.compile(r"is-(pass|fail)"), timeout=timeout)
    return ex.locator("[data-run-status]").inner_text()

results: list[tuple[bool, str]] = []


def ok(cond: bool, msg: str):
    results.append((bool(cond), msg))
    print(("  ✓ " if cond else "  ✗ ") + msg)


def serve() -> str:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DIST))
    handler.log_message = lambda *a: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{httpd.server_address[1]}/"


def main() -> int:
    base = serve()
    launch = {"executable_path": os.environ["CHROMIUM"]} if os.environ.get("CHROMIUM") else {}
    with sync_playwright() as p:
        browser = p.chromium.launch(**launch)
        ctx = browser.new_context(viewport={"width": 1440, "height": 1000}, device_scale_factor=1,
                                  permissions=["clipboard-read", "clipboard-write"])
        page = ctx.new_page()
        errors: list[str] = []
        page.on("console", lambda m: m.type == "error" and errors.append(m.text))
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("requestfailed", lambda r: errors.append(f"request failed: {r.url}"))

        # ── Index ────────────────────────────────────────
        page.goto(base + "index.html")
        page.wait_for_selector("html[data-ready='1']")
        ok(page.locator(".card.is-ready").count() == 2, "index lists 2 ready chapters")
        ok(page.locator(".card.is-soon").count() == 26, "index lists 26 coming-soon chapters")
        page.screenshot(path=OUT / "01-index-light.png", full_page=True)

        # ── Chapter render ───────────────────────────────
        page.goto(base + CHAPTER)
        page.wait_for_selector("html[data-ready='1']")
        page.evaluate("document.fonts.ready")
        ok(page.locator(".exercise").count() == 3, "chapter has 3 exercises")
        n_math, n_katex = page.locator(".tex").count(), page.locator(".tex .katex").count()
        ok(n_math > 20 and n_math == n_katex, f"all visible-source maths rendered by KaTeX ({n_katex}/{n_math})")
        ok(page.locator(".katex-error").count() == 0, "no KaTeX errors")
        n_disp = page.locator(".tex[data-display='1']").count()
        ok(n_disp > 5 and page.locator(".tex[data-display='1'] .katex-display").count() == n_disp,
           f"display equations rendered in display mode ({n_disp})")
        fam = page.evaluate("getComputedStyle(document.querySelector('.prose p')).fontFamily")
        ok("Source Serif" in fam, f"prose font is Source Serif ({fam.split(',')[0]})")
        bg = page.evaluate("getComputedStyle(document.body).backgroundColor")
        ok(bg == "rgb(248, 249, 250)", f"light background is dopamine #F8F9FA ({bg})")
        page.screenshot(path=OUT / "02-chapter-top-light.png")

        ex1 = page.locator("#softmax-ce-stable")
        ex1.scroll_into_view_if_needed()
        ex1.screenshot(path=OUT / "03-ex1-blank-tab.png")

        # ── Tabs ─────────────────────────────────────────
        ex1.locator("[data-tab='2']").click()
        expect(ex1.locator("[data-panel='2']")).to_be_visible()
        ok(ex1.locator("[data-panel='1']").is_hidden(), "tab 2 hides panel 1")
        ok(ex1.locator("pre.code .ln").count() > 5, "scaffold code highlighted with line spans")

        # ── Blanks: wrong, wrong → reveal offered, correct ─
        b1 = ex1.locator("input.blank[data-i='1']")
        b1.fill("z.max()")
        b1.press("Enter")
        wrap1 = ex1.locator(".blank-wrap[data-blank='1']")
        ok("is-wrong" in wrap1.get_attribute("class"), "wrong answer marked ✗")
        pill = ex1.locator(".hint-pill[data-for='1']")
        ok(pill.is_visible() and "keepdims" in pill.inner_text(), "hint pill shows conceptual hint")
        ok(pill.locator(".hp-reveal").is_hidden(), "reveal hidden after 1 wrong try")
        b1.fill("z.max(axis=1)")
        b1.press("Enter")
        ok(pill.locator(".hp-reveal").is_visible(), "reveal offered after 2 wrong tries")
        b1.fill("z.max( axis = 1, keepdims = True )")
        b1.press("Enter")
        ok("is-correct" in wrap1.get_attribute("class"), "correct answer (with odd spacing) marked ✓")
        ok(pill.count() == 0, "hint pill removed once correct")
        focused = page.evaluate("document.activeElement.dataset.i")
        ok(focused == "2", "Enter on a correct blank moves focus to the next blank")
        ex1.locator("input.blank[data-i='2']").fill("z - lse")
        ex1.locator("input.blank[data-i='3']").fill("np.arange(B)")
        ex1.locator("input.blank[data-i='4']").fill("picked.mean()")  # missing minus → wrong
        ex1.locator("[data-check-all]").click()
        ok(ex1.locator("[data-score-n]").inner_text() == "3", "check-all scores 3/4")
        ex1.locator(".blank-wrap[data-blank='4']").scroll_into_view_if_needed()
        ex1.locator("[data-panel='2']").screenshot(path=OUT / "04-scaffold-feedback.png")
        # reveal flow on blank 4 (needs 2 wrong)
        b4 = ex1.locator("input.blank[data-i='4']")
        b4.fill("picked.sum()")
        b4.press("Enter")
        ex1.locator(".hint-pill[data-for='4'] .hp-reveal").click()
        ok("is-revealed" in ex1.locator(".blank-wrap[data-blank='4']").get_attribute("class"), "reveal fills & marks blank")
        ok("is-complete" in ex1.locator("[data-score]").get_attribute("class"), "score pill complete when all resolved")

        # ── Persistence ──────────────────────────────────
        page.reload()
        page.wait_for_selector("html[data-ready='1']")
        ok(ex1.locator("[data-panel='2']").is_visible(), "selected tab persists across reload")
        ok("is-correct" in ex1.locator(".blank-wrap[data-blank='1']").get_attribute("class"), "blank state persists across reload")

        # ── Solution split + line linking ────────────────
        ex1.locator("[data-tab='3']").click()
        para = ex1.locator(".explain-para").first
        para.hover()
        hl = ex1.locator(".split-code .ln.is-hl")
        ok(hl.count() == 1 and hl.first.get_attribute("data-line") == "4", "hovering explanation highlights its code line")
        ex1.locator("[data-panel='3']").screenshot(path=OUT / "05-solution-split.png")
        ex1.locator(".split-code .ln[data-line='9']").hover()
        ok("is-hl" in ex1.locator(".explain-para").nth(2).get_attribute("class"), "hovering a code line highlights its note")

        # ── Done toggle → progress ───────────────────────
        ex1.locator("[data-done]").click()
        ok(page.locator("[data-bar-n]").inner_text() == "1", "chapter progress bar counts completion")
        ok(page.locator(".nav-ch.is-current").evaluate("e => getComputedStyle(e).getPropertyValue('--p')").strip() not in ("", "0"),
           "sidebar ring updates")

        # ── Derive exercise: prose blanks + steps ────────
        ex2 = page.locator("#softmax-ce-grad")
        ex2.locator("[data-tab='2']").click()
        for i, ans in [("1", "p_j(δ_ij - p_i)"), ("2", "p_i - \\delta_{iy}"), ("3", "p - e_y"), ("4", "(P - Y)/B")]:
            inp = ex2.locator(f"input.blank[data-i='{i}']")
            inp.fill(ans)
            inp.press("Enter")
        ok(ex2.locator(".blank-wrap.is-correct").count() == 4, "derivation blanks accept plain-text and LaTeX forms")
        regex_inp = ex2.locator("input.blank[data-i='1']")
        ex2.locator("[data-reset-blanks]").click()
        regex_inp.fill("pj(\\delta_ij - pi)")
        regex_inp.press("Enter")
        ok("is-correct" in ex2.locator(".blank-wrap[data-blank='1']").get_attribute("class"), "regex alternative accepts sloppy notation")
        ex2.locator("[data-panel='2']").screenshot(path=OUT / "06-derive-scaffold.png")
        ex2.locator("[data-tab='3']").click()
        ok(ex2.locator(".step.is-hidden").count() == 4, "derivation starts with 1 step shown")
        leaked = ex2.locator(".step.is-hidden .step-content").evaluate_all(
            "els => els.filter(e => e.offsetParent !== null).length")
        ok(leaked == 0, "hidden derivation steps leak no content")
        ex2.locator("[data-step-next]").click()
        ok(ex2.locator(".step.is-hidden").count() == 3, "next step reveals one more")
        ex2.locator("[data-rate='good']").click()
        ex2.locator("[data-panel='3']").screenshot(path=OUT / "07-derive-steps.png")

        # keyboard tabs: focus card, press 1
        ex2.locator(".ex-card").focus()
        page.keyboard.press("1")
        ok(ex2.locator("[data-panel='1']").is_visible(), "number keys switch tabs")
        ex2.locator("[data-timer-toggle]").click()
        page.wait_for_timeout(1200)
        ok(ex2.locator("[data-timer-face]").inner_text() != f"{10}:00", "whiteboard timer counts down")

        # ── Debug exercise ───────────────────────────────
        ex3 = page.locator("#softmax-broadcast-bug")
        ex3.locator("[data-tab='2']").click()
        b = ex3.locator("input.blank")
        b.fill("z.max(axis=1)[:, None]")
        b.press("Enter")
        ok("is-correct" in ex3.locator(".blank-wrap").get_attribute("class"), "debug scaffold accepts [:, None] alternative")
        ex3.screenshot(path=OUT / "08-debug.png")

        # hints
        ex3.locator("[data-tab='1']").click()
        ex3.locator("[data-hint-next]").click()
        ok(ex3.locator(".hint:not([hidden])").count() == 1, "progressive hints reveal one at a time")

        # ── M2: in-browser runner (Pyodide) ──────────────
        ex1 = page.locator("#softmax-ce-stable")
        ex1.locator("[data-tab='1']").click()
        ok(ex1.locator(".cm-editor").count() == 1 and ex1.locator("textarea[data-editor]").is_hidden(),
           "CodeMirror replaces the textarea")
        status = run_tests(page, ex1)
        ok(status == "0/5 tests passing", f"starter runs in Pyodide and fails all tests ({status})")
        ok("NotImplementedError" in ex1.locator(".result.is-fail .result-msg").first.inner_text(),
           "failure shows the exception")
        ok(ex1.locator(".result.is-fail .result-line").first.inner_text().startswith("raise")
           or ex1.locator(".result.is-fail .result-line").first.inner_text().startswith("assert"),
           "failure shows the offending source line")
        page.evaluate("([id, code]) => window.__wb.editors[id].set(code)",
                      ["softmax-ce-stable", solution("ch04", "softmax_ce_stable")])
        status = run_tests(page, ex1, via_keyboard=True)
        ok(status.endswith("5/5 tests passing"), f"Ctrl+Enter runs; solution passes ({status})")
        ex1.locator("[data-panel='1']").screenshot(path=OUT / "14-runner-pass.png")
        # half-right code: one failing assertion with a readable line
        page.evaluate("([id, code]) => window.__wb.editors[id].set(code)", ["softmax-ce-stable",
            solution("ch04", "softmax_ce_stable").replace("return -picked.mean()", "return -picked.sum()")])
        status = run_tests(page, ex1)
        fail_line = ex1.locator(".result.is-fail .result-line").first.inner_text()
        ok(status.startswith("3/5") and fail_line.startswith("assert"), f"partial failure pinpoints the assert ({status})")
        ex1.locator("[data-panel='1']").screenshot(path=OUT / "15-runner-fail.png")
        page.evaluate("([id, code]) => window.__wb.editors[id].set(code)", ["softmax-ce-stable", "def oops(:\n    pass"])
        run_tests(page, ex1)
        ok("SyntaxError" in ex1.locator(".result.is-error").inner_text(), "syntax errors are reported, not swallowed")

        # debug exercise is editable + auto-completes when green
        ex3 = page.locator("#softmax-broadcast-bug")
        ex3.locator("[data-tab='1']").click()
        ok(ex3.locator("[data-done]").get_attribute("aria-pressed") == "false", "debug exercise not yet complete")
        page.evaluate("([id, code]) => window.__wb.editors[id].set(code)",
                      ["softmax-broadcast-bug", solution("ch04", "softmax_broadcast_bug")])
        status = run_tests(page, ex3)
        ok(status.endswith("3/3 tests passing"), f"fixed debug code passes ({status})")
        ok(ex3.locator("[data-done]").get_attribute("aria-pressed") == "true", "all-green run marks the exercise complete")

        # persistence of code + last results
        page.reload()
        page.wait_for_selector("html[data-ready='1']")
        ok("keepdims=True" in page.evaluate("window.__wb.editors['softmax-broadcast-bug'].get()"), "edited code persists across reload")
        ok(ex3.locator(".result.is-pass").count() == 3, "last test results persist across reload")

        # ── Chapter 13: browser + local-torch ────────────
        page.goto(base + CH13)
        page.wait_for_selector("html[data-ready='1']")
        ok(page.locator(".exercise").count() == 4, "chapter 13 has 4 exercises")
        nd = page.locator(".tex[data-display='1']").count()
        ok(nd >= 2 and page.locator(".tex[data-display='1'] .katex-display").count() == nd and page.locator(".katex-error").count() == 0, f"chapter 13 maths renders ({nd} display equations)")
        sd = page.locator("#sdpa-numpy")
        page.evaluate("([id, code]) => window.__wb.editors[id].set(code)", ["sdpa-numpy", "while True:\n    pass"])
        t0 = __import__("time").time()
        run_tests(page, sd, timeout=60_000)
        ok("Timed out" in sd.locator("[data-run-status]").inner_text(), f"infinite loop is stopped ({__import__('time').time() - t0:.0f}s)")
        page.evaluate("([id, code]) => window.__wb.editors[id].set(code)", ["sdpa-numpy", solution("ch13", "sdpa_numpy")])
        status = run_tests(page, sd)
        ok(status.endswith("4/4 tests passing"), f"runner recovers after a timeout; 4-D attention passes ({status})")

        mha = page.locator("#multi-head-attention")
        ok(mha.locator(".cm-editor").count() == 0 and mha.locator("[data-local]").is_visible(), "torch exercise shows the local workflow, not an editor")
        chip = mha.locator("[data-copy^='pytest']")
        ok(chip.get_attribute("data-copy") == "pytest companion/tests/ch13/test_multi_head_attention.py", "local box shows the exact pytest command")
        chip.click()
        ok(page.evaluate("navigator.clipboard.readText()") == chip.get_attribute("data-copy"), "copy chip copies the command")
        mha.scroll_into_view_if_needed()
        mha.screenshot(path=OUT / "16-local-torch.png")
        mha.locator("[data-tab='2']").click()
        for i, ans in enumerate(["self.h, self.dk", "1, 2", "-2, -1", "~mask", "-1", "reshape(B, T, D)"], 1):
            inp = mha.locator(f"input.blank[data-i='{i}']")
            inp.fill(ans)
            inp.press("Enter")
        ok(mha.locator(".blank-wrap.is-correct").count() == 6, "MHA scaffold: all 6 blanks accepted")
        mha.screenshot(path=OUT / "17-mha-scaffold.png")
        sq = page.locator("#attention-sqrt-d")
        sq.locator("[data-tab='2']").click()
        last = sq.locator("input.blank[data-i='5']")
        last.fill("Nearly one-hot")
        last.press("Enter")
        ok("is-correct" in sq.locator(".blank-wrap[data-blank='5']").get_attribute("class"), "free-text blank accepts case-insensitive regex")

        # ── Collapsible sidebar ──────────────────────────
        page.locator("[data-nav-toggle]").click()
        page.wait_for_timeout(450)
        ok(page.evaluate("document.documentElement.classList.contains('nav-collapsed')"), "toggle collapses the sidebar")
        ok(page.locator(".sidebar").bounding_box()["width"] < 2, "collapsed sidebar takes no width")
        page.evaluate("window.scrollTo(0, 0)")
        page.screenshot(path=OUT / "18-sidebar-collapsed.png")
        page.reload()
        page.wait_for_selector("html[data-ready='1']")
        ok(page.evaluate("document.documentElement.classList.contains('nav-collapsed')"), "collapsed state persists")
        page.locator("body").press("[")
        page.wait_for_timeout(450)
        ok(not page.evaluate("document.documentElement.classList.contains('nav-collapsed')"), "[ key re-opens the sidebar")
        ok(page.locator(".sidebar").bounding_box()["width"] > 200, "re-opened sidebar has full width")
        page.goto(base + CHAPTER)
        page.wait_for_selector("html[data-ready='1']")
        ex1 = page.locator("#softmax-ce-stable")

        # ── Dark theme ───────────────────────────────────
        page.locator("[data-theme-toggle]").click()
        ok(page.evaluate("document.documentElement.dataset.theme") == "dark", "theme toggles to dark")
        page.evaluate("window.scrollTo(0, 0)")
        page.screenshot(path=OUT / "09-chapter-top-dark.png")
        ex1.scroll_into_view_if_needed()
        ex1.locator("[data-tab='2']").click()
        ex1.screenshot(path=OUT / "10-ex1-scaffold-dark.png")
        page.goto(base + "index.html")
        page.wait_for_selector("html[data-ready='1']")
        ok(page.locator("[data-continue]").is_visible(), "index offers 'continue where you left off'")
        page.screenshot(path=OUT / "11-index-dark.png", full_page=True)

        # ── Mobile ───────────────────────────────────────
        m = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2)
        mp = m.new_page()
        mp.on("pageerror", lambda e: errors.append(str(e)))
        mp.goto(base + CHAPTER)
        mp.wait_for_selector("html[data-ready='1']")
        overflow = mp.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        ok(overflow <= 0, f"no horizontal page overflow on mobile ({overflow}px)")
        mp.screenshot(path=OUT / "12-mobile.png")
        mp.locator("#softmax-ce-stable [data-tab='2']").click()
        mp.locator("#softmax-ce-stable").screenshot(path=OUT / "13-mobile-scaffold.png")

        # ── Opened straight from disk (file://) ──────────
        fp = m.new_page()
        fp.on("pageerror", lambda e: errors.append(str(e)))
        fp.goto((DIST / CHAPTER).as_uri())
        fp.wait_for_selector("html[data-ready='1']", timeout=10000)
        ok(fp.locator(".tex .katex").count() > 20, "works when opened via file:// (JS + KaTeX)")
        fp.goto((DIST / CH13).as_uri())
        fp.wait_for_selector("html[data-ready='1']", timeout=10000)
        fsd = fp.locator("#sdpa-numpy")
        fp.evaluate("([id, code]) => window.__wb.editors[id].set(code)", ["sdpa-numpy", solution("ch13", "sdpa_numpy")])
        status = run_tests(fp, fsd)
        ok(status.endswith("4/4 tests passing"), f"Pyodide runner works from file:// too ({status})")

        ok(not errors, "no console errors / failed requests" + (f": {errors[:3]}" if errors else ""))
        browser.close()

    failed = [m for good, m in results if not good]
    print(f"\n{len(results) - len(failed)}/{len(results)} browser checks passed · screenshots → {OUT}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
