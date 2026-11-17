// Interview workbook — client behaviour. No framework; progress lives in localStorage.
"use strict";
(() => {

const KEY = "wb:v1";
const editors = {}; // exercise id -> {get, set}; exposed for the headless checks
window.__wb = { editors };
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

/* ── Progress store ───────────────────────────────────── */
const store = {
  data: (() => {
    try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; }
  })(),
  ex(id) {
    this.data.ex ??= {};
    return (this.data.ex[id] ??= { blanks: {} });
  },
  save() { localStorage.setItem(KEY, JSON.stringify(this.data)); },
};

/* ── Helpers ──────────────────────────────────────────── */
const decode = (s) => JSON.parse(new TextDecoder().decode(Uint8Array.from(atob(s), (c) => c.charCodeAt(0))));

// Must match normalise() in build/blanks.py
function normalise(s) {
  return s.trim().replace(/\\left|\\right/g, "").replace(/\s+/g, "").replace(/'/g, '"');
}

function matches(input, answers) {
  const n = normalise(input);
  if (!n) return false;
  return answers.some((a) => {
    if (a.startsWith("re:")) {
      try { return new RegExp(a.slice(3), "i").test(n); } catch { return false; }
    }
    return normalise(a) === n;
  });
}

function renderMath(root = document) {
  if (!window.katex) return;
  $$(".tex", root).forEach((el) => {
    if (el.dataset.rendered) return;
    const tex = el.textContent;
    try {
      katex.render(tex, el, { displayMode: el.dataset.display === "1", throwOnError: false, strict: "ignore" });
    } catch (e) { console.warn("KaTeX", e); }
    el.dataset.rendered = "1";
    el.setAttribute("aria-label", tex);
  });
}

/* ── Theme & nav ──────────────────────────────────────── */
function initChrome() {
  $("[data-theme-toggle]")?.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("wb:theme", next);
  });
  // Sidebar: a drawer on narrow screens, collapsible (and remembered) on wide ones.
  const narrow = () => matchMedia("(max-width: 960px)").matches;
  const navBtn = $("[data-nav-toggle]");
  const paintNav = () => {
    const open = narrow() ? document.body.classList.contains("nav-open") : !document.documentElement.classList.contains("nav-collapsed");
    navBtn?.setAttribute("aria-expanded", open);
    navBtn?.setAttribute("title", (open ? "Hide" : "Show") + " chapters  [");
  };
  const toggleNav = () => {
    if (narrow()) document.body.classList.toggle("nav-open");
    else {
      const collapsed = document.documentElement.classList.toggle("nav-collapsed");
      localStorage.setItem("wb:nav", collapsed ? "collapsed" : "open");
    }
    paintNav();
  };
  navBtn?.addEventListener("click", toggleNav);
  document.addEventListener("keydown", (e) => {
    if (e.key !== "[" || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.target.closest("input, textarea, [contenteditable], .cm-editor")) return;
    toggleNav();
  });
  paintNav();
  // Copy chips (paths and commands in the local-workflow box)
  document.addEventListener("click", async (e) => {
    const chip = e.target.closest("[data-copy]");
    if (!chip) return;
    try { await navigator.clipboard.writeText(chip.dataset.copy); chip.classList.add("is-copied"); }
    catch { chip.classList.add("is-copy-failed"); }
    setTimeout(() => chip.classList.remove("is-copied", "is-copy-failed"), 1300);
  });
  document.addEventListener("click", (e) => {
    if (document.body.classList.contains("nav-open") && !e.target.closest(".sidebar, [data-nav-toggle]")) {
      document.body.classList.remove("nav-open");
    }
  });
}

/* ── Chapter-level progress (rings, bars, TOC) ────────── */
function refreshProgress() {
  const done = (id) => !!store.data.ex?.[id]?.done;
  $$("[data-chapter-progress], [data-chapter-bar]").forEach((el) => {
    const ids = (el.dataset.chapterProgress ?? el.dataset.chapterBar).split(",").filter(Boolean);
    const n = ids.filter(done).length;
    el.style.setProperty("--p", ids.length ? n / ids.length : 0);
    const label = $("[data-bar-n]", el);
    if (label) label.textContent = n;
    if (el.dataset.chapterProgress !== undefined) el.title = `${n}/${ids.length} exercises complete`;
  });
  $$("[data-toc]").forEach((a) => a.classList.toggle("is-done", done(a.dataset.toc)));
}

function rememberVisit(exEl) {
  const title = $(".ex-title", exEl)?.textContent.trim();
  store.data.last = { href: `${location.pathname.split("/").pop()}#${exEl.id}`, title };
  store.save();
}

function initIndex() {
  const link = $("[data-continue]");
  const last = store.data.last;
  if (link && last) {
    link.href = last.href;
    $("[data-continue-label]", link).textContent = last.title;
    link.hidden = false;
  }
}

/* ── Exercise ─────────────────────────────────────────── */
function initExercise(exEl) {
  const id = exEl.dataset.ex;
  const st = store.ex(id);
  const card = $(".ex-card", exEl);
  const tabs = $$("[role=tab]", exEl);
  const panels = $$("[role=tabpanel]", exEl);

  // tabs
  function select(n, focus = false) {
    tabs.forEach((t) => {
      const on = t.dataset.tab === String(n);
      t.setAttribute("aria-selected", on);
      t.tabIndex = on ? 0 : -1;
      if (on && focus) t.focus();
    });
    panels.forEach((p) => (p.hidden = p.dataset.panel !== String(n)));
    exEl.classList.toggle("is-wide", String(n) === "3" && !!$("[data-split]", exEl));
    st.tab = n;
    store.save();
  }
  tabs.forEach((t, i) => {
    t.addEventListener("click", () => { select(t.dataset.tab); rememberVisit(exEl); });
    t.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        const j = (i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length;
        select(tabs[j].dataset.tab, true);
        e.preventDefault();
      }
    });
  });
  card.addEventListener("keydown", (e) => {
    if (e.target.matches("input, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
    if (["1", "2", "3"].includes(e.key)) { select(e.key); e.preventDefault(); }
  });
  if (st.tab) select(st.tab);

  initBlanks(exEl, st);
  initHints(exEl);
  initEditor(exEl, st);
  initTimer(exEl);
  initSteps(exEl, st);
  initRating(exEl, st);
  initSplit(exEl);
  initDone(exEl, st);
}

function initBlanks(exEl, st) {
  const inputs = $$("input.blank", exEl);
  if (!inputs.length) return;
  const log = $("[data-hint-log]", exEl);
  const scoreEl = $("[data-score]", exEl);
  const tabStatus = $('[data-tab-status="2"]', exEl);

  function updateScore() {
    const states = inputs.map((inp) => st.blanks[inp.dataset.i]?.s);
    const correct = states.filter((s) => s === "correct").length;
    const finished = states.every((s) => s === "correct" || s === "revealed");
    $("[data-score-n]", scoreEl).textContent = correct;
    scoreEl.classList.toggle("is-complete", finished);
    tabStatus.textContent = finished ? (correct === inputs.length ? "✓" : `${correct}/${inputs.length}`) : "";
  }

  function pill(inp, b) {
    const i = inp.dataset.i;
    let el = $(`.hint-pill[data-for="${i}"]`, log);
    if (!el) {
      el = document.createElement("div");
      el.className = "hint-pill";
      el.dataset.for = i;
      el.innerHTML = `<span class="hp-n">${i}</span><div><span class="hp-label">What's missing</span>${decode(inp.dataset.h) || "Not quite. Re-read the prerequisites above."}</div><button type="button" class="hp-reveal" hidden>Reveal answer</button>`;
      $(".hp-reveal", el).addEventListener("click", () => reveal(inp));
      log.append(el);
      // keep pills in blank order
      [...log.children].sort((a, z) => a.dataset.for - z.dataset.for).forEach((c) => log.append(c));
      renderMath(el);
    }
    $(".hp-reveal", el).hidden = (b.t || 0) < 2;
    return el;
  }
  const dropPill = (inp) => $(`.hint-pill[data-for="${inp.dataset.i}"]`, log)?.remove();

  function paint(inp, s) {
    const wrap = inp.closest(".blank-wrap");
    wrap.classList.remove("is-correct", "is-wrong", "is-revealed");
    if (s) wrap.classList.add(`is-${s}`);
    inp.readOnly = s === "correct" || s === "revealed";
    inp.setAttribute("aria-invalid", s === "wrong");
  }

  function check(inp, { quiet = false } = {}) {
    const b = (st.blanks[inp.dataset.i] ??= {});
    if (b.s === "correct" || b.s === "revealed") return;
    if (!inp.value.trim()) { if (!quiet) inp.focus(); return; }
    b.v = inp.value;
    if (matches(inp.value, decode(inp.dataset.a))) {
      b.s = "correct";
      dropPill(inp);
    } else {
      b.s = "wrong";
      b.t = (b.t || 0) + 1;
      const wrap = inp.closest(".blank-wrap");
      wrap.classList.remove("shake"); void wrap.offsetWidth; wrap.classList.add("shake");
      pill(inp, b);
    }
    paint(inp, b.s);
    store.save();
    updateScore();
  }

  function reveal(inp) {
    const b = (st.blanks[inp.dataset.i] ??= {});
    inp.value = decode(inp.dataset.a)[0];
    b.v = inp.value;
    b.s = "revealed";
    paint(inp, "revealed");
    dropPill(inp);
    store.save();
    updateScore();
  }

  inputs.forEach((inp, k) => {
    const b = st.blanks[inp.dataset.i];
    if (b?.v) inp.value = b.v;
    if (b?.s) { paint(inp, b.s); if (b.s === "wrong") pill(inp, b); }
    inp.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        check(inp);
        if (st.blanks[inp.dataset.i]?.s === "correct") inputs.slice(k + 1).find((x) => !x.readOnly)?.focus();
      }
    });
    inp.addEventListener("input", () => {
      const s = st.blanks[inp.dataset.i];
      if (s?.s === "wrong") { s.s = ""; paint(inp, ""); }
    });
  });

  $("[data-check-all]", exEl)?.addEventListener("click", () => inputs.forEach((inp) => check(inp, { quiet: true })));
  $("[data-reset-blanks]", exEl)?.addEventListener("click", () => {
    st.blanks = {};
    inputs.forEach((inp) => { inp.value = ""; paint(inp, ""); });
    log.replaceChildren();
    store.save();
    updateScore();
  });
  updateScore();
}

function initHints(exEl) {
  const box = $("[data-hints]", exEl);
  if (!box) return;
  const items = $$(".hint", box);
  const btn = $("[data-hint-next]", box);
  btn.addEventListener("click", () => {
    const next = items.find((h) => h.hidden);
    if (next) { next.hidden = false; renderMath(next); }
    const shown = items.filter((h) => !h.hidden).length;
    $("[data-hint-count]", box).textContent = shown;
    if (shown === items.length) btn.hidden = true;
  });
}

function initEditor(exEl, st) {
  const ta = $("[data-editor]", exEl);
  if (!ta) return;
  const starter = $("template[data-starter]", exEl).content.textContent;
  const tests = $("template[data-tests]", exEl)?.content.textContent ?? "";
  if (st.code != null) ta.value = st.code;

  let timer;
  const persist = (text) => {
    clearTimeout(timer);
    timer = setTimeout(() => { st.code = text; store.save(); }, 250);
  };

  // CodeMirror when available; the plain textarea is the fallback.
  let getCode = () => ta.value;
  let setCode = (text) => { ta.value = text; };
  const host = $("[data-editor-host]", exEl);
  if (window.WBCodeMirror && host) {
    const view = WBCodeMirror.create(host, ta.value, { onChange: persist, onRun: () => runTests() });
    ta.hidden = true;
    getCode = () => view.state.doc.toString();
    setCode = (text) => WBCodeMirror.setDoc(view, text);
    editors[exEl.id] = { get: getCode, set: setCode };
  } else {
    host?.remove();
    ta.addEventListener("input", () => persist(ta.value));
    ta.addEventListener("keydown", (e) => {
      if (e.key === "Tab" && !e.shiftKey) {
        e.preventDefault();
        ta.setRangeText("    ", ta.selectionStart, ta.selectionEnd, "end");
        ta.dispatchEvent(new Event("input"));
      } else if (e.key === "Escape") ta.blur();
      else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); runTests(); }
    });
  }

  $("[data-reset-editor]", exEl)?.addEventListener("click", () => {
    if (getCode() !== starter && !confirm("Discard your code and restore the stub?")) return;
    setCode(starter);
    delete st.code;
    store.save();
  });
  $("[data-copy-editor]", exEl)?.addEventListener("click", async (e) => {
    const footer = "\n\n# --- run the tests ---\nfor _n, _f in list(globals().items()):\n    if _n.startswith('test_'):\n        _f(); print('✓', _n)\n";
    try {
      await navigator.clipboard.writeText(`${getCode()}\n\n# --- tests ---\n${tests}${footer}`);
      flash(e.currentTarget, "Copied ✓");
    } catch { flash(e.currentTarget, "Copy failed"); }
  });

  // ── Runner ──
  const runBtn = $("[data-run]", exEl);
  const status = $("[data-run-status]", exEl);
  const results = $("[data-results]", exEl);
  if (!runBtn) return;
  if (st.lastRun) showResults(st.lastRun, false);

  async function runTests() {
    if (runBtn.disabled) return;
    if (!window.WBRunner) { status.textContent = "Runner unavailable."; return; }
    runBtn.disabled = true;
    exEl.classList.add("is-running");
    status.className = "run-status";
    status.textContent = "Starting…";
    rememberVisit(exEl);
    try {
      const res = await WBRunner.run(getCode(), tests, (msg) => (status.textContent = msg));
      showResults(res, true);
    } catch (err) {
      status.className = "run-status is-fail";
      status.textContent = err.message;
    } finally {
      runBtn.disabled = false;
      exEl.classList.remove("is-running");
    }
  }
  runBtn.addEventListener("click", runTests);

  function showResults(res, fresh) {
    const passed = res.tests.filter((t) => t.ok).length;
    const total = res.tests.length;
    const allOk = !res.error && total > 0 && passed === total;
    status.className = "run-status " + (allOk ? "is-pass" : "is-fail");
    status.textContent = res.error ? "Your code didn't run" : `${passed}/${total} tests passing`;
    results.hidden = false;
    results.replaceChildren();
    if (res.error) results.append(errorRow(res.error));
    for (const t of res.tests) {
      const li = document.createElement("div");
      li.className = "result " + (t.ok ? "is-pass" : "is-fail");
      li.innerHTML = `<span class="result-mark" aria-hidden="true">${t.ok ? "✓" : "✗"}</span>
        <span class="result-name"></span><span class="result-ms">${t.ok ? t.ms + " ms" : ""}</span>`;
      $(".result-name", li).textContent = t.name.replace(/^test_/, "").replace(/_/g, " ");
      if (!t.ok) li.append(detail(t));
      results.append(li);
    }
    if (fresh && window.WBRunner?.mode === "main") {
      const note = document.createElement("p");
      note.className = "result-note";
      note.textContent = "Opened from disk, so tests run on the page itself and an infinite loop would freeze it. `make serve` gives you the interruptible runner.";
      results.append(note);
    }
    if (res.stdout && res.stdout.trim()) {
      const pre = document.createElement("pre");
      pre.className = "result-stdout";
      pre.textContent = res.stdout;
      results.append(pre);
    }
    $('[data-tab-status="1"]', exEl).dataset.pass = allOk ? "1" : "";
    if (fresh) {
      st.lastRun = { tests: res.tests, error: res.error, stdout: (res.stdout || "").slice(-1000) };
      if (allOk && !st.done) exEl._setDone?.(true);
      store.save();
    }
  }

  function detail(t) {
    const d = document.createElement("div");
    d.className = "result-detail";
    const msg = document.createElement("div");
    msg.className = "result-msg";
    msg.textContent = t.msg;
    d.append(msg);
    if (t.line) {
      const code = document.createElement("code");
      code.className = "result-line";
      code.textContent = t.line;
      const where = document.createElement("span");
      where.className = "result-where";
      where.textContent = t.where;
      d.append(code, where);
    }
    return d;
  }

  function errorRow(err) {
    const row = document.createElement("div");
    row.className = "result is-error";
    row.innerHTML = `<span class="result-mark" aria-hidden="true">!</span><span class="result-name">Couldn't run your code</span>`;
    row.append(detail(err));
    return row;
  }
}

function flash(btn, msg) {
  const old = btn.textContent;
  btn.textContent = msg;
  setTimeout(() => (btn.textContent = old), 1400);
}

function initTimer(exEl) {
  const box = $("[data-timer]", exEl);
  if (!box) return;
  const total = +box.dataset.seconds;
  const face = $("[data-timer-face]", box);
  const btn = $("[data-timer-toggle]", box);
  let left = total, handle = null;
  const fmt = (s) => `${s < 0 ? "−" : ""}${Math.floor(Math.abs(s) / 60)}:${String(Math.abs(s) % 60).padStart(2, "0")}`;
  btn.addEventListener("click", () => {
    if (handle) {
      clearInterval(handle); handle = null; left = total;
      face.textContent = fmt(left); btn.textContent = "Start timer";
      box.classList.remove("is-running", "is-over");
      return;
    }
    box.classList.add("is-running");
    btn.textContent = "Reset";
    handle = setInterval(() => {
      left -= 1;
      face.textContent = fmt(left);
      box.classList.toggle("is-over", left < 0);
    }, 1000);
  });
}

function initSteps(exEl, st) {
  const list = $("[data-steps]", exEl);
  if (!list) return;
  const steps = $$(".step", list);
  const next = $("[data-step-next]", exEl);
  const count = $("[data-step-count]", exEl);
  function show(n, animate = true) {
    steps.forEach((s, i) => {
      const was = s.classList.contains("is-hidden");
      s.classList.toggle("is-hidden", i >= n);
      if (was && i < n && animate) { s.classList.add("is-new"); setTimeout(() => s.classList.remove("is-new"), 400); }
    });
    count.textContent = `${n} / ${steps.length}`;
    next.disabled = n >= steps.length;
    next.textContent = n >= steps.length ? "All steps shown" : "Next step ↓";
    st.steps = n;
    store.save();
  }
  next.addEventListener("click", () => {
    show(Math.min(steps.length, (st.steps || 1) + 1));
    steps[(st.steps || 1) - 1]?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  });
  $("[data-step-all]", exEl).addEventListener("click", () => show(steps.length));
  show(st.steps || 1, false);
}

function initRating(exEl, st) {
  const box = $("[data-rating]", exEl);
  if (!box) return;
  const btns = $$("[data-rate]", box);
  const paint = () => btns.forEach((b) => b.setAttribute("aria-pressed", b.dataset.rate === st.rating?.grade));
  btns.forEach((b) => b.addEventListener("click", () => {
    st.rating = { grade: b.dataset.rate, at: new Date().toISOString() }; // feeds the Drill Deck (M4)
    store.save();
    paint();
  }));
  paint();
}

function initSplit(exEl) {
  $$("[data-split]", exEl).forEach((split) => {
    const lines = $$(".ln", split);
    const paras = $$(".explain-para", split);
    const range = (p) => p.dataset.lines.split("-").map(Number);
    const hl = (p, on) => {
      p.classList.toggle("is-hl", on);
      const [a, b] = range(p);
      lines.forEach((l) => { const n = +l.dataset.line; if (n >= a && n <= b) l.classList.toggle("is-hl", on); });
    };
    paras.forEach((p) => {
      ["mouseenter", "focus"].forEach((ev) => p.addEventListener(ev, () => hl(p, true)));
      ["mouseleave", "blur"].forEach((ev) => p.addEventListener(ev, () => hl(p, false)));
    });
    lines.forEach((l) => {
      const n = +l.dataset.line;
      const owners = paras.filter((p) => { const [a, b] = range(p); return n >= a && n <= b; });
      if (!owners.length) return;
      l.addEventListener("mouseenter", () => owners.forEach((p) => hl(p, true)));
      l.addEventListener("mouseleave", () => owners.forEach((p) => hl(p, false)));
    });
  });
}

function initDone(exEl, st) {
  const btn = $("[data-done]", exEl);
  const chip = $("[data-done-chip]", exEl);
  const label = $("[data-done-label]", exEl);
  const paint = () => {
    btn.setAttribute("aria-pressed", !!st.done);
    label.textContent = st.done ? "Completed" : "Mark exercise complete";
    chip.hidden = !st.done;
    exEl.classList.toggle("is-done", !!st.done);
    $('[data-tab-status="1"]', exEl).textContent = st.done ? "✓" : "";
  };
  const set = (v) => {
    st.done = v;
    store.save();
    paint();
    refreshProgress();
    if (st.done) { exEl.classList.add("just-done"); setTimeout(() => exEl.classList.remove("just-done"), 1000); }
  };
  btn.addEventListener("click", () => set(!st.done));
  exEl._setDone = set;
  paint();
}

/* ── Boot ─────────────────────────────────────────────── */
initChrome();
renderMath();
$$(".exercise").forEach(initExercise);
refreshProgress();
initIndex();
document.documentElement.dataset.ready = "1";
})();
