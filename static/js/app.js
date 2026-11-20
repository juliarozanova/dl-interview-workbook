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
  const { items } = paintChecklistCount();
  paintDeckCount();
  paintChecklistPage(items);
  paintDashboard(items);
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

/* ── Site data, checklist, dashboard (M3) ─────────────── */
const WB = (() => {
  try { return JSON.parse($("#wb-data")?.textContent || "null"); } catch { return null; }
})();
const isDone = (id) => !!store.data.ex?.[id]?.done;
const DAY = 86_400_000;
const INTERVAL_DAYS = { again: 1, hard: 3, good: 7, easy: 14 };

function checklistStatus() {
  if (!WB) return [];
  return WB.checklist.map((item) => {
    const exs = WB.exercises.filter((e) => e.tags.some((t) => item.tags.includes(t)));
    const nDone = exs.filter((e) => isDone(e.id)).length;
    const auto = exs.length > 0 && nDone === exs.length;
    const manual = !!store.data.checklist?.[item.id];
    return { id: item.id, exs, nDone, auto, manual, done: auto || manual };
  });
}

function setRing(el, frac) {
  el?.style.setProperty("--p", Math.max(0, Math.min(1, frac)));
}

function paintDeckCount() {
  // Derivations due, the same number everywhere (interview-question cards are counted on the deck page itself).
  const n = WB ? dueDerivations().filter((d) => d.isDue).length : 0;
  $$("[data-deck-count]").forEach((el) => (el.textContent = n ? `${n} due` : ""));
}

function paintChecklistCount() {
  const items = checklistStatus();
  const n = items.filter((i) => i.done).length;
  $$("[data-checklist-count]").forEach((el) => (el.textContent = items.length ? `${n}/${items.length}` : ""));
  return { items, n };
}

function initChecklistPage() {
  const rows = $$("[data-item]");
  if (!rows.length) return;
  rows.forEach((row) => {
    $("[data-cl-toggle]", row).addEventListener("click", () => {
      store.data.checklist ??= {};
      const s = checklistStatus().find((i) => i.id === row.dataset.item);
      if (s.auto) return; // already ticked by its exercises
      store.data.checklist[row.dataset.item] = !s.manual;
      store.save();
      refreshProgress();
    });
  });
}

function paintChecklistPage(items) {
  const rows = $$("[data-item]");
  if (!rows.length) return;
  const byId = Object.fromEntries(items.map((i) => [i.id, i]));
  rows.forEach((row) => {
    const s = byId[row.dataset.item];
    row.classList.toggle("is-done", s.done);
    row.classList.toggle("is-auto", s.auto);
    const btn = $("[data-cl-toggle]", row);
    btn.setAttribute("aria-pressed", s.done);
    btn.title = s.auto ? "Ticked automatically: all linked exercises are complete" : s.manual ? "Ticked by hand: click to untick" : "Tick by hand";
    $("[data-cl-state]", row).textContent = s.exs.length ? `${s.nDone}/${s.exs.length}` : "";
    $$("[data-ex-link]", row).forEach((a) => a.classList.toggle("is-done", isDone(a.dataset.exLink)));
  });
  const n = items.filter((i) => i.done).length;
  const bar = $("[data-checklist-bar]");
  setRing(bar, n / items.length);
  $("[data-checklist-n]").textContent = n;
  $("[data-checklist-total]").textContent = items.length;
}

function currentWeek() {
  const start = store.data.startDate;
  if (!start) return null;
  const days = Math.floor((Date.now() - new Date(start + "T00:00:00").getTime()) / DAY);
  return { raw: Math.floor(days / 7) + 1, days };
}

function dueDerivations() {
  if (!WB) return [];
  const now = Date.now();
  return WB.exercises.filter((e) => e.type === "derive").map((e) => {
    const r = store.data.ex?.[e.id]?.rating;
    const due = r ? new Date(r.at).getTime() + INTERVAL_DAYS[r.grade] * DAY : 0;
    return { ...e, rating: r, due, isDue: due <= now };
  }).sort((a, b) => a.due - b.due);
}

function paintDashboard(checklistItems) {
  const dash = $("[data-dashboard]");
  if (!dash || !WB) return;
  const total = WB.exercises.length;
  const done = WB.exercises.filter((e) => isDone(e.id)).length;
  $("[data-stat='exercises-n']").textContent = done;
  $("[data-stat='exercises-text']").textContent = `${done} of ${total}`;
  setRing($("[data-stat-ring='exercises']"), done / total);
  const cl = checklistItems.filter((i) => i.done).length;
  $("[data-stat='checklist-n']").textContent = cl;
  $("[data-stat='checklist-text']").textContent = `${cl} of ${checklistItems.length}`;
  setRing($("[data-stat-ring='checklist']"), cl / checklistItems.length);
  const drills = dueDerivations();
  const practised = drills.filter((d) => d.rating).length;
  $("[data-stat='drills-n']").textContent = practised;
  $("[data-stat='drills-text']").textContent = `${practised} of ${drills.length}`;
  setRing($("[data-stat-ring='drills']"), drills.length ? practised / drills.length : 0);

  // Derivations due
  const list = $("[data-due-list]");
  const due = drills.filter((d) => d.isDue).slice(0, 6);
  list.replaceChildren(...due.map((d) => {
    const li = document.createElement("li");
    li.innerHTML = `<a class="due-item" href="${d.href}"><span class="cl-ex-n"></span><span class="due-title"></span><span class="due-tag"></span></a>`;
    $(".cl-ex-n", li).textContent = d.number;
    $(".due-title", li).textContent = d.title;
    const tag = $(".due-tag", li);
    tag.textContent = d.rating ? `last: ${d.rating.grade}` : "new";
    tag.classList.toggle("is-new", !d.rating);
    return li;
  }));
  $("[data-due-empty]").hidden = due.length > 0;

  // This week
  const wk = currentWeek();
  $("[data-week-setup]").hidden = !!wk;
  $("[data-week-body]").hidden = !wk;
  const pill = $("[data-week-pill]");
  pill.hidden = !wk;
  $$(".card-now").forEach((c) => (c.hidden = true));
  $$(".card").forEach((c) => c.classList.remove("is-now"));
  if (!wk) return;
  const week = Math.min(12, Math.max(1, wk.raw));
  pill.textContent = wk.raw < 1 ? `Starts in ${-wk.days} day${wk.days === -1 ? "" : "s"}` : wk.raw > 12 ? "Plan complete · week 12" : `Week ${week} of 12`;
  const chapters = WB.chapters.filter((c) => c.weeks.includes(week));
  $("[data-week-list]").replaceChildren(...chapters.map((c) => {
    const li = document.createElement("li");
    const n = c.exercises.filter(isDone).length;
    li.className = "week-item" + (c.href ? "" : " is-soon");
    li.innerHTML = c.href
      ? `<a href="${c.href}"><span class="cl-ex-n">${c.num}</span><span class="wk-title"></span><span class="wk-prog">${n}/${c.exercises.length}</span></a>`
      : `<span><span class="cl-ex-n">${c.num}</span><span class="wk-title"></span><span class="wk-prog">coming soon</span></span>`;
    $(".wk-title", li).textContent = c.title;
    if (c.href && n === c.exercises.length && n > 0) li.classList.add("is-done");
    return li;
  }));
  $$(".card[data-weeks]").forEach((card) => {
    if (card.dataset.weeks.split(",").map(Number).includes(week)) {
      card.classList.add("is-now");
      $(".card-now", card).hidden = false;
    }
  });
}

/* ── Drill deck (M4) ──────────────────────────────────── */
// Derivation cards share their rating with the exercise page (store.data.ex[id].rating);
// interview-question cards keep theirs in store.data.cards[cardId].
function cardRating(card) {
  return card.dataset.kind === "derivation"
    ? store.data.ex?.[card.dataset.card]?.rating
    : store.data.cards?.[card.dataset.card];
}

function cardDue(card) {
  const r = cardRating(card);
  return r ? new Date(r.at).getTime() + INTERVAL_DAYS[r.grade] * DAY : 0;
}

function initDeck() {
  const deck = $("[data-deck]");
  if (!deck) return;
  const cards = $$("[data-card]", deck);
  const reveal = $("[data-reveal]", deck);
  const grades = $("[data-grades]", deck);
  const empty = $("[data-empty]", deck);
  let filter = "all", current = null, skipped = new Set();

  const inView = (c) => filter === "all" || (filter === "star" ? c.dataset.star === "1" : c.dataset.kind === filter);
  const fmtDays = (ms) => {
    const d = Math.ceil(ms / DAY);
    return d <= 1 ? "tomorrow" : `in ${d} days`;
  };

  function queue() {
    const now = Date.now();
    // Built order is ★ first, then derivations, then by chapter; stable sort keeps it within equal due times.
    return cards.filter((c) => inView(c) && cardDue(c) <= now && !skipped.has(c))
                .sort((a, b) => cardDue(a) - cardDue(b));
  }

  function stats() {
    const now = Date.now(), view = cards.filter(inView);
    $("[data-deck-due]", deck).textContent = view.filter((c) => cardRating(c) && cardDue(c) <= now).length;
    $("[data-deck-new]", deck).textContent = view.filter((c) => !cardRating(c)).length;
    $("[data-deck-total]", deck).textContent = view.length;
  }

  function show() {
    cards.forEach((c) => (c.hidden = true));
    stats();
    const q = queue();
    current = q[0] || null;
    reveal.hidden = !current;
    grades.hidden = true;
    $("[data-skip]", deck).hidden = !current || q.length < 2;
    empty.hidden = !!current;
    if (!current) {
      const upcoming = cards.filter(inView).map(cardDue).filter((t) => t > Date.now()).sort((a, b) => a - b)[0];
      $("[data-empty-next]", empty).textContent = upcoming ? `Next card due ${fmtDays(upcoming - Date.now())}.` : "Nothing in this view yet.";
      return;
    }
    current.hidden = false;
    $("[data-answer]", current).hidden = true;
    const r = cardRating(current);
    $("[data-due-label]", current).textContent = r ? `last: ${r.grade}` : "new";
    current.classList.remove("is-flipped");
  }

  function flip() {
    if (!current || !grades.hidden) return;
    $("[data-answer]", current).hidden = false;
    current.classList.add("is-flipped");
    renderMath(current);
    reveal.hidden = true;
    grades.hidden = false;
    $("[data-grade='good']", grades).focus({ preventScroll: true });
  }

  function grade(g) {
    if (!current || grades.hidden) return;
    const rating = { grade: g, at: new Date().toISOString() };
    if (current.dataset.kind === "derivation") store.ex(current.dataset.card).rating = rating;
    else (store.data.cards ??= {})[current.dataset.card] = rating;
    store.save();
    refreshProgress();
    show();
  }

  reveal.addEventListener("click", flip);
  $$("[data-grade]", grades).forEach((b) => b.addEventListener("click", () => grade(b.dataset.grade)));
  $("[data-skip]", deck).addEventListener("click", () => { skipped.add(current); show(); });
  $$("[data-filter]", deck).forEach((b) => b.addEventListener("click", () => {
    filter = b.dataset.filter;
    skipped = new Set();
    $$("[data-filter]", deck).forEach((x) => x.setAttribute("aria-checked", x === b));
    show();
  }));
  document.addEventListener("keydown", (e) => {
    if (e.target.closest("input, textarea, select") || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === " " && !reveal.hidden) { e.preventDefault(); flip(); }
    const g = { 1: "again", 2: "hard", 3: "good", 4: "easy" }[e.key];
    if (g && !grades.hidden) { e.preventDefault(); grade(g); }
  });
  show();
}

function initDashboard() {
  const form = $("[data-start-form]");
  if (!form) return;
  const input = $("[data-start-input]");
  input.value = store.data.startDate || new Date().toISOString().slice(0, 10);
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    store.data.startDate = input.value;
    store.save();
    refreshProgress();
  });
  $("[data-start-reset]").addEventListener("click", () => {
    delete store.data.startDate;
    store.save();
    refreshProgress();
    input.focus();
  });
}

/* ── Exercise ─────────────────────────────────────────── */
function initExercise(exEl) {
  const id = exEl.dataset.ex;
  const st = store.ex(id);
  const card = $(".ex-card", exEl);
  const tabs = $$("[role=tab]", exEl);
  const panels = $$("[role=tabpanel]", exEl);

  // tabs
  const locked = (n) => exEl.dataset.type === "predict" && !st.predicted && String(n) !== "1";
  function select(n, focus = false) {
    if (locked(n)) {
      const fb = $("[data-predict-feedback]", exEl);
      fb.className = "predict-feedback is-nudge";
      fb.textContent = "Commit to a prediction first (or skip).";
      return;
    }
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

  initPredict(exEl, st, tabs);
  initDesign(exEl, st);
  initInterviewTimer(exEl);
  initBlanks(exEl, st);
  initHints(exEl);
  initEditor(exEl, st);
  initTimer(exEl);
  initSteps(exEl, st);
  initRating(exEl, st);
  initSplit(exEl);
  initDone(exEl, st);
}

function initDesign(exEl, st) {
  const notes = $("[data-notes]", exEl);
  if (notes) {
    if (st.notes) notes.value = st.notes;
    let t;
    notes.addEventListener("input", () => {
      clearTimeout(t);
      t = setTimeout(() => { st.notes = notes.value; store.save(); }, 250);
    });
  }
  const boxes = $$("[data-rubric-i]", exEl);
  if (!boxes.length) return;
  st.rubric ??= [];
  const paint = () => {
    const n = boxes.filter((b) => b.checked).length;
    $("[data-rubric-n]", exEl).textContent = n;
    $("[data-rubric-score]", exEl).classList.toggle("is-complete", n === boxes.length);
    $('[data-tab-status="2"]', exEl).textContent = n ? `${n}/${boxes.length}` : "";
  };
  boxes.forEach((b) => {
    b.checked = st.rubric.includes(+b.dataset.rubricI);
    b.addEventListener("change", () => {
      st.rubric = boxes.filter((x) => x.checked).map((x) => +x.dataset.rubricI);
      store.save();
      paint();
    });
  });
  paint();
}

// Interview mode: a per-exercise clock in the header (only visible in interview mode).
function initInterviewTimer(exEl) {
  const chip = $("[data-iv-timer]", exEl);
  if (!chip) return;
  const total = +chip.dataset.seconds;
  const face = $("[data-iv-face]", chip);
  let left = total, handle = null;
  const fmt = (s) => `${s < 0 ? "−" : ""}${Math.floor(Math.abs(s) / 60)}:${String(Math.abs(s) % 60).padStart(2, "0")}`;
  chip.addEventListener("click", () => {
    if (handle) {
      clearInterval(handle);
      handle = null;
      left = total;
      face.textContent = fmt(left);
      chip.classList.remove("is-running", "is-over");
      chip.firstChild.textContent = "⏵ ";
      return;
    }
    chip.classList.add("is-running");
    chip.firstChild.textContent = "⏹ ";
    handle = setInterval(() => {
      left -= 1;
      face.textContent = fmt(left);
      chip.classList.toggle("is-over", left < 0);
    }, 1000);
  });
}

function initInterviewMode() {
  const btn = $("[data-interview-toggle]");
  const root = document.documentElement;
  const paint = () => {
    const on = root.classList.contains("interview-mode");
    btn?.setAttribute("aria-pressed", on);
    if (btn) btn.title = on ? "Leave interview mode" : "Interview mode: hide scaffolds and solutions, add a clock";
  };
  btn?.addEventListener("click", () => {
    const on = root.classList.toggle("interview-mode");
    localStorage.setItem("wb:interview", on ? "1" : "");
    paint();
  });
  paint();
}

function initPredict(exEl, st, tabs) {
  const form = $("[data-predict]", exEl);
  if (!form) return;
  const correct = decode(form.dataset.correct);
  const radios = $$("input[type=radio]", form);
  const commit = $("[data-commit]", form);
  const fb = $("[data-predict-feedback]", form);
  const letter = (i) => "ABCDEFGH"[i];

  function paint() {
    const p = st.predicted;
    tabs.forEach((t) => {
      const lock = !p && t.dataset.tab !== "1";
      t.classList.toggle("is-locked", lock);
      t.setAttribute("aria-disabled", lock);
    });
    if (!p) return;
    radios.forEach((r, i) => {
      r.disabled = true;
      r.checked = i === p.choice;
      const label = r.closest(".choice");
      label.classList.toggle("is-correct", i === correct);
      label.classList.toggle("is-wrong", i === p.choice && i !== correct);
    });
    commit.disabled = true;
    commit.textContent = "Prediction committed";
    $("[data-predict-skip]", form).hidden = true;
    const status = $('[data-tab-status="1"]', exEl);
    if (p.choice == null) {
      fb.className = "predict-feedback";
      fb.textContent = `Skipped. The answer is ${letter(correct)}. Run the experiment to see why.`;
    } else if (p.choice === correct) {
      fb.className = "predict-feedback is-right";
      fb.textContent = "✓ Right. Now confirm it in the experiment.";
      status.textContent = "✓";
    } else {
      fb.className = "predict-feedback is-wrong";
      fb.textContent = `✗ Not quite: it's ${letter(correct)}. The experiment shows why.`;
      status.textContent = "✗";
    }
  }
  radios.forEach((r) => r.addEventListener("change", () => (commit.disabled = false)));
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const chosen = radios.findIndex((r) => r.checked);
    if (chosen < 0 || st.predicted) return;
    st.predicted = { choice: chosen, at: new Date().toISOString() };
    store.save();
    paint();
  });
  $("[data-predict-skip]", form).addEventListener("click", () => {
    st.predicted = { choice: null, at: new Date().toISOString() };
    store.save();
    paint();
  });
  paint();
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
    const fine = !res.error && passed === total;
    const noun = exEl.dataset.type === "predict" ? "claims hold" : "tests passing";
    status.className = "run-status " + (fine ? "is-pass" : "is-fail");
    status.textContent = res.error ? "Your code didn't run" : total ? `${passed}/${total} ${noun}` : "Finished";
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
// Printing: closed <details> can't be opened by CSS, so open the prerequisites for the printout and restore after.
window.addEventListener("beforeprint", () => {
  $$("details.prereq:not([open])").forEach((d) => { d.open = true; d.dataset.printOpened = "1"; });
});
window.addEventListener("afterprint", () => {
  $$("details[data-print-opened]").forEach((d) => { d.open = false; delete d.dataset.printOpened; });
});

initChrome();
initInterviewMode();
renderMath();
$$(".exercise").forEach(initExercise);
initChecklistPage();
initDashboard();
initDeck();
refreshProgress();
initIndex();
document.documentElement.dataset.ready = "1";
})();
