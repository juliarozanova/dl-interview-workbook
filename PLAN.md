# Interview Practice Workbook: Plan, Structure & Design

> Handoff document for the implementing agent. Read `brief.md` first if you have it (it is local-only and git-ignored); this file turns it into a buildable spec.
> Guiding text: Simon Prince, *Understanding Deep Learning* (UDL), https://udlbook.github.io/udlbook/

---

## 1. Goals (what "done" looks like)

1. A **static HTML workbook** generated from **Markdown sources**, nice to read and fast to use.
2. **Implementation-first.** Each exercise combines theory and code, unless it is clearly only one of them.
3. **Self-contained.** Prerequisite knowledge sits next to the exercise. UDL is a pointer, not a dependency.
4. Each exercise has **three layers (tabs)**:
   1. **Blank:** task statement, signature, shape contract, tests. You write the code.
   2. **Scaffold:** a partly worked solution with small inline blanks, an auto ✓/✗ for each blank, and a *conceptual* hint when an answer is wrong.
   3. **Solution:** the full implementation **beside** a short explanation. Keep code comments minimal.
5. Supports the **spoken** interview: derivation drills you can rehearse at a whiteboard, plus an interview mode.
6. Uses tiny synthetic datasets, plus MNIST and Colored MNIST.

Non-goals for v1: user accounts, a server backend, running PyTorch in the browser.

---

## 2. Key technical decisions

| Decision | Choice | Why |
|---|---|---|
| Source format | One Markdown file per chapter with custom `:::` container directives (spec in §5) | Prose stays as Markdown; structure stays machine-readable |
| Build | Python: `markdown-it-py` + `mdit-py-plugins` (container, dollarmath, front_matter, attrs) + Jinja2 templates | Fits the Python/ML toolchain; one command builds the site |
| Code highlighting | Pygments, server-side | No JS needed; the output is stable |
| Maths | KaTeX (vendored locally) with auto-render | Fast; works offline |
| Blank checking | **In-browser string matching**: normalise whitespace, accept a list of alternatives, optional `re:` regex | Instant and offline. No runtime needed. Works for torch code too |
| Running NumPy/pure-Python code | **Pyodide**, lazy-loaded only when you press ▶ Run, in a CodeMirror 6 editor | Lets the Blank tab run tests in the browser |
| Running PyTorch code | A **companion Python package** (`companion/`) holding stubs and pytest tests, both **generated from the same Markdown** | Torch can't run in Pyodide; one source of truth |
| Progress | `localStorage`, keyed by exercise id; export/import as JSON | No backend |
| Serving | `make serve`, which runs `python -m http.server` in `dist/` | Pyodide needs http://; `file://` still works for reading |

**Highlighting trick for blanks:** before running Pygments, replace each blank with a sentinel identifier such as `__BLANK_3__`. A valid identifier gets lexed as a single Name token. After highlighting, swap each sentinel's `<span>` for the `<input>` element. This avoids broken highlighting around the inputs.

**Single source of truth:** the build script extracts, from each exercise,
- `companion/exercises/<chapter>/<ex_id>.py`: the stub, from `:::starter`
- `companion/solutions/<chapter>/<ex_id>.py`: from `:::solution`
- `companion/tests/<chapter>/test_<ex_id>.py`: from `:::tests`

`make verify` runs every test against its *solution* (and against the scaffold with the canonical answers filled in), so no exercise can ship broken. Make this a hard build gate.

---

## 3. Repository layout

```
interview_practice/
├── brief.md                     # original brief: local only, git-ignored
├── PLAN.md                      # this file
├── Makefile                     # build / serve / verify / clean
├── pyproject.toml               # build deps + companion package
├── content/
│   ├── _meta.yaml               # part/chapter order, week mapping, checklist mapping
│   ├── 00-how-to-use.md
│   ├── part1-foundations/
│   │   ├── 01-tensors-shapes-broadcasting.md
│   │   └── ...
│   ├── part2-architectures/
│   ├── part3-generative-advanced/
│   └── part4-research-interview/
├── build/
│   ├── build.py                 # entry point
│   ├── directives.py            # ::: container parsing → exercise model
│   ├── blanks.py                # {{ }} parsing, sentinel highlight trick
│   ├── extract.py               # writes companion stubs/solutions/tests
│   └── verify.py
├── templates/                   # Jinja2: base, chapter, index, drills, checklist
├── static/
│   ├── css/  (tokens.css, base.css, components.css, print.css)
│   ├── js/   (tabs.js, blanks.js, runner.js [pyodide], progress.js, drills.js, interview-mode.js)
│   └── vendor/ (katex, codemirror bundle)
├── companion/
│   ├── workbook/
│   │   ├── data.py              # dataset generators (numpy-only for synthetic!)
│   │   ├── testing.py           # assert_close, gradcheck helpers, shape asserts
│   │   └── viz.py               # tiny matplotlib helpers
│   ├── exercises/   (generated)
│   ├── solutions/   (generated)
│   └── tests/       (generated)
└── dist/                        # generated site (git-ignored)
```

---

## 4. Exercise types

Every block declares a `type`. The three-tab pattern belongs to **implement**. The other types adapt it.

| Type | Tab 1 | Tab 2 | Tab 3 | Used for |
|---|---|---|---|---|
| **implement** | Blank (stub + tests) | Scaffold (code with blanks) | Solution ‖ explanation | Most exercises |
| **derive** | Prompt ("derive X out loud") | Scaffold (equations with blanks) | Full derivation, with a **step-by-step reveal** | The whiteboard drills |
| **debug** | Buggy code + symptom ("loss is NaN after 200 steps") | Hints narrowing the location; blank for the fixed line | Bug explained + fix | Debugging skill (§3 of guideline) |
| **predict** | "What happens if…?" Commit to a prediction (MCQ or free text) | Run the experiment (Pyodide or local) | Explanation of the result | Init, LR, norm, and shift intuition |
| **design** | Open prompt (ablation, experiment design) | Checklist scaffold (what a good answer covers) | Model answer + rubric | Research methodology and spoken rounds |

Each exercise header carries the metadata badges: **UDL ref**, **type**, **difficulty** (1–3 dots), **time estimate**, **runtime** (`browser` / `local-torch` / `none`), and **checklist tags**.

---

## 5. Authoring spec (Markdown directives)

### 5.1 Chapter front matter

```yaml
---
id: ch04-losses
title: Loss functions from maximum likelihood
part: 1
weeks: [2]
udl: ["Ch. 5"]
checklist: ["bias-variance", "custom-losses"]
summary: Every standard loss is a negative log-likelihood. Implement them stably.
---
```

### 5.2 Containers

```
::::exercise{id=softmax-ce-stable type=implement difficulty=2 time=15 runtime=browser udl="5.5" tags="numerical-stability,custom-losses"}
# Numerically stable softmax cross-entropy

:::prereq
Collapsible "what you need to know" box. Markdown + KaTeX. Self-contained theory lives here.
:::

:::theory
Main theory text shown above the tabs (short!).
:::

:::task
What to implement, signature, **shape contract table**, edge cases.
:::

:::starter{lang=python}
def softmax_ce(logits, y):
    """logits: (B, C) float, y: (B,) int -> scalar mean loss"""
    raise NotImplementedError
:::

:::scaffold{lang=python}
def softmax_ce(logits, y):
    m = {{ logits.max(axis=1, keepdims=True) | logits.max(1, keepdims=True) ? You must keep the class axis so the subtraction broadcasts over C. }}
    z = logits - m
    lse = np.log(np.exp(z).sum(axis={{ 1 ? Which axis indexes classes? }}, keepdims=True))
    logp = z - lse
    return -logp[np.arange(len(y)), {{ y ? Pick the log-prob of the *true* class per row. }}].mean()
:::

:::solution{lang=python}
...full code...
:::

:::explain
Paragraphs here. A paragraph may start with `[L3-5]` to link itself to solution lines 3–5
(hovering it highlights those lines).
:::

:::tests{lang=python}
# pytest-style; also runnable in Pyodide for runtime=browser
:::

:::hints
1. Progressive hint 1 (conceptual)
2. Progressive hint 2 (more specific)
:::

:::interview
Follow-up questions an interviewer might ask: "Why not compute softmax then log?" "What's the gradient wrt logits?"
:::
::::
```

### 5.3 Blank syntax

`{{ answer | alt1 | alt2 ? hint }}`

- Answers are compared after normalisation: collapse whitespace, strip spaces around operators and commas, unify quotes.
- `re:` prefix marks an alternative as a regex, e.g. `{{ re:^-?1$ | dim=-1 ? ... }}`.
- `? hint` explains **the missing understanding**, not the answer.
- In `derive` scaffolds, blanks hold LaTeX. Compare after stripping spaces and `\left`/`\right`, and give **"Show answer + self-mark"** as a fallback, because LaTeX has too many equivalent forms.
- Rule: **≤ 6 blanks per scaffold**, each testing **one** concept.

### 5.4 Derivation step-reveal

Inside `:::solution` of a `derive` exercise, separate steps with `---step---`. The UI shows one step at a time ("Next step ↓"), so you can derive a step aloud and then check it.

---

## 6. Curriculum (chapters, mapped to the 12-week plan)

UDL chapter numbers are from the current online edition. **Verify each section number against udlbook.github.io during authoring.** Where a topic is not in UDL, the chapter says so and is fully self-contained. UDL also publishes official Python notebooks, so link the relevant one from each chapter as "further practice".

★ = one of the five priority derivations from the guideline (these also go into the Drill Deck, §7).

### Part 0: Orientation
- **00 How to use this workbook:** the tab flow, interview mode, running local tests, the 12-week schedule.

### Part I: Foundations (Weeks 1–3) · UDL 2–9
| # | Chapter | UDL | Key exercises |
|---|---|---|---|
| 01 | Tensors, shapes & broadcasting | App. / — | shape puzzles; broadcasting rules (predict); `einsum` rewrites; batched matmul; masking; `gather`/`scatter`; stable `logsumexp` |
| 02 | Supervised learning & linear models | 2 | least squares closed form vs GD (numpy); 1D sine regression |
| 03 | Shallow → deep networks | 3–4 | MLP forward in numpy; counting linear regions (predict); `nn.Module` MLP |
| 04 | Loss functions from max likelihood | 5 | MSE ⇐ Gaussian (derive); stable softmax-CE (implement); BCE-with-logits; **custom losses**: Huber, focal, label smoothing; ★ **backprop through softmax + CE** (derive) |
| 05 | Backprop & automatic differentiation | 7 | manual backprop for a 2-layer MLP (numpy); micrograd-style scalar autograd engine; computational graphs; custom `torch.autograd.Function` + `gradcheck`; finite-difference checker |
| 06 | Initialisation & gradient flow | 7 | Xavier/He derivation (derive); orthogonal init; activation/gradient stats through 50 layers (predict); vanishing/exploding demo |
| 07 | Optimisation | 6 | SGD, momentum, Adam, AdamW written as `torch.optim.Optimizer` subclasses; warmup + cosine schedule; gradient clipping (norm vs value); saddle & Rosenbrock landscape (predict); curvature/Hessian-vector product |
| 08 | Generalisation & regularisation | 8–9 | ★ **bias–variance decomposition** (derive + polynomial-fit experiment); double descent; L1/L2; **weight decay ≠ L2 under Adam** (predict + derive); dropout (implement, train/eval modes); early stopping |
| 09 | The training loop & debugging | 6–9 | canonical loop (train/eval, `no_grad`, `zero_grad`, seeding, checkpointing); overfit-one-batch; **debug set**: forgot `model.eval()`, softmax before CE, wrong reduction axis, broadcasting bug `(B,) - (B,1)`, in-place op breaking autograd, NaN from `log(0)` |

### Part II: Architectures & representation (Weeks 4–6) · UDL 10–12, 14
| # | Chapter | UDL | Key exercises |
|---|---|---|---|
| 10 | Normalisation & residuals | 11 | BatchNorm forward **and backward** (numpy); LayerNorm, RMSNorm; residual block; why residuals help gradient flow (predict) |
| 11 | CNNs & inductive bias | 10 | conv2d via `unfold`/im2col; pooling; receptive-field calculator; tiny CNN on MNIST; **inductive bias experiment**: CNN vs MLP on pixel-shuffled MNIST (predict) |
| 12 | RNNs, LSTMs, GRUs | — (not in UDL) | RNN cell + BPTT; LSTM and GRU cells from equations; gradient norms over time; char-level model on a tiny text |
| 13 | Attention | 12 | scaled dot-product attention; causal and padding masks; multi-head attention (reshape/transpose shape drill); cross-attention; ★ **why scale by √d** (derive + variance experiment) |
| 14 | Transformers | 12 | pre-LN vs post-LN block; sinusoidal / learned / RoPE positional encodings; tiny decoder-only GPT on a copy/reverse task; encoder-only vs decoder-only vs encoder-decoder (design + mask comparison); KV cache; autoregressive sampling (temperature, top-k) |
| 15 | Mixture of experts | — | top-k router; load-balancing auxiliary loss; capacity/dropped tokens (predict) |
| 16 | Embeddings & autoencoders | 14 | linear AE = PCA (derive + verify); conv AE on MNIST; embedding geometry |

### Part III: Generative & advanced (Weeks 7–9) · UDL 15–19
| # | Chapter | UDL | Key exercises |
|---|---|---|---|
| 17 | VAEs | 17 | ★ **ELBO derivation** (derive: Jensen route **and** KL route); reparameterisation trick; closed-form Gaussian KL; VAE on MNIST |
| 18 | GANs | 15 | 1D/2D GAN on a Gaussian mixture; non-saturating loss (derive); mode collapse (predict); training dynamics plots |
| 19 | Diffusion & score matching | 18 | DDPM forward process closed form (derive); ε-prediction loss; sampler on 2D swiss roll; denoising score matching ↔ ε-prediction (derive) |
| 20 | Autoregressive & flow models | 16 (flows) | masked conv / MADE-style masking; likelihood evaluation; brief normalising flow (change of variables) |
| 21 | Contrastive / self-supervised | — (UDL 14 brief) | InfoNCE loss (implement + derive as a classification problem); SimCLR on toy augmentations; temperature (predict) |
| 22 | Reinforcement learning | 19 | ★ **policy gradient theorem** (derive); REINFORCE on a bandit and a gridworld (numpy); baselines & variance reduction (predict) |
| 23 | Scaling & training dynamics | 20 | fit power laws to synthetic loss curves; LR sensitivity sweep; loss spikes and grad-norm monitoring |

### Part IV: Research methodology & interview (Weeks 10–12)
| # | Chapter | Key exercises |
|---|---|---|
| 24 | Evaluation & statistics | bootstrap CIs; paired tests across seeds; how many seeds?; metrics (AUROC, calibration/ECE, F1); multiple comparisons |
| 25 | Leakage & distribution shift | spot-the-leak (debug: normalising before the split, duplicate items, temporal leakage); **Colored MNIST spurious correlation** (train p=0.9 colour↔label, test flipped): the model fails, you diagnose and mitigate |
| 26 | Ablations & experimental design | design prompts with rubrics: "your method improves +1.2 pts, design the ablations"; confounders; compute-matched baselines |
| 27 | No-framework drills | timed: softmax, CE, conv, attention, LayerNorm, Adam step, k-means, beam search in plain numpy/Python |
| 28 | Mock interview kit | ML theory rapid-fire (timed); paper-critique template; research-walkthrough template; proposal template |

---

## 7. Cross-cutting pages

1. **Home / Dashboard:** a 12-week timeline with chapters slotted into weeks, completion rings per chapter, "continue where you left off", and the derivations due today.
2. **Drill Deck:** every `derive` exercise, plus `:::interview` questions, as cards. Simple spaced repetition: you self-rate Again / Hard / Good / Easy, and due dates are stored in localStorage. ★ priority derivations are pinned at the top.
3. **Checklist:** mirrors the checklist in `brief.md` (see Section 6 for the topics). Each item links to its exercises (from the `tags` metadata) and ticks itself when they are complete. You can also tick items by hand.
4. **Shape & stability cheat sheet:** auto-collected from the shape-contract tables and every exercise tagged `numerical-stability`.
5. **Datasets page:** what each dataset is, how to load it in the browser and locally, plus a thumbnail.

---

## 8. Datasets (`companion/workbook/data.py`)

Synthetic generators are **NumPy-only and seeded**, so the same code runs in Pyodide and locally.

| Name | Shape | Used in |
|---|---|---|
| `sine_regression(n, noise)` | 1D | 02, 08 (bias–variance) |
| `two_moons`, `spirals(n, k)` | 2D classification | 03, 05, 06, 07 |
| `gaussian_mixture_2d(k)` | 2D | 18, 19 |
| `swiss_roll_2d` | 2D | 19 |
| `copy_task`, `reverse_task`, `sort_task` | token sequences | 12, 14 |
| `tiny_text()` | ~20 KB public-domain text, bundled | 12, 14 |
| `bandit(k)`, `gridworld(n)` | RL envs | 22 |
| `mnist(subset=…)` | torchvision, local only | 11, 16, 17 |
| `colored_mnist(p_train=0.9, p_test=0.1)` | 2×28×28, built from MNIST | 21, 25 |

---

## 9. Design ideas

### 9.1 Concept: "Lab notebook ↔ chalkboard"
- **Light theme, "Lab notebook":** warm paper (`#FAF8F3`), ink text (`#1E1E24`), a faint 24px dot grid in page margins only, and one accent colour (deep teal `#0F6E6E`) for interactive elements. ✓ in green (`#2E7D4F`) and ✗ in brick (`#B5452F`), each paired with an icon, never colour alone.
- **Dark theme, "Chalkboard":** slate (`#1F2A2E`), chalk-white text, pale-yellow accent. It ties into the whiteboard practice. Derivation cards could use a subtle chalk texture here.
- **Type:** a serif for prose (*Source Serif 4* or *Newsreader*); a sans for UI (*Inter*); a mono for code (*JetBrains Mono* or *IBM Plex Mono*). Self-host all fonts. KaTeX's default font sits well beside the serif.

### 9.2 Layout
```
┌──────────────┬──────────────────────────────────────┬───────────────┐
│ Chapter nav  │  Prose column (~68ch)                │ Margin notes  │
│ (sticky)     │                                      │ • UDL §5.5 ↗  │
│ Part I       │  ▸ Before you start (prereq, folded) │ • Prereqs     │
│  01 Tensors  │                                      │ • Shape chips │
│  04 Losses ● │  ┌ Exercise card ───────────────────┐│ • Interview   │
│  ...         │  │ ●●○ 15m  browser  UDL 5.5        ││   follow-ups  │
│              │  │ [Blank] [Scaffold] [Solution]    ││               │
│ progress ▓▓░ │  └──────────────────────────────────┘│               │
└──────────────┴──────────────────────────────────────┴───────────────┘
```
- Tufte-style **margin notes** for UDL references, prerequisites, and "interviewers ask…" asides. Below ~1100px they collapse inline.
- **Solution tab:** a two-column split (**code left, explanation right**). Hovering an explanation paragraph highlights its linked lines (`[L3-5]`), and hovering lines highlights the paragraph. This keeps code free of comment clutter.
- **Shape chips:** a shape contract like `logits: (B, C)` renders as small pill chips in the task and in the code gutter. Hovering a chip highlights every variable with that shape.

### 9.3 Exercise card interactions
- **Tabs** remember the last one used per exercise. Keyboard: `1/2/3` switch tabs.
- **Blanks:** inline monospace inputs sized to the expected answer, with an underline style. `Enter` checks; `Tab` moves to the next blank.
  - Correct → ✓ fades in and the input locks green.
  - Wrong → ✗ with a tiny shake, and a **hint pill** slides in below the line: *"What's missing: you must keep the class axis so the subtraction broadcasts."*
  - After 2 wrong tries, a "Reveal" link appears. A revealed blank is marked as revealed, not correct, in progress.
  - A scaffold counter in the card corner: `4/6 ✓`.
- **Blank tab:** a CodeMirror editor prefilled with the stub, a ▶ Run tests button (Pyodide; the first click shows "Loading Python… ~10s"), and test results as a ✓/✗ list. For `local-torch` exercises it shows a copyable command instead: `pytest companion/tests/ch13/test_mha.py`. A "Mark as done" button records completion.
- **Derive tab:** "Next step ↓" reveal, a hidden **"Say it out loud" timer**, and self-rating buttons that feed the Drill Deck.
- **Predict:** you must commit to an answer before the Run/Explanation unlocks.

### 9.4 Interview mode (global toggle)
Hides Scaffold and Solution tabs, starts a per-exercise countdown, shows only prompts and `:::interview` follow-ups, and says "Explain your approach aloud before coding." Ending the session shows a summary. Good for weeks 10–12.

### 9.5 Small delights (cheap, optional)
- Completion ring animation on a chapter's last exercise.
- Little 2D dataset thumbnails (two moons, swiss roll) as chapter header art, generated at build time with matplotlib as SVG.
- A print stylesheet: Blank tabs only, with space to write, for offline whiteboard practice.

### 9.6 Accessibility & performance
Semantic HTML, `aria-selected` tabs, visible focus rings, colour + icon for ✓/✗, `prefers-reduced-motion`, `prefers-color-scheme`. No JS framework is needed; plain ES modules are enough. Pyodide loads lazily, so pages stay light.

---

## 10. Milestones (implement in this order)

> **Status (2026-11-17): M1 done.** `make all` = verify (3/3 exercises) → build → headless browser check (40/40).
> Notes for the next agent:
> - Theme colours come from `~/dopamine-dotfiles/.chezmoidata/palette.toml` (light + dark variants). The gentle green `#AEBA5A` / `#E0E7CD` leads; see `static/css/tokens.css`.
> - JS is a **classic script**, not an ES module, so pages work from `file://`.
> - Maths uses class `.tex`, not `.math`, because `mdit-py-plugins` dollarmath already wraps blocks in `.math`.
> - The browser check uses a cached Chromium + locally extracted libs (no sudo); paths are in `Makefile`.
> - The Blank tab has a plain textarea editor (saved to localStorage) and a "copy code + tests" button. The Pyodide runner, CodeMirror, and companion stub extraction remain for M2.
> - Blank input width = canonical answer length.
>
> **Status (2026-11-17): M2 done.** `make all` = build → verify (7/7 in-process, plus 18 pytest against `WB_IMPL=solutions`) → headless check (67/67).
> - **Runner:** `static/js/runner.js`. Pyodide v314.0.8 runs in a **module** worker built from a Blob. Recent Pyodide refuses classic workers, and classic `importScripts` of the CDN fails in blob workers. From `file://`, Chrome refuses module workers, so the runner falls back to Pyodide on the main thread there (no infinite-loop timeout, and the UI says so). Over http, each run has a 20 s timeout that kills and respawns the worker.
> - **Editor:** CodeMirror 6, vendored as an IIFE bundle in `static/vendor/codemirror/`. Rebuild with `make cm` from `tools/codemirror/entry.js`. The theme uses the CSS variables, so it follows light/dark. Ctrl/Cmd+Enter runs the tests.
> - **Companion:** `build/companion.py`. `exercises/` is written once and never overwritten (it's the user's work). `solutions/` and `tests/` are regenerated and git-ignored. Run `make test` for your own work.
> - Exercises with `runtime=local-torch` show the local workflow panel instead of an editor. `runtime=browser` debug exercises are editable too.
> - Regex blank alternatives (`re:`) are case-insensitive and are matched against the *normalised* input (whitespace removed).
> - An all-green run marks the exercise complete automatically.
> - **Sidebar:** collapsible on wide screens (button or `[`), remembered in `localStorage` as `wb:nav`; still a drawer on mobile.
>
> **Status (2026-11-18): M3 done.** Part I is complete: chapters 01–09 (plus 13), 32 exercises. `make all` gives 32/32 verified, 77 pytest, and 88/88 headless checks.
> - **New `predict` type:** sections `task`, `choices` (`- [x]`/`- [ ]`), `experiment` (editable and runnable), `explain`, plus optional `tests`, which act as "claims" the experiment must satisfy. The Experiment and Explanation tabs stay locked until the user commits or skips.
> - **Dashboard (`index.html`):** progress stats, "this week" from a user-set start date (chapter `weeks` now live in `_meta.yaml`), and derivations due. The spacing is again → 1 day, hard → 3, good → 7, easy → 14. The full Drill Deck is still M4.
> - **Checklist (`checklist.html`):** groups and items come from `content/_checklist.yaml`. An item auto-ticks when every exercise sharing one of its tags is done; it can also be ticked by hand. Exercise data is embedded on every page as JSON (`#wb-data`).
> - **Verifier additions:** `[La-b]` explain refs must point at real, non-blank solution lines. Regex alternatives must not accept empty input, but no longer need to match the canonical answer. Predict experiments must run, and their claims must hold.
> - **Browser check additions:** every chapter is rendered, and *every* in-browser reference solution and experiment is run in Pyodide. Anything near the 20 s runner limit is flagged (Pyodide's NumPy has no fast BLAS, so keep experiments small: the slowest is about 4 s).
> - **Authoring gotchas:** a blank closes at the first `}}`, so LaTeX alternatives ending in `}}` must become regexes. Exercise titles may use `code`: `title_html` is rendered, and `title` is plain text.
> - **Datasets:** these are still inline and seeded per exercise. `companion/workbook/data.py` (Section 8) hasn't been needed yet; build it when Part II needs MNIST.
>
> **Status (2026-11-19): M4 done.** Part II is complete: chapters 10–16, 51 exercises in total. `make all` gives 51/51 verified, 113 pytest, 102/102 headless checks, and 156/156 figure checks.
> - **Drill Deck (`drills.html`):** cards come from every `derive` exercise (its task as the front, its steps as the back) and every `:::interview` bullet. A bullet ending in `*(…)*` gets that text as its answer. There are 174 cards, 81 of them interview questions with answers. ★ = the derive exercises tagged with one of the five priority-derivation tags. Derivation ratings are shared with the exercise page (`store.ex[id].rating`); interview cards use `store.cards[cardId]`. The sidebar counts derivations due, the same everywhere.
> - **Figures:** `@figure(name.svg) "caption"` on its own line in any Markdown inlines `content/figures/name.svg`, so it can use the theme's CSS variables. Styles and behaviour live in `static/css/figures.css` and `static/js/figures.js`. The broadcasting figures (Ch01 and the Ch04 prereq) are generated by `tools/figures/make_broadcast.py` (`make figures`) and checked by `build/check_figures.py` (`make check-figures`). These were built by a parallel agent.
> - **`WB_DIST`** overrides the output directory, so parallel builds don't clash.
> - **Datasets:** Part II still uses small synthetic data generated in the tests (so CI needs no downloads). MNIST and Colored MNIST belong with Part III/IV (Ch 21, 25) via `companion/workbook/data.py`.
> - **Test-design lessons:** a perturbation that adds a constant to every feature is invisible after LayerNorm, so use random perturbations. A shuffled loader changes floating-point summation order, so compare losses approximately.
>
> **Status (2026-11-19): M5 done.** Part III is complete: chapters 17–23, 72 exercises in total. `make all` gives 72/72 verified, 150 pytest (1 download test skipped by default), 103/103 headless checks, and 156/156 figure checks. The deck holds 243 cards, including all five ★ derivations.
> - **Known-answer tests for generative models:** the DDPM sampler is tested with the analytic optimal denoiser for Gaussian data (it must reproduce N(μ, s²)). The ELBO derivation is checked on a conjugate Gaussian model, where log p(x), the posterior and the gap are all exact. The REINFORCE estimate is checked against the analytic softmax-bandit gradient.
> - **`companion/workbook/data.py`:** `mnist(split)` downloads from the PyTorch S3 mirror into `~/.cache/interview-workbook` (override with `WB_DATA`). `colored_mnist(...)` / `colored_mnist_envs()` follow Arjovsky et al.: label = digit<5 with 25% noise; colour = label flipped with p = 0.1 / 0.2 (train) and 0.9 (test). Its tests are offline (synthetic digits); `WB_DOWNLOAD=1` also tests the real download. pytest `testpaths` now includes `companion/workbook`. Chapter 25 (distribution shift) is where it gets used.
> - **Gotchas:** YAML front matter with `: ` in the summary must be quoted (the parser now says so). NumPy 2 removed `np.trapz`. A precedence bug (`x ** a / b`) was caught by the alternatives check, which is exactly what that check is for.
> - **Next, M6 (Part IV + polish):** chapters 24–28 (evaluation and statistics, leakage and shift *using Colored MNIST*, ablations and design with `design`-type exercises, no-framework drills, mock interview kit), interview mode, the cheat sheet, and the print stylesheet.
>
> **Status (2026-11-19): M6 done. All planned milestones are complete.** 28 chapters, 87 exercises. `make all` gives 87/87 verified, 170 pytest, 120 headless checks, and 156/156 figure checks.
> - **New `design` type:** sections `task`, `rubric` (3+ `- ` items) and `answer`. Its layers are Prompt (timer and saved notes), Checklist (self-assessed rubric ticks, saved) and Model answer. These are verified at parse time only, since there's nothing to execute.
> - **Interview mode** (topbar toggle; `html.interview-mode`, persisted as `wb:interview`): hides layers 2–3, prerequisites, hints and the answers in the interview notes. It shows a banner and a per-exercise clock chip. It's pure CSS plus a small amount of JS.
> - **Cheat sheet** (`cheatsheet.html`): built from task tables with Shape/Inputs/Argument/Output headers, from starter docstrings (function signatures with shapes), and from the bold lead-ins of the explanations of `numerical-stability` exercises.
> - **Print** (`static/css/print.css`): Blank layers only, one exercise per page, with a write-in box. `beforeprint` opens the prerequisite `<details>` and `afterprint` restores them.
> - **Lesson from Ch24:** a shared baseline makes the multiple-comparison tests correlated (about 42% rather than 64% false-positive rate). The experiment caught a wrong claim in the draft explanation.
> - **Ideas beyond the plan:** a real-MNIST CNN exercise for Colored MNIST (local-torch), equal-mass ECE, a `make pdf` target for printable chapter packs, and syncing progress across devices (export/import JSON from the plan's progress section).

1. **M1: Pipeline vertical slice.** `build.py`, directive parser, templates, CSS tokens, tabs, blanks with ✓/✗/hint, KaTeX, progress in localStorage. Content: **chapter 04 only** (softmax-CE implement + ★ softmax-CE derive + one debug exercise). `make verify` passes.
2. **M2: Runners.** Pyodide runner for the Blank tab; generated companion stubs and tests; `local-torch` flow. Add chapter 13 (attention, MHA, ★ √d), which exercises both runtimes.
3. **M3: Part I complete** (01–09) plus the Home dashboard and Checklist page.
4. **M4: Part II** plus the Drill Deck with spaced repetition.
5. **M5: Part III**, including the Colored MNIST loader.
6. **M6: Part IV**, interview mode, cheat sheet, print stylesheet, polish.

## 11. Acceptance criteria per exercise
- [ ] Has the `id`, `type`, `difficulty`, `time`, `runtime`, `udl` (or `udl="—"` plus a note), and `tags` attributes
- [ ] `prereq` makes the exercise solvable without opening UDL
- [ ] Tests pass on the solution, and on the scaffold with canonical answers filled in (`make verify`)
- [ ] The scaffold has ≤ 6 blanks, each with a *conceptual* hint
- [ ] The explanation is linked to line ranges; the solution has ≤ 1 comment per ~5 lines
- [ ] Has at least one `:::interview` follow-up question
- [ ] Shapes are stated for every tensor input and output

## 12. Open questions for the user (defaults are in bold)
- Track the companion with git? **Yes; run `git init` at M1.**
- Pyodide from CDN or vendored? **CDN, with an offline note**
- Include Goodfellow cross-references alongside UDL? **No for v1; add a `goodfellow=` attribute later**
