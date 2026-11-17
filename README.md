# Deep-learning interview workbook

An implementation-first practice workbook for ML research interviews, built as a static site. It's paced over twelve weeks alongside Simon Prince's [*Understanding Deep Learning*](https://udlbook.github.io/udlbook/).

**Live site:** https://juliarozanova.github.io/dl-interview-workbook/

Every exercise has three layers:

1. **Blank:** write it yourself. NumPy exercises run their tests in the browser (Pyodide); PyTorch ones run locally with `pytest`.
2. **Scaffold:** a partly worked solution with inline blanks. Each blank gives an instant ✓/✗, and a wrong answer gets a hint about the missing idea.
3. **Solution:** the reference code beside a line-linked explanation.

Derivation exercises (e.g. backprop through softmax + cross-entropy, why attention is scaled by √d) reveal one step at a time, for practising out loud. Progress is stored in your browser's `localStorage`.

## Working locally

```sh
make setup    # .venv with build deps (add torch yourself for the PyTorch exercises)
make serve    # build and open http://localhost:8000
make test     # run your own answers in companion/exercises against the tests
make verify   # check every exercise's solution, scaffold answers, and starter
make check    # headless-browser smoke test (Playwright + Chromium)
```

Content lives in `content/` as Markdown with `::::exercise` directives; see [PLAN.md](PLAN.md) for the format and the roadmap. `companion/exercises/` is your private workspace: it's created on first build, never overwritten, and git-ignored.
