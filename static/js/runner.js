// In-browser Python test runner: Pyodide inside a *module* Web Worker built from a Blob, so it also works from
// file://. Recent Pyodide refuses classic workers, and classic importScripts of the CDN fails in blob workers.
// API: WBRunner.run(code, tests, onStatus) -> Promise<{tests: [{name, ok, ms, msg, where, line}], error, stdout}>
"use strict";
(() => {
  const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v314.0.8/full/";
  const BOOT_TIMEOUT_MS = 180_000;
  const RUN_TIMEOUT_MS = 20_000;

  // Python harness. Kept free of backslashes and `${` so it can live in a template literal.
  const HARNESS = String.raw`
import ast, contextlib, io, time, traceback

def _wb_describe(e, sources):
    msg = type(e).__name__ + (": " + str(e) if str(e) else "")
    frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename in sources]
    if isinstance(e, SyntaxError) and e.filename in sources:
        return {"msg": msg, "where": e.filename.strip("<>") + ", line " + str(e.lineno),
                "line": (e.text or "").strip()}
    if not frames:
        return {"msg": msg, "where": "", "line": ""}
    f = frames[-1]
    lines = sources[f.filename]
    text = lines[f.lineno - 1].strip() if 0 < f.lineno <= len(lines) else ""
    return {"msg": msg, "where": f.filename.strip("<>") + ", line " + str(f.lineno), "line": text}

def _wb_run(code, tests):
    sources = {"<your code>": code.splitlines(), "<tests>": tests.splitlines()}
    out = io.StringIO()
    ns = {"__name__": "__exercise__"}
    result = {"tests": [], "error": None, "stdout": ""}
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            exec(compile(code, "<your code>", "exec"), ns)
        except BaseException as e:
            result["error"] = _wb_describe(e, sources)
        if result["error"] is None:
            try:
                exec(compile(tests, "<tests>", "exec"), ns)
            except BaseException as e:
                result["error"] = _wb_describe(e, sources)
        if result["error"] is None:
            names = [n.name for n in ast.parse(tests).body
                     if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
            for name in names:
                t0 = time.perf_counter()
                try:
                    ns[name]()
                    row = {"ok": True, "msg": "", "where": "", "line": ""}
                except BaseException as e:
                    row = dict(_wb_describe(e, sources), ok=False)
                row.update(name=name, ms=round((time.perf_counter() - t0) * 1000, 1))
                result["tests"].append(row)
    result["stdout"] = out.getvalue()[-4000:]
    return result
`;

  const WORKER_SRC = `
    import { loadPyodide } from ${JSON.stringify(PYODIDE + "pyodide.mjs")};
    let py;
    const ready = (async () => {
      py = await loadPyodide({ indexURL: ${JSON.stringify(PYODIDE)} });
      await py.loadPackage("numpy");
      py.runPython(${JSON.stringify(HARNESS)});
      postMessage({ type: "booted" });
    })();
    ready.catch((e) => postMessage({ type: "boot-error", msg: String(e) }));
    onmessage = async (e) => {
      const { id, code, tests } = e.data;
      try {
        await ready;
        postMessage({ id, type: "status", msg: "Running tests…" });
        await py.loadPackagesFromImports(code + "\\n" + tests);
        const fn = py.globals.get("_wb_run");
        const proxy = fn(code, tests);
        const res = proxy.toJs({ dict_converter: Object.fromEntries });
        proxy.destroy(); fn.destroy();
        postMessage({ id, type: "result", res });
      } catch (err) {
        postMessage({ id, type: "result", res: { tests: [], stdout: "", error: { msg: String(err), where: "", line: "" } } });
      }
    };
  `;

  let worker = null, booted = false, bootPromise = null, seq = 0, queue = Promise.resolve();
  const pending = new Map();

  function spawn() {
    const url = URL.createObjectURL(new Blob([WORKER_SRC], { type: "text/javascript" }));
    worker = new Worker(url, { type: "module" });
    booted = false;
    bootPromise = new Promise((resolve, reject) => {
      const t = setTimeout(() => reject(new Error("Python took too long to load. Check your connection.")), BOOT_TIMEOUT_MS);
      worker.addEventListener("error", (e) => {
        clearTimeout(t);
        reject(new Error("Could not start Python" + (e.message ? ": " + e.message : ". Are you offline?")));
      });
      worker.addEventListener("message", function onBoot(e) {
        if (e.data.type === "booted") { clearTimeout(t); booted = true; resolve(); worker.removeEventListener("message", onBoot); }
        if (e.data.type === "boot-error") { clearTimeout(t); reject(new Error("Could not load Python: " + e.data.msg)); }
      });
    });
    worker.addEventListener("message", (e) => {
      const p = pending.get(e.data.id);
      if (!p) return;
      if (e.data.type === "status") p.onStatus?.(e.data.msg);
      if (e.data.type === "result") { pending.delete(e.data.id); p.resolve(e.data.res); }
    });
    worker.addEventListener("error", (e) => {
      for (const [id, p] of pending) { pending.delete(id); p.reject(new Error(e.message || "Worker crashed")); }
    });
  }

  function kill() {
    worker?.terminate();
    worker = null;
    booted = false;
  }

  async function runOnce(code, tests, onStatus) {
    if (!worker) spawn();
    if (!booted) {
      onStatus?.("Loading Python + NumPy (first run only, ~10 s)…");
      try { await bootPromise; } catch (e) { kill(); throw e; }
    }
    const id = ++seq;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        pending.delete(id);
        kill(); // the only way to stop a runaway loop; the next run reloads Python
        reject(new Error(`Timed out after ${RUN_TIMEOUT_MS / 1000} s. Is there an infinite loop?`));
      }, RUN_TIMEOUT_MS);
      pending.set(id, {
        onStatus,
        resolve: (r) => { clearTimeout(timer); resolve(r); },
        reject: (e) => { clearTimeout(timer); reject(e); },
      });
      worker.postMessage({ id, code, tests });
    });
  }

  // ── Fallback: Pyodide on the page itself. Used from file://, where Chrome refuses module workers.
  // No timeout protection here (a runaway loop would freeze the tab), hence worker-first over http.
  let mainPy = null;
  function bootMain() {
    mainPy ??= (async () => {
      await new Promise((resolve, reject) => {
        const s = document.createElement("script");
        s.src = PYODIDE + "pyodide.js";
        s.onload = resolve;
        s.onerror = () => reject(new Error("Could not load Python. Are you offline?"));
        document.head.append(s);
      });
      const py = await loadPyodide({ indexURL: PYODIDE });
      await py.loadPackage("numpy");
      py.runPython(HARNESS);
      return py;
    })();
    mainPy.catch(() => { mainPy = null; });
    return mainPy;
  }

  async function runMain(code, tests, onStatus) {
    onStatus?.("Loading Python + NumPy (first run only, ~10 s)…");
    const py = await bootMain();
    onStatus?.("Running tests…");
    await new Promise((r) => setTimeout(r, 30)); // let the status paint before Python blocks the thread
    await py.loadPackagesFromImports(code + "\n" + tests);
    const fn = py.globals.get("_wb_run");
    const proxy = fn(code, tests);
    const res = proxy.toJs({ dict_converter: Object.fromEntries });
    proxy.destroy(); fn.destroy();
    return res;
  }

  let mode = location.protocol === "file:" ? "main" : "worker";

  // Runs are serialised: one Python interpreter, one test suite at a time.
  function run(code, tests, onStatus) {
    const once = async () => {
      if (mode === "main") return runMain(code, tests, onStatus);
      try {
        return await runOnce(code, tests, onStatus);
      } catch (e) {
        if (!booted && /Could not start Python/.test(e.message)) { mode = "main"; return runMain(code, tests, onStatus); }
        throw e;
      }
    };
    const p = queue.then(once);
    queue = p.catch(() => {});
    return p;
  }

  window.WBRunner = { run, get mode() { return mode; } };
})();
