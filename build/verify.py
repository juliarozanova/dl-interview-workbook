"""Build gate: every exercise must be internally consistent.

For each exercise with tests:
  * the solution passes all tests;
  * the scaffold passes with the canonical answers filled in, and with every literal alternative;
  * the starter FAILS (otherwise the tests don't test anything).
For every blank: answers are non-empty after normalisation, regex alternatives compile and
accept the canonical answer, and each hint exists.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import blanks as B  # noqa: E402
from build import CONTENT, load_curriculum  # noqa: E402
from parse import parse_chapter  # noqa: E402


def run(code: str, tests: str) -> tuple[bool, str]:
    ns: dict = {"__name__": "__exercise__"}
    with contextlib.redirect_stdout(io.StringIO()):  # experiments print; keep the report clean
        try:
            exec(compile(code, "<impl>", "exec"), ns)
            exec(compile(tests, "<tests>", "exec"), ns)
        except Exception:
            return False, traceback.format_exc(limit=1)
        for name, fn in list(ns.items()):
            if name.startswith("test_") and callable(fn):
                try:
                    fn()
                except Exception as e:  # noqa: BLE001
                    return False, f"{name}: {type(e).__name__}: {e}"
    return True, ""


def check_blanks(ex) -> list[str]:
    errs = []
    _, bl = B.extract(ex.get("scaffold").body, 1, "S{i}")
    if not bl:
        errs.append("scaffold has no blanks")
    if len(bl) > 6:
        errs.append(f"scaffold has {len(bl)} blanks (max 6)")
    for b in bl:
        if not b.hint:
            errs.append(f"blank {b.index}: missing hint")
        for a in b.answers:
            if a.startswith("re:"):
                try:
                    rx = re.compile(a[3:], re.I)  # regex alternatives are case-insensitive (JS uses the "i" flag)
                except re.error as e:
                    errs.append(f"blank {b.index}: bad regex {a!r}: {e}")
                    continue
                if rx.search(""):
                    errs.append(f"blank {b.index}: regex {a!r} accepts an empty answer")
            elif not B.normalise(a):
                errs.append(f"blank {b.index}: empty answer")
    return errs


def check_line_refs(ex) -> list[str]:
    """`[La-b]` notes in :::explain must point at real, non-blank lines of the solution code."""
    sol, expl = ex.get("solution"), ex.get("explain")
    if not (sol and sol.is_code and expl):
        return []
    lines = sol.body.split("\n")
    errs = []
    for m in re.finditer(r"^\[L(\d+)(?:[-–](\d+))?\]", expl.body, flags=re.M):
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        if not 1 <= a <= b <= len(lines):
            errs.append(f"explain ref [L{a}-{b}] is outside the solution (1–{len(lines)})")
        elif not lines[a - 1].strip() or not lines[b - 1].strip():
            errs.append(f"explain ref [L{a}-{b}] starts or ends on a blank line: off by one?")
    return errs


def check_exercise(ex) -> list[str]:
    return check_line_refs(ex) + _check_exercise(ex)


def _check_exercise(ex) -> list[str]:
    if ex.type == "predict":  # the experiment must run, and the claims in its tests must hold
        tests = ex.get("tests").body if ex.get("tests") else ""
        ok, msg = run(ex.get("experiment").body, tests)
        return [] if ok else [f"experiment fails: {msg}"]
    errs = check_blanks(ex)
    tests = ex.get("tests")
    if not tests:
        return errs
    sol = ex.get("solution")
    ok, msg = run(sol.body, tests.body)
    if not ok:
        errs.append(f"solution fails: {msg}")
    scaf = ex.get("scaffold")
    if scaf.is_code:
        ok, msg = run(B.fill(scaf.body), tests.body)
        if not ok:
            errs.append(f"scaffold (canonical answers) fails: {msg}")
        _, bl = B.extract(scaf.body, 1, "S{i}")
        for target in bl:
            for alt in target.literal_answers[1:]:
                filled = B.fill(scaf.body, lambda b, t=target, a=alt: a if b.index == t.index else b.canonical)
                ok, msg = run(filled, tests.body)
                if not ok:
                    errs.append(f"blank {target.index} alternative {alt!r} fails: {msg}")
    if ex.get("starter") and run(ex.get("starter").body, tests.body)[0]:
        errs.append("starter passes the tests — tests are too weak")
    return errs


def main() -> int:
    _, chapters = load_curriculum()
    failed = total = 0
    for ch in chapters:
        if not ch.get("file"):
            continue
        for ex in parse_chapter(CONTENT / ch["file"]).exercises:
            total += 1
            errs = check_exercise(ex)
            mark = "✓" if not errs else "✗"
            print(f"  {mark} {ch['num']} {ex.id}")
            for e in errs:
                print(f"      - {e}")
            failed += bool(errs)
    print(f"\n{total - failed}/{total} exercises verified")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
