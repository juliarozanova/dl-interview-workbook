"""Turn parsed exercises into HTML fragments for the Jinja templates."""

from __future__ import annotations

import html
import re

from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

import blanks as B
from parse import Exercise


def _math(content: str, opts: dict) -> str:
    tag = "div" if opts.get("display_mode") else "span"
    return f'<{tag} class="tex" data-display="{int(bool(opts.get("display_mode")))}">{html.escape(content)}</{tag}>'


md = (
    MarkdownIt("commonmark", {"html": True, "typographer": True})
    .enable(["table", "replacements", "smartquotes"])
    .use(dollarmath_plugin, renderer=_math, allow_space=True, double_inline=True)
)

PROSE_SENTINEL = "XBLANKX{i}X"
CODE_SENTINEL = "__BLANK_{i}__"


def markdown(text: str) -> str:
    return md.render(text)


def inline(text: str) -> str:
    return md.renderInline(text)


def code(source: str, lang: str = "python", prefix: str = "c") -> str:
    """Highlight with one <span class="ln" data-line=N> per line."""
    fmt = HtmlFormatter(linespans=prefix)  # linespans is ignored when nowrap=True
    out = highlight(source, get_lexer_by_name(lang), fmt)
    out = re.search(r"<pre>(?:<span></span>)?(.*)</pre>", out, re.S).group(1)
    out = re.sub(rf'<span id="{re.escape(prefix)}-(\d+)">', r'<span class="ln" data-line="\1">', out)
    return f'<pre class="code"><code class="language-{lang}">{out.rstrip()}</code></pre>'


def _blank_inputs(text: str, blanks: list[B.Blank], ex_id: str, code_mode: bool) -> str:
    for b in blanks:
        widget = B.input_html(b, ex_id, inline(b.hint) if b.hint else "", code_mode)
        token = (CODE_SENTINEL if code_mode else PROSE_SENTINEL).format(i=b.index)
        text, n = re.subn(rf'<span class="[^"]*">{re.escape(token)}</span>', widget, text)
        if not n:
            text, n = re.subn(re.escape(token), widget, text)
        if n != 1:
            raise ValueError(f"{ex_id}: blank {b.index} placed {n} times after rendering")
    return text


def scaffold(ex: Exercise) -> tuple[str, list[B.Blank]]:
    sec = ex.get("scaffold")
    if sec.is_code:
        src, bl = B.extract(sec.body, 1, CODE_SENTINEL)
        out = code(src, sec.attrs["lang"], f"{ex.id}-scaffold")
        return _blank_inputs(out, bl, ex.id, True), bl
    src, bl = B.extract(sec.body, 1, PROSE_SENTINEL)
    return _blank_inputs(markdown(src), bl, ex.id, False), bl


LINE_REF = re.compile(r"<p>\[L(\d+)(?:[-–](\d+))?\]\s*")


def explain(text: str) -> str:
    def sub(m):
        a, b = m.group(1), m.group(2) or m.group(1)
        label = f"L{a}" if a == b else f"L{a}–{b}"
        return f'<p class="explain-para" data-lines="{a}-{b}" tabindex="0"><span class="line-ref">{label}</span> '

    return LINE_REF.sub(sub, markdown(text))


def hints(text: str) -> list[str]:
    items = re.split(r"^\d+\.\s+", text.strip(), flags=re.M)
    return [inline(i.strip()) for i in items if i.strip()]


TAB_LABELS = {
    "implement": ("Blank", "Scaffold", "Solution"),
    "derive": ("Prompt", "Scaffold", "Derivation"),
    "debug": ("Spot the bug", "Scaffold", "Fix"),
}

RUNTIME_LABELS = {"browser": "NumPy · in browser", "local-torch": "PyTorch · local", "none": "pen & paper"}


def exercise(ex: Exercise, number: str) -> dict:
    g = ex.get
    vm = {
        "id": ex.id,
        "type": ex.type,
        "title": ex.title,
        "number": number,
        "difficulty": int(ex.attrs["difficulty"]),
        "time": ex.attrs["time"],
        "runtime": ex.attrs["runtime"],
        "runtime_label": RUNTIME_LABELS.get(ex.attrs["runtime"], ex.attrs["runtime"]),
        "udl": ex.attrs["udl"],
        "tags": [t for t in ex.attrs.get("tags", "").split(",") if t],
        "tabs": TAB_LABELS[ex.type],
        "prereq": markdown(g("prereq").body) if g("prereq") else "",
        "theory": markdown(g("theory").body) if g("theory") else "",
        "task": markdown(g("task").body),
        "hints": hints(g("hints").body) if g("hints") else [],
        "interview": markdown(g("interview").body) if g("interview") else "",
    }
    vm["scaffold"], bl = scaffold(ex)
    vm["scaffold_is_code"] = g("scaffold").is_code
    vm["n_blanks"] = len(bl)
    if g("starter"):
        s = g("starter")
        vm["starter_raw"] = s.body
        vm["starter"] = code(s.body, s.attrs.get("lang", "python"), f"{ex.id}-starter")
    if g("tests"):
        vm["tests_raw"] = g("tests").body
        vm["tests"] = code(g("tests").body, "python", f"{ex.id}-tests")
    if g("solution"):
        vm["solution"] = code(g("solution").body, g("solution").attrs.get("lang", "python"), f"{ex.id}-solution")
    if g("explain"):
        vm["explain"] = explain(g("explain").body)
    if g("derivation"):
        steps = re.split(r"^---step---\s*$", g("derivation").body, flags=re.M)
        vm["steps"] = [markdown(s.strip()) for s in steps if s.strip()]
    return vm
