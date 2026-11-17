"""Fill-in blanks: `{{ answer | alt | re:regex ? conceptual hint }}`  (re: alternatives are case-insensitive
 and are matched against the *normalised* input, i.e. with whitespace removed).

Blanks are swapped for sentinels before Markdown/Pygments run (so neither mangles them),
then the sentinels are replaced with <input> elements.
"""

from __future__ import annotations

import base64
import html
import json
import re
from dataclasses import dataclass

BLANK_RE = re.compile(r"\{\{(.+?)\}\}")
HINT_SPLIT = re.compile(r"\s\?\s")
ALT_SPLIT = re.compile(r"\s\|\s")


@dataclass
class Blank:
    index: int          # 1-based within the exercise
    answers: list[str]  # first is canonical
    hint: str           # markdown

    @property
    def canonical(self) -> str:
        return self.answers[0]

    @property
    def literal_answers(self) -> list[str]:
        return [a for a in self.answers if not a.startswith("re:")]


def parse_blank(inner: str, index: int) -> Blank:
    parts = HINT_SPLIT.split(" " + inner.strip() + " ", maxsplit=1)
    ans_part = parts[0].strip()
    hint = parts[1].strip() if len(parts) > 1 else ""
    answers = [a.strip() for a in ALT_SPLIT.split(ans_part) if a.strip()]
    if not answers or answers[0].startswith("re:"):
        raise ValueError(f"blank {index}: first answer must be a literal (canonical) answer: {inner!r}")
    return Blank(index, answers, hint)


def normalise(s: str) -> str:
    """Must stay identical to normalise() in static/js/app.js."""
    s = re.sub(r"\\left|\\right", "", s.strip())
    s = re.sub(r"\s+", "", s)
    return s.replace("'", '"')


def extract(text: str, start: int, sentinel: str) -> tuple[str, list[Blank]]:
    """Replace blanks with sentinel.format(i=...) and return them. `start` = next index."""
    blanks: list[Blank] = []

    def sub(m):
        b = parse_blank(m.group(1), start + len(blanks))
        blanks.append(b)
        return sentinel.format(i=b.index)

    return BLANK_RE.sub(sub, text), blanks


def fill(text: str, choose=lambda b: b.canonical, start: int = 1) -> str:
    """Substitute answers back in (used by verify)."""
    counter = [start]

    def sub(m):
        b = parse_blank(m.group(1), counter[0])
        counter[0] += 1
        return choose(b)

    return BLANK_RE.sub(sub, text)


def b64(obj) -> str:
    return base64.b64encode(json.dumps(obj).encode()).decode()


def input_html(b: Blank, ex_id: str, hint_html: str, code: bool) -> str:
    width = max(4, len(b.canonical) + 1)  # canonical length: long alternatives must not bloat the input
    cls = "blank blank-code" if code else "blank blank-prose"
    return (
        f'<span class="blank-wrap" data-blank="{b.index}">'
        f'<input class="{cls}" type="text" spellcheck="false" autocomplete="off" autocapitalize="off" '
        f'style="--w:{width}ch" aria-label="Blank {b.index}" placeholder="{b.index}" '
        f'data-ex="{html.escape(ex_id)}" data-i="{b.index}" '
        f'data-a="{b64(b.answers)}" data-h="{b64(hint_html)}">'
        f'<span class="blank-mark" aria-hidden="true"></span></span>'
    )
