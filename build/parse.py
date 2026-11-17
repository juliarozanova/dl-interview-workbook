"""Parse chapter Markdown (front matter + ::::exercise / :::section directives) into a model.

Grammar (line-based, deliberately simple):

    ::::exercise{key=value key="value with spaces"}
    # Exercise title
    :::section-name{attrs}
    ...raw body...
    :::
    ::::

Text outside exercises is ordinary Markdown.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

import yaml

EX_OPEN = re.compile(r"^::::exercise(\{.*\})?\s*$")
EX_CLOSE = re.compile(r"^::::\s*$")
SEC_OPEN = re.compile(r"^:::([a-z][a-z0-9-]*)(\{.*\})?\s*$")
SEC_CLOSE = re.compile(r"^:::\s*$")

EXERCISE_TYPES = {"implement", "derive", "debug"}
KNOWN_SECTIONS = {
    "prereq", "theory", "task", "starter", "scaffold", "solution", "derivation",
    "explain", "tests", "hints", "interview",
}
REQUIRED_ATTRS = {"id", "type", "difficulty", "time", "runtime", "udl"}


class ParseError(Exception):
    pass


@dataclass
class Section:
    name: str
    attrs: dict
    body: str

    @property
    def is_code(self) -> bool:
        return "lang" in self.attrs


@dataclass
class Exercise:
    attrs: dict
    title: str
    sections: dict[str, Section] = field(default_factory=dict)
    source: str = ""

    @property
    def id(self) -> str:
        return self.attrs["id"]

    @property
    def type(self) -> str:
        return self.attrs["type"]

    def get(self, name: str) -> Section | None:
        return self.sections.get(name)


@dataclass
class Chapter:
    meta: dict
    blocks: list  # list of ("md", str) | ("exercise", Exercise)
    path: Path

    @property
    def exercises(self) -> list[Exercise]:
        return [b for kind, b in self.blocks if kind == "exercise"]


def parse_attrs(text: str | None) -> dict:
    if not text:
        return {}
    inner = text.strip()[1:-1]
    out = {}
    for tok in shlex.split(inner):
        if "=" not in tok:
            out[tok] = True
            continue
        k, v = tok.split("=", 1)
        out[k] = v
    return out


def split_front_matter(text: str) -> tuple[dict, str]:
    if text.startswith("---\n"):
        end = text.index("\n---\n", 4)
        return yaml.safe_load(text[4:end]) or {}, text[end + 5:]
    return {}, text


def parse_chapter(path: Path) -> Chapter:
    meta, body = split_front_matter(path.read_text())
    lines = body.split("\n")
    blocks: list = []
    md_buf: list[str] = []
    i = 0

    def where(n):
        return f"{path}:{n + 1}"

    while i < len(lines):
        m = EX_OPEN.match(lines[i])
        if not m:
            md_buf.append(lines[i])
            i += 1
            continue
        if md_buf and "".join(md_buf).strip():
            blocks.append(("md", "\n".join(md_buf)))
        md_buf = []
        start = i
        attrs = parse_attrs(m.group(1))
        i += 1
        title = ""
        sections: dict[str, Section] = {}
        while True:
            if i >= len(lines):
                raise ParseError(f"{where(start)}: unclosed ::::exercise")
            line = lines[i]
            if EX_CLOSE.match(line):
                i += 1
                break
            sm = SEC_OPEN.match(line)
            if sm:
                name = sm.group(1)
                if name not in KNOWN_SECTIONS:
                    raise ParseError(f"{where(i)}: unknown section ':::{name}'")
                sattrs = parse_attrs(sm.group(2))
                i += 1
                buf = []
                while True:
                    if i >= len(lines):
                        raise ParseError(f"{where(i)}: unclosed :::{name}")
                    if SEC_CLOSE.match(lines[i]):
                        i += 1
                        break
                    if EX_CLOSE.match(lines[i]) or EX_OPEN.match(lines[i]):
                        raise ParseError(f"{where(i)}: :::{name} not closed before exercise end")
                    buf.append(lines[i])
                    i += 1
                if name in sections:
                    raise ParseError(f"{where(i)}: duplicate :::{name}")
                sections[name] = Section(name, sattrs, "\n".join(buf).strip("\n"))
                continue
            if line.startswith("# ") and not title:
                title = line[2:].strip()
            elif line.strip():
                raise ParseError(f"{where(i)}: stray text inside exercise (put it in a section): {line!r}")
            i += 1
        ex = Exercise(attrs, title, sections, source=where(start))
        validate(ex)
        blocks.append(("exercise", ex))
    if md_buf and "".join(md_buf).strip():
        blocks.append(("md", "\n".join(md_buf)))
    return Chapter(meta, blocks, path)


def validate(ex: Exercise) -> None:
    missing = REQUIRED_ATTRS - ex.attrs.keys()
    if missing:
        raise ParseError(f"{ex.source}: missing attrs {sorted(missing)}")
    if ex.type not in EXERCISE_TYPES:
        raise ParseError(f"{ex.source}: unknown type {ex.type!r}")
    if not ex.title:
        raise ParseError(f"{ex.source}: exercise needs a '# Title' line")
    need = {"task", "scaffold"}
    need |= {"derivation"} if ex.type == "derive" else {"starter", "solution", "explain", "tests"}
    absent = need - ex.sections.keys()
    if absent:
        raise ParseError(f"{ex.source} ({ex.id}): missing sections {sorted(absent)}")
