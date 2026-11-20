"""Build the workbook: content/*.md → dist/*.html (+ companion/).  Usage: python build/build.py"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

sys.path.insert(0, str(Path(__file__).parent))
import companion  # noqa: E402
import render  # noqa: E402
from parse import parse_chapter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "content"
DIST = Path(os.environ["WB_DIST"]).resolve() if os.environ.get("WB_DIST") else ROOT / "dist"  # override for parallel builds


def load_curriculum():
    meta = yaml.safe_load((CONTENT / "_meta.yaml").read_text())
    chapters = []
    for part in meta["parts"]:
        for ch in part["chapters"]:
            ch["part"] = part
            ch["href"] = f"{ch['slug']}.html" if ch.get("file") else None
            ch.setdefault("weeks", [])
            chapters.append(ch)
    return meta, chapters


def load_checklist(manifest: list[dict]) -> list[dict]:
    groups = yaml.safe_load((CONTENT / "_checklist.yaml").read_text())["groups"]
    for g in groups:
        for item in g["items"]:
            item["exercises"] = [e for e in manifest if set(e["tags"]) & set(item["tags"])]
    return groups


DRILL_TAGS = {"bias-variance-derivation", "softmax-ce-derivation", "elbo-derivation",
              "policy-gradient-derivation", "sqrt-d-derivation"}  # the five priority derivations (★)
LI = re.compile(r"<li>(.*?)</li>", re.S)
ANSWER = re.compile(r"^(.*?)\s*<em>\((.*)\)</em>\s*$", re.S)


def drill_cards(vm: dict, ch: dict) -> list[dict]:
    """Flashcards from one exercise: the derivation itself, plus each 'Interviewers ask' bullet."""
    base = {"exercise": vm["id"], "chapter": ch["num"], "title": vm["title_html"], "number": vm["number"],
            "href": f"{ch['href']}#{vm['id']}"}
    cards = []
    if vm["type"] == "derive":
        cards.append({**base, "id": vm["id"], "kind": "derivation", "q": vm["task"],
                      "a": "".join(f'<div class="dk-step"><span class="dk-step-n">{i}</span><div class="prose">{s}</div></div>'
                                   for i, s in enumerate(vm["steps"], 1)),
                      "star": bool(DRILL_TAGS & set(vm["tags"]))})
    for i, li in enumerate(LI.findall(vm.get("interview", "")), 1):
        m = ANSWER.match(li.strip())
        q, a = (m.group(1), m.group(2)) if m else (li.strip(), "")
        cards.append({**base, "id": f"{vm['id']}#q{i}", "kind": "interview", "q": q, "a": a, "star": False})
    return cards


TABLE = re.compile(r"<table>.*?</table>", re.S)
STRONG = re.compile(r"<strong>(.*?)</strong>", re.S)
SIG = re.compile(r'def (\w+)\(([^)]*)\):\s*\n\s+"""(.+?)"""', re.S)


def cheatsheet_entries(vm: dict, ch: dict) -> dict:
    """Shape-contract tables from task sections, and key ideas (bold lead-ins) from explanations."""
    base = {"title": vm["title_html"], "number": vm["number"], "href": f"{ch['href']}#{vm['id']}"}
    out = {}
    for t in TABLE.findall(vm["task"]):
        if re.search(r"Shape|Inputs|Argument|Output", t.split("</thead>")[0]):
            out["shapes"] = {**base, "table": t}
            break
    sigs = []
    for name, args, doc in SIG.findall(vm.get("starter_raw", "")):
        doc = re.sub(r"\s+", " ", doc).strip()
        if "(" in doc and ("->" in doc or ":" in doc):
            sigs.append({"sig": f"{name}({args})", "doc": doc})
    if sigs:
        out["signatures"] = {**base, "sigs": sigs}
    if "numerical-stability" in vm["tags"] and vm.get("explain"):
        ideas = [re.sub(r"\s+", " ", x).strip().rstrip(".") for x in STRONG.findall(vm["explain"])]
        if ideas:
            out["stability"] = {**base, "ideas": ideas}
    return out


def main() -> int:
    meta, chapters = load_curriculum()
    env = Environment(loader=FileSystemLoader(ROOT / "templates"), undefined=StrictUndefined,
                      trim_blocks=True, lstrip_blocks=True, autoescape=False)
    env.filters["json"] = lambda o: json.dumps(o, ensure_ascii=False).replace("</", "<\\/")
    if DIST.exists():
        shutil.rmtree(DIST)
    shutil.copytree(ROOT / "static", DIST / "static")

    manifest: list[dict] = []
    cards: list[dict] = []
    sheet: dict[str, list] = {"shapes": [], "signatures": [], "stability": []}
    for ch in chapters:
        if not ch.get("file"):
            continue
        parsed = parse_chapter(CONTENT / ch["file"])
        blocks, n = [], 0
        for kind, item in parsed.blocks:
            if kind == "md":
                blocks.append({"kind": "md", "html": render.markdown(item)})
                continue
            n += 1
            vm = render.exercise(item, f"{int(ch['num'])}.{n}")
            vm["local"] = companion.write(ch["num"], item, f"content/{ch['file']}")
            blocks.append({"kind": "exercise", "ex": vm})
            cards.extend(drill_cards(vm, ch))
            for k, v in cheatsheet_entries(vm, ch).items():
                sheet[k].append(v)
            manifest.append({"id": vm["id"], "title": vm["title"], "titleHtml": vm["title_html"], "type": vm["type"], "number": vm["number"],
                             "chapter": ch["num"], "chapterTitle": ch["title"], "tags": vm["tags"],
                             "href": f"{ch['href']}#{vm['id']}"})
        ch["exercise_ids"] = [b["ex"]["id"] for b in blocks if b["kind"] == "exercise"]
        parsed.meta["summary_html"] = render.inline(parsed.meta.get("summary", ""))
        ch["_page"] = (parsed.meta, blocks)

    checklist = load_checklist(manifest)
    known = {t for g in checklist for i in g["items"] for t in i["tags"]}
    stray = sorted({t for e in manifest for t in e["tags"]} - known)
    if stray:
        print(f"  note: tags not on the checklist (fine, but they won't tick anything): {', '.join(stray)}")

    data = {"exercises": manifest,
            "chapters": [{"num": c["num"], "title": c["title"], "href": c["href"], "weeks": c["weeks"],
                          "exercises": c.get("exercise_ids", [])} for c in chapters],
            "checklist": [{"id": i["id"], "tags": i["tags"]} for g in checklist for i in g["items"]]}
    common = dict(site=meta, chapters=chapters, parts=meta["parts"], data=data)
    for ch in chapters:
        if "_page" not in ch:
            continue
        cmeta, blocks = ch.pop("_page")
        html = env.get_template("chapter.html").render(
            **common, ch=ch, cmeta=cmeta, blocks=blocks, page_title=f"{ch['num']} · {ch['title']}", page="chapter")
        (DIST / ch["href"]).write_text(html)
        print(f"  built {ch['href']}  ({len(ch['exercise_ids'])} exercises)")

    (DIST / "index.html").write_text(env.get_template("index.html").render(
        **common, ch=None, page_title=meta["title"], page="home"))
    (DIST / "checklist.html").write_text(env.get_template("checklist.html").render(
        **common, ch=None, groups=checklist, page_title="Checklist", page="checklist"))
    cards.sort(key=lambda c: (not c["star"], c["kind"] != "derivation", c["chapter"]))
    (DIST / "cheatsheet.html").write_text(env.get_template("cheatsheet.html").render(
        **common, ch=None, sheet=sheet, page_title="Cheat sheet", page="cheatsheet"))
    print(f"  built cheatsheet.html  ({len(sheet['shapes'])} shape tables, {sum(len(e['sigs']) for e in sheet['signatures'])} signatures, {len(sheet['stability'])} stability lessons)")
    (DIST / "drills.html").write_text(env.get_template("drills.html").render(
        **common, ch=None, cards=cards, page_title="Drill deck", page="drills"))
    n_ans = sum(1 for c in cards if c["kind"] == "interview" and c["a"])
    print(f"  built drills.html  ({len(cards)} cards: {sum(c['kind'] == 'derivation' for c in cards)} derivations, "
          f"{len(cards) - sum(c['kind'] == 'derivation' for c in cards)} interview questions, {n_ans} with answers)")
    print(f"  built index.html, checklist.html  ({len(manifest)} exercises in total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
