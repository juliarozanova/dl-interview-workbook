"""Build the workbook: content/*.md → dist/*.html (+ companion/).  Usage: python build/build.py"""

from __future__ import annotations

import json
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
CONTENT, DIST = ROOT / "content", ROOT / "dist"


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


def main() -> int:
    meta, chapters = load_curriculum()
    env = Environment(loader=FileSystemLoader(ROOT / "templates"), undefined=StrictUndefined,
                      trim_blocks=True, lstrip_blocks=True, autoescape=False)
    env.filters["json"] = lambda o: json.dumps(o, ensure_ascii=False).replace("</", "<\\/")
    if DIST.exists():
        shutil.rmtree(DIST)
    shutil.copytree(ROOT / "static", DIST / "static")

    manifest: list[dict] = []
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
    print(f"  built index.html, checklist.html  ({len(manifest)} exercises in total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
