"""Build the workbook: content/*.md → dist/*.html.  Usage: python build/build.py"""

from __future__ import annotations

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
            chapters.append(ch)
    return meta, chapters


def main() -> int:
    meta, chapters = load_curriculum()
    env = Environment(loader=FileSystemLoader(ROOT / "templates"), undefined=StrictUndefined,
                      trim_blocks=True, lstrip_blocks=True, autoescape=False)
    if DIST.exists():
        shutil.rmtree(DIST)
    shutil.copytree(ROOT / "static", DIST / "static")

    for ch in chapters:
        if not ch.get("file"):
            continue
        parsed = parse_chapter(CONTENT / ch["file"])
        blocks, n = [], 0
        for kind, item in parsed.blocks:
            if kind == "md":
                blocks.append({"kind": "md", "html": render.markdown(item)})
            else:
                n += 1
                vm = render.exercise(item, f"{int(ch['num'])}.{n}")
                vm["local"] = companion.write(ch["num"], item, f"content/{ch['file']}")
                blocks.append({"kind": "exercise", "ex": vm})
        ch["exercise_ids"] = [b["ex"]["id"] for b in blocks if b["kind"] == "exercise"]
        parsed.meta["summary_html"] = render.inline(parsed.meta.get("summary", ""))
        ch["_page"] = (parsed.meta, blocks)

    for ch in chapters:
        if "_page" not in ch:
            continue
        cmeta, blocks = ch.pop("_page")
        html = env.get_template("chapter.html").render(
            site=meta, chapters=chapters, parts=meta["parts"], ch=ch, cmeta=cmeta, blocks=blocks,
            page_title=f"{ch['num']} · {ch['title']}")
        (DIST / ch["href"]).write_text(html)
        print(f"  built {ch['href']}  ({len(ch['exercise_ids'])} exercises)")

    (DIST / "index.html").write_text(env.get_template("index.html").render(
        site=meta, chapters=chapters, parts=meta["parts"], ch=None, page_title=meta["title"]))
    print("  built index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
