"""Generate (python tools/figures/make_broadcast.py) the broadcasting SVG figures into content/figures/.

All colour comes from classes styled in static/css/figures.css (theme variables).
Animation hooks: class fx-a-in / fx-a-slide / fx-a-flash / fx-a-merge plus inline
--d (delay), --dx/--dy (start offset), --dur. Without .fx-on on the <figure>, the
static final state shows (no JS, reduced motion).
"""
from pathlib import Path
from html import escape

OUT = Path(__file__).resolve().parents[2] / "content" / "figures"


def st(**kw):
    m = {"d": "--d", "dx": "--dx", "dy": "--dy", "dur": "--dur"}
    parts = []
    for k, v in kw.items():
        if v is None:
            continue
        unit = "s" if k in ("d", "dur") else "px"
        parts.append(f"{m[k]}:{v}{unit}")
    return f' style="{";".join(parts)}"' if parts else ""


def anim(cls, **kw):
    """Return (class string, style string) for an animated group."""
    return cls, st(**kw)


def g(children, cls="", **kw):
    c = f' class="{cls}"' if cls else ""
    return f"<g{c}{st(**kw)}>" + "".join(children) + "</g>"


def rect(x, y, w, h, cls, rx=7):
    return f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"/>'


def text(x, y, s, cls, size, anchor="middle", weight=None):
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    w = f' font-weight="{weight}"' if weight else ""
    return f'<text class="{cls}" x="{x}" y="{y}" font-size="{size}"{a}{w}>{s}</text>'


def cell(x, y, w, h, s, kind, size=16, rx=7):
    """kind: real | virt | ok | err | ghost | stretch | blank"""
    out = rect(x, y, w, h, f"fx-cell fx-{kind}", rx)
    if s != "":
        out += text(x + w / 2, y + h / 2 + size * 0.36, s, f"fx-mono fx-tx-{kind}", size)
    return out


def svg(name, w, h, title, desc, body, animated):
    cls = "fx fx-anim" if animated else "fx"
    tid, did = f"fx-{name}-t", f"fx-{name}-d"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" class="{cls}" viewBox="0 0 {w} {h}" '
        f'role="img" aria-labelledby="{tid} {did}">\n'
        f'<title id="{tid}">{escape(title)}</title>\n<desc id="{did}">{escape(desc)}</desc>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def legend(y, items, x0=14, gap=14):
    """items: list of (kind, label, symbol)."""
    out, x = [], x0
    for kind, label, sym in items:
        out.append(rect(x, y - 13, 20, 18, f"fx-cell fx-{kind}", 4))
        out.append(text(x + 10, y + 1, sym, f"fx-ui fx-tx-{kind}", 13, weight=700))
        out.append(text(x + 26, y + 1, label, "fx-ui fx-muted", 14, anchor="start"))
        x += 26 + len(label) * 7.3 + gap
    return "".join(out)

W = 380
MINUS = "−"


# ─────────────────────────────────────────────────────────────── 1. align
def fig_align():
    BW, BH = 58, 40
    cx = [58, 126, 194]  # x of the boxes for axes -3, -2, -1 (right edge 252)
    SX = 266  # shape labels
    body = []

    def row_label(y, s):
        return text(14, y + BH / 2 + 6, s, "fx-mono fx-ink", 17, anchor="start", weight=600)

    def shape_label(y, s):
        return text(SX, y + BH / 2 + 5, s, "fx-mono fx-muted", 15, anchor="start")

    def col_hl(c, y0, y1, d, dur=1.1, err=False):
        k = "fx-hl-err" if err else "fx-hl"
        return f'<rect class="{k} fx-a-flash" x="{cx[c] - 5}" y="{y0}" width="{BW + 10}" height="{y1 - y0}" rx="10"{st(d=d, dur=dur)}/>'

    # ---- panel A: (4,1,3) - (2,3)
    ta, tb, tr = 36, 84, 162
    body.append(text(14, 20, f'Works: <tspan class="fx-mono">(4, 1, 3) {MINUS} (2, 3)</tspan>', "fx-ui fx-ok-ink", 15, anchor="start", weight=600))
    for c, d in ((2, 2.6), (1, 3.7), (0, 4.8)):
        body.append(col_hl(c, 30, tr + BH + 30, d))
    body.append(row_label(ta, "a"))
    body.append(cell(cx[0], ta, BW, BH, "4", "real", 18))
    body.append(cell(cx[1], ta, BW, BH, "1", "one", 18))
    body.append(cell(cx[2], ta, BW, BH, "3", "real", 18))
    body.append(shape_label(ta, "(4, 1, 3)"))
    body.append(row_label(tb, "b"))
    body.append(g([cell(cx[0], tb, BW, BH, "1", "ghost", 18),
                   text(cx[0] + BW / 2, tb + BH + 17, "missing → 1", "fx-ui fx-muted", 13)], "fx-a-in", d=1.9))
    body.append(g([cell(cx[1], tb, BW, BH, "2", "real", 18), cell(cx[2], tb, BW, BH, "3", "real", 18)],
                  "fx-a-slide", d=0.5, dx=-(cx[1] - cx[0])))
    body.append(shape_label(tb, "(2, 3)"))
    body.append(f'<line class="fx-rule" x1="14" y1="{tr - 10}" x2="366" y2="{tr - 10}"/>')
    body.append(row_label(tr, f"a{MINUS}b"))
    res = [("4", "stretch", "1 → 4", 4.9), ("2", "stretch", "1 → 2", 3.8), ("3", "ok", "3 = 3 ✓", 2.7)]
    for c, (v, kind, lab, d) in enumerate(res):
        body.append(g([cell(cx[c], tr, BW, BH, v, kind, 18),
                       text(cx[c] + BW / 2, tr + BH + 20, lab, f"fx-ui fx-tx-{kind}", 14, weight=600)],
                      "fx-a-in", d=d))
    body.append(g([shape_label(tr, "(4, 2, 3)")], "fx-a-in", d=5.6))

    # ---- panel B: (4,3) - (4,)
    o = 250
    ta, tb, tr = o + 36, o + 84, o + 146
    body.append(f'<line class="fx-sep" x1="14" y1="{o - 8}" x2="366" y2="{o - 8}"/>')
    body.append(text(14, o + 20, f'Fails: <tspan class="fx-mono">(4, 3) {MINUS} (4,)</tspan>', "fx-ui fx-err-ink", 15, anchor="start", weight=600))
    body.append(col_hl(2, o + 30, tr + BH + 30, 8.0, dur=1.6, err=True))
    body.append(row_label(ta, "a"))
    body.append(cell(cx[1], ta, BW, BH, "4", "real", 18))
    body.append(cell(cx[2], ta, BW, BH, "3", "real", 18))
    body.append(shape_label(ta, "(4, 3)"))
    body.append(row_label(tb, "b"))
    body.append(g([cell(cx[1], tb, BW, BH, "1", "ghost", 18)], "fx-a-in", d=7.3))
    body.append(g([cell(cx[2], tb, BW, BH, "4", "real", 18)], "fx-a-slide", d=6.3, dx=-(cx[2] - cx[1])))
    body.append(shape_label(tb, "(4,)"))
    body.append(f'<line class="fx-rule" x1="14" y1="{tr - 10}" x2="366" y2="{tr - 10}"/>')
    body.append(row_label(tr, f"a{MINUS}b"))
    body.append(g([cell(cx[2], tr, BW, BH, "✗", "err", 18),
                   text(cx[2] + BW / 2, tr + BH + 20, "3 ≠ 4", "fx-ui fx-tx-err", 14, weight=600),
                   text(cx[1] + BW, tr + BH / 2 + 6, "ValueError", "fx-mono fx-tx-err", 16, anchor="end", weight=600)],
                  "fx-a-in", d=8.3))
    fy = tr + BH + 52
    body.append(g([text(14, fy, 'Meant “one value per row”? Use <tspan class="fx-mono fx-ink">b[:, None]</tspan>:', "fx-ui fx-text", 14, anchor="start"),
                   text(14, fy + 20, 'shape (4, 1), and the result is (4, 3) ✓', "fx-ui fx-text", 14, anchor="start")],
                  "fx-a-in", d=9.3))
    ly = fy + 56
    body.append(legend(ly, [("ok", "equal: keep", "✓"), ("stretch", "1: stretch", "↔"), ("err", "else: error", "✗")]))
    H = ly + 12
    desc = ("Shapes are written as rows of boxes and lined up from the right. "
            "Top: a is (4, 1, 3) and b is (2, 3); b slides right so its last axis sits under a's last axis, and its missing "
            "leading axis becomes a ghost 1. Comparing column by column from the right: 3 and 3 are equal, so keep 3; "
            "1 and 2, so the 1 stretches to 2; 4 and the ghost 1, so the 1 stretches to 4. Result (4, 2, 3). "
            "Bottom: a is (4, 3) and b is (4,); right-aligned, the last column pairs 3 with 4, which are unequal and neither "
            "is 1, so NumPy raises ValueError. If you meant one value per row, b[:, None] has shape (4, 1) and gives (4, 3).")
    return svg("align", W, H, "Broadcasting aligns shapes from the right", desc, body, True)


# ─────────────────────────────────────────────────────────────── 2. stretch
def fig_stretch():
    s, c = 34, 30  # pitch, cell size
    a = [10, 20, 30]
    b = [1, 2, 3, 4]
    ax, bx, gy = 26, 218, 78
    rx, ry = 124, 262
    body = []
    steps = [("① stretch a →", 64, 0.3, 2.0), ("② stretch b ↓", 190, 2.4, 1.7), ("③ subtract", 316, 4.2, 2.2)]
    for lab, x, d, dur in steps:
        body.append(f'<rect class="fx-pill fx-a-flash" x="{x - 59}" y="6" width="118" height="28" rx="14"{st(d=d, dur=dur)}/>')
        body.append(text(x, 25, lab, "fx-ui fx-text", 14, weight=600))
    body.append(text(ax, gy - 12, 'a <tspan class="fx-muted">(3, 1)</tspan>', "fx-mono fx-ink", 15, anchor="start", weight=600))
    body.append(text(bx, gy - 12, 'b <tspan class="fx-muted">(1, 4)</tspan>', "fx-mono fx-ink", 15, anchor="start", weight=600))
    for i in range(3):
        body.append(cell(ax, gy + i * s, c, c, str(a[i]), "real", 15, 6))
    for j in range(1, 4):
        body.append(g([cell(ax + j * s, gy + i * s, c, c, str(a[i]), "virt", 15, 6) for i in range(3)],
                      "fx-a-slide fx-fade", d=0.5 + 0.4 * j, dx=-s))
    for j in range(4):
        body.append(cell(bx + j * s, gy, c, c, str(b[j]), "real", 15, 6))
    for i in range(1, 3):
        body.append(g([cell(bx + j * s, gy + i * s, c, c, str(b[j]), "virt", 15, 6) for j in range(4)],
                      "fx-a-slide fx-fade", d=2.6 + 0.4 * i, dy=-s))
    body.append(text((ax + 4 * s + bx) / 2 - 2, gy + 1.5 * s + 4, MINUS, "fx-mono fx-ink", 24))
    body.append(g([text(ax + 2 * s - 2, gy + 3 * s + 16, "acts as (3, 4)", "fx-ui fx-tx-virt", 14, weight=600)], "fx-a-in", d=1.9))
    body.append(g([text(bx + 2 * s - 2, gy + 3 * s + 16, "acts as (3, 4)", "fx-ui fx-tx-virt", 14, weight=600)], "fx-a-in", d=3.6))
    body.append(text(rx - 22, ry + 1.5 * s + 4, "=", "fx-mono fx-ink", 24))
    body.append(text(rx, ry - 12, f'a {MINUS} b <tspan class="fx-muted">(3, 4)</tspan>', "fx-mono fx-ink", 15, anchor="start", weight=600))
    for i in range(3):
        body.append(g([cell(rx + j * s, ry + i * s, c, c, str(a[i] - b[j]), "ok", 15, 6) for j in range(4)],
                      "fx-a-in", d=4.5 + 0.45 * i))
    body.append(g([text(rx + 4 * s + 6, ry + s + 20, f"← 20 {MINUS} b", "fx-mono fx-muted", 14, anchor="start")], "fx-a-in", d=6.0))
    ny = ry + 3 * s + 30
    body.append(text(14, ny, "Dashed cells are never stored. The view steps 0 bytes", "fx-ui fx-text", 14, anchor="start"))
    body.append(text(14, ny + 20, "along a stretched axis, so it re-reads the same number:", "fx-ui fx-text", 14, anchor="start"))
    body.append(text(14, ny + 44, 'np.broadcast_to(a, (3, 4)).strides <tspan class="fx-muted"># (8, 0)</tspan>', "fx-mono fx-ink", 14, anchor="start"))
    body.append(text(14, ny + 68, f"Only the result a {MINUS} b is a real, allocated array.", "fx-ui fx-text", 14, anchor="start"))
    H = ny + 80
    desc = ("a is a column of shape (3, 1) holding 10, 20, 30; b is a row of shape (1, 4) holding 1, 2, 3, 4. "
            "Step 1: a's column is repeated across 4 columns (dashed, virtual copies). Step 2: b's row is repeated down "
            "3 rows (dashed, virtual). Step 3: the two (3, 4) grids are subtracted elementwise, giving rows 9 8 7 6, "
            "19 18 17 16, 29 28 27 26; for example row 2, column 3 is 20 minus 3 = 17. The dashed copies are never stored: "
            "the broadcast view has stride 0 along the stretched axis. Only the result is allocated.")
    return svg("stretch", W, H, "What stretching a size-1 axis means, on actual numbers", desc, body, True)


# ─────────────────────────────────────────────────────────────── 3. keepdims
def fig_keepdims():
    s, c = 31, 27
    z = [[1, 5, 2], [7, 3, 0], [4, 4, 9]]
    m = [max(r) for r in z]
    zx, mx, rx = 22, 146, 270
    op1, op2 = (zx + 3 * s + mx) / 2 - 2, (mx + 3 * s + rx) / 2 - 2
    body = []

    def num(v):
        return str(v).replace("-", MINUS)

    def zgrid(y):
        return [cell(zx + j * s, y + i * s, c, c, str(z[i][j]), "real", 15, 6) for i in range(3) for j in range(3)]

    def ops(y):
        return [text(op1, y + 1.5 * s + 3, MINUS, "fx-mono fx-ink", 22), text(op2, y + 1.5 * s + 3, "=", "fx-mono fx-ink", 22)]

    def labels(y, mid):
        return [text(zx + 1.5 * s - 2, y + 3 * s + 15, "z (3, 3)", "fx-mono fx-muted", 14),
                text(mx + 1.5 * s - 2, y + 3 * s + 15, mid, "fx-mono fx-muted", 14),
                text(rx + 1.5 * s - 2, y + 3 * s + 15, f"z {MINUS} m", "fx-mono fx-muted", 14)]

    # ---- panel A
    y = 36
    body.append(text(14, 20, '✓ <tspan class="fx-mono">m = z.max(axis=1, keepdims=True)</tspan>', "fx-ui fx-ok-ink", 14.5, anchor="start", weight=600))
    body += zgrid(y) + ops(y)
    for i in range(3):
        body.append(cell(mx, y + i * s, c, c, str(m[i]), "real", 15, 6))
    for j in (1, 2):
        body.append(g([cell(mx + j * s, y + i * s, c, c, str(m[i]), "virt", 15, 6) for i in range(3)],
                      "fx-a-slide fx-fade", d=0.6 + 0.4 * j, dx=-s))
    body.append(g([cell(rx + j * s, y + i * s, c, c, num(z[i][j] - m[i]), "ok" if z[i][j] == m[i] else "plain", 15, 6)
                   for i in range(3) for j in range(3)], "fx-a-in", d=2.0))
    body += labels(y, "m (3, 1)")
    body.append(g([text(14, y + 3 * s + 42, "Each row loses its own max (the 0s). Correct.", "fx-ui fx-ok-ink", 14, anchor="start")], "fx-a-in", d=2.6))

    # ---- panel B
    o = 186
    y = o + 54
    body.append(f'<line class="fx-sep" x1="14" y1="{o - 8}" x2="366" y2="{o - 8}"/>')
    body.append(text(14, o + 16, '✗ <tspan class="fx-mono">m = z.max(axis=1)</tspan>', "fx-ui fx-err-ink", 14.5, anchor="start", weight=600))
    body.append(text(14, o + 36, "shape (3,) lines up with the last axis: acts as (1, 3)", "fx-ui fx-err-ink", 14, anchor="start"))
    body += zgrid(y) + ops(y)
    for k in range(3):  # m lies down: (col 0, row k) -> (col k, row 0)
        body.append(g([cell(mx + k * s, y, c, c, str(m[k]), "real", 15, 6)], "fx-a-slide", d=4.2, dx=-k * s, dy=k * s, dur=1.3))
    for i in (1, 2):
        body.append(g([cell(mx + j * s, y + i * s, c, c, str(m[j]), "virt", 15, 6) for j in range(3)],
                      "fx-a-slide fx-fade", d=5.6 + 0.4 * i, dy=-s))
    body.append(g([cell(rx + j * s, y + i * s, c, c, num(z[i][j] - m[j]), "err" if z[i][j] > m[j] else "plain", 15, 6)
                   for i in range(3) for j in range(3)], "fx-a-in", d=6.9))
    body += labels(y, "m (3,)")
    ny = y + 3 * s + 42
    body.append(g([text(14, ny, "Column j loses m[j], the max of row j. Each row", "fx-ui fx-err-ink", 14, anchor="start"),
                   text(14, ny + 20, "gets uneven shifts, so the softmax itself changes", "fx-ui fx-err-ink", 14, anchor="start"),
                   text(14, ny + 40, f"(and 7 {MINUS} 5 = 2 is left above 0).", "fx-ui fx-err-ink", 14, anchor="start")],
                  "fx-a-in", d=7.6))
    yb = ny + 70
    body.append(text(14, yb, "B = C = 3 here, so it runs silently. B ≠ C raises ValueError.", "fx-ui fx-muted", 14, anchor="start"))
    H = yb + 10
    desc = ("z is a 3 by 3 batch of logits: rows 1 5 2, 7 3 0, 4 4 9. Its row maxima are 5, 7, 9. "
            "Top, correct: with keepdims=True, m has shape (3, 1), a column, and stretches across each row, so every row "
            "has its own max subtracted; each row of the result contains a 0 where its max was. "
            "Bottom, bug: without keepdims, m has shape (3,). Broadcasting lines it up with the last axis, so it acts "
            "like a row (1, 3) and stretches down the columns: element [i, j] has m[j] subtracted. Rows get different "
            "shifts per column, which changes the softmax, and row 2 ends up with 7 minus 5 = 2, a positive value. "
            "Because the batch is square it runs without an error; with a non-square batch it raises ValueError.")
    return svg("keepdims", W, H, "keepdims decides which way the row maxima stretch", desc, body, True)


# ─────────────────────────────────────────────────────────────── 4. pairwise
def fig_pairwise():
    s, c, D, off = 28, 24, 3, 7
    body = []

    def sheet(x0, y0, rows, cols, k, kind):
        out = [rect(x0 + k * off - 3, y0 - k * off - 3, cols * s + 2, rows * s + 2, f"fx-sheet fx-sheet-{min(k, 1)}", 6)]
        for i in range(rows):
            for j in range(cols):
                out.append(rect(x0 + k * off + j * s, y0 - k * off + i * s, c, c, f"fx-cell fx-{kind}", 5))
        return "".join(out)

    def stack(x0, y0, rows, cols, kind):
        return [sheet(x0, y0, rows, cols, k, kind) for k in reversed(range(D))]

    yx, yy = 112, 72     # Y' front sheet
    xx, xy = 52, 136     # X' front sheet
    gx, gy = yx, xy      # grid
    # Y' on top
    body.append(text(yx, 22, "Y[None, :, :]", "fx-mono fx-ink", 15, anchor="start", weight=600))
    body.append(text(yx, 41, "(1, M, D): a row of y’s", "fx-ui fx-muted", 14, anchor="start"))
    body += stack(yx, yy, 1, 4, "real")
    for j in range(4):
        body.append(text(yx + j * s + c / 2, yy + 17, f"y{'₀₁₂₃'[j]}", "fx-mono fx-tx-real", 14))
    # X' on the left
    body += stack(xx, xy, 3, 1, "real")
    for i in range(3):
        body.append(text(xx + c / 2, xy + i * s + 17, f"x{'₀₁₂'[i]}", "fx-mono fx-tx-real", 14))
    body.append(text(14, xy + 3 * s + 26, "X[:, None, :]", "fx-mono fx-ink", 15, anchor="start", weight=600))
    body.append(text(14, xy + 3 * s + 45, "(N, 1, D)", "fx-ui fx-muted", 14, anchor="start"))
    # difference grid, sheets appear back to front
    hi, hj = 1, 2
    for k in reversed(range(D)):  # back to front; the highlighted cell [1, 2] sits inside each sheet
        body.append(g([sheet(gx, gy, 3, 4, k, "plain")], "fx-a-in", d=0.4 + 0.35 * (2 - k)))
        body.append(g([rect(gx + k * off + hj * s, gy - k * off + hi * s, c, c, "fx-cell fx-focus", 5)], "fx-a-in", d=1.9))
    body.append(text(gx + 4 * s + 2 * off + 2, gy - 2 * off - 6, "D", "fx-mono fx-muted", 14, anchor="start", weight=600))
    body.append(g([text(gx + 2 * s + 34, xy + 3 * s + 26, "difference", "fx-ui fx-ink", 14, anchor="start", weight=600),
                   text(gx + 2 * s + 34, xy + 3 * s + 45, "(N, M, D)", "fx-mono fx-muted", 14, anchor="start")],
                  "fx-a-in", d=1.2))
    # highlighted cell [1, 2] through all sheets + guides
    tube = [f'<line class="fx-guide" x1="{xx + c + 2}" y1="{xy + hi * s + c / 2}" x2="{gx + hj * s - 2}" y2="{gy + hi * s + c / 2}"/>',
            f'<line class="fx-guide" x1="{yx + hj * s + c / 2}" y1="{yy + c + 2}" x2="{yx + hj * s + c / 2}" y2="{gy + hi * s - 2}"/>']
    tx = gx + 4 * s + 2 * off + 12
    tube += [text(tx, gy + 4, "cell [1, 2]:", "fx-ui fx-text", 14, anchor="start"),
             text(tx, gy + 24, f"x₁ {MINUS} y₂", "fx-mono fx-tx-focus", 15, anchor="start", weight=600),
             text(tx, gy + 44, "D numbers,", "fx-ui fx-text", 14, anchor="start"),
             text(tx, gy + 62, "one per sheet", "fx-ui fx-text", 14, anchor="start")]
    body.append(g(tube, "fx-a-in", d=1.9))
    # arrow + reduction
    ay0 = xy + 3 * s + 60
    body.append(g([f'<path class="fx-arrow" d="M{gx + 2 * s - 2} {ay0} v26 m-6 -8 l6 8 l6 -8"/>',
                   text(gx + 2 * s + 12, ay0 + 19, "(diff ** 2).sum(-1)", "fx-mono fx-ink", 14, anchor="start", weight=600)],
                  "fx-a-in", d=3.4))
    fx_, fy = gx, ay0 + 44 + 2 * off
    for k in (2, 1):
        body.append(g([sheet(fx_, fy, 3, 4, k, "plain")], "fx-a-merge", d=4.0, dx=-k * off, dy=k * off, dur=2.2))
    body.append(g([sheet(fx_, fy, 3, 4, 0, "plain")], "fx-a-in", d=4.0))
    done = [rect(fx_ + j * s, fy + i * s, c, c, "fx-cell fx-ok", 5) for i in range(3) for j in range(4)]
    done.append(text(fx_ + hj * s + c / 2, fy + hi * s + 17, "d", "fx-mono fx-tx-ok", 14, weight=600))
    body.append(g(done, "fx-a-in", d=5.9))
    body.append(g([text(tx, fy + 10, "(N, M)", "fx-mono fx-ink", 15, anchor="start", weight=600),
                   text(tx, fy + 30, "D summed", "fx-ui fx-muted", 14, anchor="start"),
                   text(tx, fy + 48, "away: one", "fx-ui fx-muted", 14, anchor="start"),
                   text(tx, fy + 66, "distance", "fx-ui fx-muted", 14, anchor="start"),
                   text(tx, fy + 84, "per pair", "fx-ui fx-muted", 14, anchor="start")],
                  "fx-a-in", d=6.3))
    H = fy + 3 * s + 8
    desc = ("Pairwise distances as a broadcast. Y[None, :, :] has shape (1, M, D): a row of M vectors y0 to y3, drawn as D "
            "stacked sheets, one per feature. X[:, None, :] has shape (N, 1, D): a column of N vectors x0 to x2. Subtracting "
            "them fills an N by M grid like a times table, still with D sheets: shape (N, M, D). Cell [1, 2] holds x1 minus "
            "y2, one difference per feature. Squaring and summing over the last axis collapses the D sheets into one, "
            "giving the (N, M) matrix of squared distances.")
    return svg("pairwise", W, H, "Pairwise distances: (N, 1, D) minus (1, M, D) gives (N, M, D), then sum over D", desc, body, True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("broadcast-align", fig_align), ("broadcast-stretch", fig_stretch),
                     ("broadcast-keepdims", fig_keepdims), ("broadcast-pairwise", fig_pairwise)]:
        (OUT / f"{name}.svg").write_text(fn())
        print("wrote", name)
