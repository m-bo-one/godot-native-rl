#!/usr/bin/env python3
"""Generate the Godot Native RL Asset Library icon (neural-net policy motif).

Emits an SVG (vector source) describing a 3-layer network on a dark navy gradient:
cyan input/hidden nodes feeding a highlighted orange output node (echoing the demos'
agent=cyan / target=orange palette). Rasterize with rsvg-convert. Pure stdlib.
"""

CY = "#2ec6e0"   # cyan — agent / signal
OR = "#f4a72c"   # orange — output action / target
W = 512

# Layered node positions (x, [y...]).
INPUT = (150, [136, 218, 300, 382])
HIDDEN = (256, [136, 196, 256, 316, 382])
OUTPUT = (362, [206, 306])

# A single "signal path" we brighten to suggest forward inference flow.
HOT = {"in": 1, "hid": 2, "out": 0}


def nodes(x, ys):
    return [(x, y) for y in ys]


def main() -> None:
    inp, hid, out = nodes(*INPUT), nodes(*HIDDEN), nodes(*OUTPUT)
    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{W}" viewBox="0 0 {W} {W}">')
    parts.append('<defs>')
    parts.append('<linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">'
                 '<stop offset="0" stop-color="#173a55"/>'
                 '<stop offset="0.55" stop-color="#0d1f30"/>'
                 '<stop offset="1" stop-color="#070d15"/></linearGradient>')
    parts.append('<radialGradient id="glow" cx="0.5" cy="0.46" r="0.55">'
                 f'<stop offset="0" stop-color="{CY}" stop-opacity="0.22"/>'
                 f'<stop offset="1" stop-color="{CY}" stop-opacity="0"/></radialGradient>')
    parts.append(f'<radialGradient id="ncy" cx="0.38" cy="0.34" r="0.75">'
                 '<stop offset="0" stop-color="#bff2fb"/>'
                 f'<stop offset="0.45" stop-color="{CY}"/>'
                 '<stop offset="1" stop-color="#1689a3"/></radialGradient>')
    parts.append(f'<radialGradient id="nor" cx="0.38" cy="0.34" r="0.75">'
                 '<stop offset="0" stop-color="#ffe2ad"/>'
                 f'<stop offset="0.45" stop-color="{OR}"/>'
                 '<stop offset="1" stop-color="#c2761a"/></radialGradient>')
    parts.append('</defs>')

    # Background panel + glow + subtle border.
    parts.append(f'<rect x="0" y="0" width="{W}" height="{W}" rx="104" fill="url(#bg)"/>')
    parts.append(f'<rect x="0" y="0" width="{W}" height="{W}" rx="104" fill="url(#glow)"/>')
    parts.append(f'<rect x="14" y="14" width="{W-28}" height="{W-28}" rx="92" fill="none" '
                 f'stroke="{CY}" stroke-opacity="0.22" stroke-width="3"/>')

    # Connections (drawn under nodes).
    def line(a, b, hot):
        op, wdt, col = (0.7, 3.4, CY) if hot else (0.18, 2.0, CY)
        return (f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" '
                f'stroke="{col}" stroke-opacity="{op}" stroke-width="{wdt}" stroke-linecap="round"/>')

    for i, a in enumerate(inp):
        for j, b in enumerate(hid):
            parts.append(line(a, b, i == HOT["in"] and j == HOT["hid"]))
    for j, a in enumerate(hid):
        for k, b in enumerate(out):
            parts.append(line(a, b, j == HOT["hid"] and k == HOT["out"]))

    # Nodes (glow halo + core).
    def node(p, r, grad, halo):
        return (f'<circle cx="{p[0]}" cy="{p[1]}" r="{r*2.0}" fill="{halo}" fill-opacity="0.14"/>'
                f'<circle cx="{p[0]}" cy="{p[1]}" r="{r}" fill="url(#{grad})" '
                f'stroke="#06121b" stroke-opacity="0.5" stroke-width="2"/>')

    for p in inp + hid:
        parts.append(node(p, 15, "ncy", CY))
    for p in out:
        parts.append(node(p, 23, "nor", OR))

    parts.append('</svg>')
    with open("media/icon.svg", "w") as f:
        f.write("\n".join(parts))
    print("wrote media/icon.svg")


if __name__ == "__main__":
    main()
