#!/usr/bin/env python3
"""
prelabel — turn rough weapon hints into tight CVAT pre-labels with SAM 2.

The hints file (JSON) lists, per image, every weapon a person (or assistant)
spotted, with a ROUGH box and rough end points:

  {"fight01_camA_026648ms.png": [
     {"cls": "sword", "box": [810, 205, 990, 380],
      "handle": [828, 360], "strike": [975, 218],      # null = cannot locate -> Outside
      "pos": [[900, 290]],                              # optional extra points on the weapon
      "neg": [[880, 330]],                              # optional points NOT on it (SAM spilled there)
      "nobox": false},                                  # true: prompt SAM with points only
     {"cls": "sword", "box": [...], "line": [[x, y], ...], "width": 6, ...},
                                                        # SAM cannot isolate it: trace the blade instead
     {"cls": "shield", "box": [622, 190, 698, 288],
      "top": [660, 196]},                               # which way is up; null -> all 4 points Outside
     {"cls": "sword", "box": [...], "strike": [...],
      "occluded": {"handle": [579, 287]},               # hidden but obvious: placed exactly here, flagged occluded
      "outside": ["left", "right"]}                     # force these points Outside (e.g. edge-on shield)
  ]}

  segment  SAM 2 cuts out each weapon from its hint. The tight box is the
           cut-out's bounding box; end points snap to the cut-out's extremes:
             handle_end / striking_end : centre of the extreme edge along the
                                         weapon's axis, nearest the hint
             shield top/bottom          : ends of the long axis, top = side the hint points to
             shield left/right          : ends of the widest line across; left is
                                         90 deg counter-clockwise from "up"
           -> <out>/prelabels.json and <out>/overlays/*.png for checking

  push     Replace the CVAT task's annotations with prelabels.json
           (each box grouped with its skeleton).
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

WEAPONS = ("sword", "axe", "mace")
ALIAS = {"handle": "handle_end", "strike": "striking_end"}   # hint name -> skeleton point name
COLORS = {"sword": (230, 25, 75), "axe": (245, 130, 49), "mace": (145, 30, 180), "shield": (67, 99, 216)}


# ---------------------------------------------------------------- geometry

def edge_centre(pix: np.ndarray, d: np.ndarray, near: np.ndarray | None, radius: float, band: float = 3.0):
    """Centre of the mask's extreme edge in direction d (pix: N x 2 of x,y).
    If `near` is given, only pixels within `radius` of it are considered."""
    cand = pix
    if near is not None:
        m = np.linalg.norm(pix - near, axis=1) <= radius
        if m.sum() >= 3:
            cand = pix[m]
    proj = cand @ d
    edge = cand[proj >= proj.max() - band]
    return np.median(edge, axis=0)


def line_mask(pts, width: float, W: int, H: int) -> np.ndarray:
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).line([tuple(p) for p in pts], fill=255, width=int(round(width)))
    return np.array(im) > 0


def blade_line(h: dict) -> list:
    """Hinted points along a blade (strike, pos..., handle) ordered along it; the open end(s)
    are extended to the hint box edge so a hidden hilt or tip is still covered."""
    pts = [p for p in [h.get("strike"), *(h.get("pos") or []), h.get("handle")] if p]
    pts = np.array(pts, float)
    if len(pts) < 2:
        x1, y1, x2, y2 = h["box"]
        return [(x1, y1), (x2, y2)]
    d = unit(pts[-1] - pts[0])
    pts = pts[np.argsort(pts @ d)]
    x1, y1, x2, y2 = h["box"]
    diag = np.hypot(x2 - x1, y2 - y1)
    ends = []
    for p, s in ((pts[0], -1), (pts[-1], 1)):
        named = any(np.allclose(p, q) for q in (h.get("strike"), h.get("handle")) if q)
        if named:
            ends.append(p)
        else:
            q = p + s * d * diag
            ends.append(np.clip(q, [x1, y1], [x2, y2]))
    return [tuple(ends[0])] + [tuple(p) for p in pts[1:-1]] + [tuple(ends[1])]


def principal_axis(pix: np.ndarray) -> np.ndarray:
    c = pix - pix.mean(0)
    w, v = np.linalg.eigh(np.cov(c.T))
    return v[:, np.argmax(w)]


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


# ---------------------------------------------------------------- segment

def cmd_segment(a) -> None:
    from ultralytics import SAM
    hints = json.loads(Path(a.hints).read_text())
    img_dir = Path(a.images)
    out = Path(a.out)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    model = SAM(a.model)
    only = set(a.only.split(",")) if a.only else None
    result = {}
    if (out / "prelabels.json").exists() and only:
        result = json.loads((out / "prelabels.json").read_text())
    for name in sorted(hints):
        if only and name not in only:
            continue
        im = Image.open(img_dir / name).convert("RGB")
        W, H = im.size
        objs = []
        for h in hints[name]:
            x1, y1, x2, y2 = h["box"]
            if h.get("line"):
                # thin blade SAM cannot isolate: mask = the traced centre line, h["width"] px wide
                mask = line_mask(h["line"], h.get("width", 6), W, H)
            else:
                pos = h.get("pos") or [[(x1 + x2) / 2, (y1 + y2) / 2]]
                neg = h.get("neg") or []
                kw = dict(points=[pos + neg], labels=[[1] * len(pos) + [0] * len(neg)], verbose=False)
                if not h.get("nobox"):
                    kw["bboxes"] = [h["box"]]
                r = model(str(img_dir / name), **kw)[0]
                mask = r.masks.data[0].cpu().numpy().astype(bool) if r.masks is not None else None
            # keep the cut-out inside the hint box (plus a margin): SAM sometimes spills into the arm
            mg = a.margin
            clip = np.zeros((H, W), bool)
            clip[max(0, int(y1 - mg)):int(y2 + mg) + 1, max(0, int(x1 - mg)):int(x2 + mg) + 1] = True
            if mask is None or (mask & clip).sum() < 20:
                print(f"  ! {name} {h['cls']}: SAM found nothing, using hint box")
                mask = np.zeros((H, W), bool)
                mask[int(y1):int(y2) + 1, int(x1):int(x2) + 1] = True
            mask &= clip
            ys, xs = np.nonzero(mask)
            pix = np.stack([xs, ys], 1).astype(float)
            box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
            if h["cls"] == "sword" and not h.get("line"):
                ax0 = principal_axis(pix)
                width = mask.sum() / (np.ptp(pix @ ax0) + 1)
                if width > a.max_width:
                    # SAM spilled sideways: keep only the part within a corridor along the hinted blade
                    corridor = line_mask(blade_line(h), a.corridor, W, H)
                    trimmed = mask & corridor
                    if trimmed.sum() >= 20:
                        mask = trimmed
                        ys, xs = np.nonzero(mask)
                        pix = np.stack([xs, ys], 1).astype(float)
                        box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
                    print(f"  ~ {name} sword #{len(objs)}: mask {width:.0f}px wide, trimmed to the blade corridor")
            o = dict(cls=h["cls"], box=box, area=int(mask.sum()))
            if h["cls"] in WEAPONS:
                hd = np.array(h["handle"], float) if h.get("handle") else None
                st = np.array(h["strike"], float) if h.get("strike") else None
                ax = principal_axis(pix)
                if hd is not None and st is not None:
                    ax = unit(st - hd) if np.dot(ax, st - hd) == 0 else (ax if np.dot(ax, st - hd) > 0 else -ax)
                elif st is not None and np.dot(ax, st - pix.mean(0)) < 0:
                    ax = -ax
                elif hd is not None and np.dot(ax, pix.mean(0) - hd) < 0:
                    ax = -ax
                o["handle_end"] = edge_centre(pix, -ax, hd, a.radius).tolist() if hd is not None else None
                o["striking_end"] = edge_centre(pix, ax, st, a.radius).tolist() if st is not None else None
            else:  # shield
                tp = np.array(h["top"], float) if h.get("top") else None
                if tp is None:
                    o.update(top=None, bottom=None, left=None, right=None)
                else:
                    ax = principal_axis(pix)
                    c = pix.mean(0)
                    if np.dot(ax, tp - c) < 0:
                        ax = -ax                      # ax points to the shield's top
                    left = np.array([ax[1], -ax[0]])  # 90 deg CCW from up, in image coords
                    o["top"] = edge_centre(pix, ax, None, 0).tolist()
                    o["bottom"] = edge_centre(pix, -ax, None, 0).tolist()
                    o["left"] = edge_centre(pix, left, None, 0).tolist()
                    o["right"] = edge_centre(pix, -left, None, 0).tolist()
            for k in h.get("outside") or []:
                o[ALIAS.get(k, k)] = None
            for k, p in (h.get("occluded") or {}).items():
                n = ALIAS.get(k, k)          # hidden but obvious: placed as given, not snapped to the cut-out
                o[n] = [float(p[0]), float(p[1])]
                o.setdefault("occluded", []).append(n)
            objs.append((o, mask))
        result[name] = [o for o, _ in objs]
        draw_overlay(im, objs).save(out / "overlays" / name)
        print(f"[segment] {name}: {len(objs)} objects")
    (out / "prelabels.json").write_text(json.dumps(result, indent=1))
    print(f"[segment] -> {out / 'prelabels.json'}, overlays in {out / 'overlays'}")


def draw_overlay(im: Image.Image, objs) -> Image.Image:
    base = im.copy()
    tint = Image.new("RGBA", im.size, (0, 0, 0, 0))
    td = ImageDraw.Draw(tint)
    for o, mask in objs:
        col = COLORS[o["cls"]]
        ys, xs = np.nonzero(mask)
        m = Image.fromarray((mask * 90).astype(np.uint8))
        tint.paste(Image.new("RGBA", im.size, col + (255,)), (0, 0), m)
    base = Image.alpha_composite(base.convert("RGBA"), tint).convert("RGB")
    d = ImageDraw.Draw(base)
    for o, _ in objs:
        col = COLORS[o["cls"]]
        d.rectangle(o["box"], outline=col, width=1)
        d.text((o["box"][0] + 2, o["box"][1] - 11), o["cls"], fill=col)
        names = ("handle_end", "striking_end") if o["cls"] in WEAPONS else ("top", "bottom", "left", "right")
        pc = {"handle_end": (255, 225, 25), "striking_end": (240, 50, 230), "top": (255, 225, 25),
              "bottom": (240, 50, 230), "left": (191, 239, 69), "right": (250, 190, 212)}
        for n in names:
            p = o.get(n)
            if p:
                x, y = p
                hollow = n in o.get("occluded", [])
                d.ellipse([x - 4, y - 4, x + 4, y + 4], outline=pc[n] if hollow else (0, 0, 0),
                          fill=None if hollow else pc[n], width=2 if hollow else 1)
    return base


# ---------------------------------------------------------------- push to CVAT

def api(host, auth, method, path, body=None):
    req = urllib.request.Request(host + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": "Basic " + auth, "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        txt = r.read()
        return json.loads(txt) if txt else None


def cmd_push(a) -> None:
    user, _, pw = a.login.partition(":")
    if a.login_file:
        kv = dict(l.split(": ", 1) for l in Path(a.login_file).read_text().splitlines() if ": " in l)
        user, pw = kv["user"], kv["password"]
    auth = base64.b64encode(f"{user}:{pw}".encode()).decode()
    pre = json.loads(Path(a.prelabels).read_text())
    task = api(a.host, auth, "GET", f"/api/tasks/{a.task}")
    meta = api(a.host, auth, "GET", f"/api/tasks/{a.task}/data/meta")
    frame_of = {Path(f["name"]).name: i for i, f in enumerate(meta["frames"])}
    labels = api(a.host, auth, "GET", f"/api/labels?task_id={a.task}&page_size=100")["results"]
    lid = {l["name"]: l["id"] for l in labels}
    sub = {}
    for l in labels:
        if l["type"] == "skeleton":
            full = api(a.host, auth, "GET", f"/api/labels/{l['id']}")
            sub[l["name"]] = {s["name"]: s["id"] for s in full["sublabels"]}
    shapes, g = [], 0
    for name, objs in pre.items():
        fr = frame_of[name]
        for o in objs:
            g += 1
            shapes.append(dict(type="rectangle", label_id=lid[o["cls"]], frame=fr, group=g, source=a.source,
                               points=o["box"], occluded=False, z_order=0, attributes=[]))
            sk = "shield_points" if o["cls"] == "shield" else "weapon_points"
            pts = ("top", "bottom", "left", "right") if sk == "shield_points" else ("handle_end", "striking_end")
            cx, cy = (o["box"][0] + o["box"][2]) / 2, (o["box"][1] + o["box"][3]) / 2
            els = [dict(type="points", label_id=sub[sk][n], frame=fr, source=a.source,
                        occluded=n in o.get("occluded", []),
                        outside=o.get(n) is None, points=list(o[n]) if o.get(n) else [cx, cy], attributes=[])
                   for n in pts]
            shapes.append(dict(type="skeleton", label_id=lid[sk], frame=fr, group=g, source=a.source,
                               points=[], occluded=False, z_order=0, attributes=[], elements=els))
    jobs = api(a.host, auth, "GET", f"/api/jobs?task_id={a.task}")["results"]
    if len(jobs) != 1:
        sys.exit(f"expected one job in task {a.task}, found {len(jobs)}")
    api(a.host, auth, "PUT", f"/api/jobs/{jobs[0]['id']}/annotations",
        dict(version=0, tags=[], shapes=shapes, tracks=[]))
    print(f"[push] task {task['id']}: replaced annotations with {g} weapons ({len(shapes)} shapes) "
          f"on {len(pre)} frames")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sg = sub.add_parser("segment")
    sg.add_argument("--hints", required=True)
    sg.add_argument("--images", required=True)
    sg.add_argument("--out", required=True)
    sg.add_argument("--model", default=str(Path.home() / ".cache/sam/sam2.1_b.pt"),
                    help="SAM 2 weights (downloaded on first use if the file name is unknown)")
    sg.add_argument("--margin", type=float, default=12, help="px the cut-out may extend past the hint box")
    sg.add_argument("--radius", type=float, default=45, help="px around an end hint to look for the extreme")
    sg.add_argument("--max-width", type=float, default=16,
                    help="a sword mask wider than this (px) on average is treated as spilled")
    sg.add_argument("--corridor", type=float, default=22,
                    help="px width of the corridor along the hinted blade that a spilled mask is trimmed to")
    sg.add_argument("--only", help="comma-separated image names to (re)run")
    ps = sub.add_parser("push")
    ps.add_argument("--prelabels", required=True)
    ps.add_argument("--task", type=int, required=True)
    ps.add_argument("--host", default="http://localhost:8080")
    ps.add_argument("--login", default="", help="user:password")
    ps.add_argument("--login-file", help="file with 'user: ...' and 'password: ...' lines")
    ps.add_argument("--source", default="semi-auto")
    a = ap.parse_args()
    {"segment": cmd_segment, "push": cmd_push}[a.cmd](a)


if __name__ == "__main__":
    main()
