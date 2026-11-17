#!/usr/bin/env python3
"""
framegrab — pull labelling frames out of a rendered "rounds only" video.

  extract  For each fight, grab frames at given percentages of the fight's
           elapsed time in the processed video, from one or more synced
           camera angles. If the exact frame is motion-blurred, the nearest
           sharp frame within --search seconds is used instead.
             -> <out>/images/fight01_camA_001250ms.png   (ms = time in the processed video)
             -> <out>/manifest.csv                       (target vs chosen time, sharpness)

  check    Validate a "CVAT for images 1.1" export against the extracted
           images, then build the delivery ZIP (annotations.xml + images/).

Fight positions come from the cut list the video was rendered with: render
cuts each kept row with --pad seconds either side and concatenates them in
order, so a fight's span in the output is the sum of its segment lengths.
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
from scipy.ndimage import laplace

from fightcut import parse_t

PW, PH = 640, 360          # analysis size for the sharpness score


# ---------------------------------------------------------------- extract

def fight_spans(cutlist: Path, pad: float) -> dict[int, tuple[float, float]]:
    """fight number -> (start, end) seconds in the rendered video."""
    with open(cutlist) as f:
        rows = [r for r in csv.DictReader(f) if r.get("keep", "yes").strip().lower() not in ("no", "n", "0")]
    spans: dict[int, tuple[float, float]] = {}
    t = 0.0
    for r in rows:
        s = max(0.0, parse_t(r["start"]) - pad)
        e = parse_t(r["end"]) + pad
        f = int(r["fight"])
        a, _ = spans.get(f, (t, t))
        spans[f] = (a, t + (e - s))
        t += e - s
    return spans


def probe(video: Path) -> tuple[float, float]:
    """(frames per second, container start_time)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=avg_frame_rate:format=start_time", "-of", "default=nw=1", str(video)],
                         capture_output=True, text=True, check=True).stdout
    kv = dict(l.split("=", 1) for l in out.split())
    n, d = kv["avg_frame_rate"].split("/")
    return float(n) / float(d), float(kv.get("start_time", 0) or 0)


def decode_window(video: Path, t0: float, n: int, start: float) -> tuple[np.ndarray, np.ndarray]:
    """Decode n frames from position t0 as small greyscale images.
    Returns (positions in seconds from the start of the file, frames). -ss seeks by
    position, while pts carry the container start_time, so subtract it back out."""
    p = subprocess.run(["ffmpeg", "-nostdin", "-v", "info", "-copyts", "-ss", f"{t0:.4f}", "-i", str(video),
                        "-frames:v", str(n), "-an",
                        "-vf", f"scale={PW}:{PH},format=gray,showinfo",
                        "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                       capture_output=True, check=True)
    frames = np.frombuffer(p.stdout, np.uint8).reshape(-1, PH, PW)
    pts = np.array([float(x) for x in re.findall(rb"pts_time:([0-9.]+)", p.stderr)]) - start
    m = min(len(pts), len(frames))
    return pts[:m], frames[:m]


def sharpness(g: np.ndarray) -> float:
    return float(laplace(g.astype(np.float32)).var())


def small_grey(png: Path) -> np.ndarray:
    p = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(png),
                        "-vf", f"scale={PW}:{PH},format=gray", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(PH, PW)


def cmd_extract(a) -> None:
    cams = {}
    for c in a.cam:
        name, _, path = c.partition("=")
        if not path or not Path(path).exists():
            sys.exit(f"bad --cam {c!r} (expected NAME=path/to/video.mp4)")
        cams[name] = Path(path)
    lo, _, hi = a.fights.partition("-")
    fights = list(range(int(lo), int(hi or lo) + 1))
    pcts = [float(x) for x in a.pcts.split(",")]
    spans = fight_spans(Path(a.cutlist), a.pad)
    missing = [f for f in fights if f not in spans]
    if missing:
        sys.exit(f"fights not in cut list: {missing}")

    out = Path(a.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for f in fights:
        f0, f1 = spans[f]
        print(f"[extract] fight {f}: {f0:.2f}–{f1:.2f}s in processed video ({f1 - f0:.1f}s)")
        for p in pcts:
            target = f0 + p / 100 * (f1 - f0)
            for cam, video in cams.items():
                fps, start = probe(video)
                w0 = max(f0, target - a.search)
                n = int(round((min(f1, target + a.search) - w0) * fps)) + 1
                pts, frames = decode_window(video, w0, n, start)
                sc = np.array([sharpness(g) for g in frames])
                ok = np.flatnonzero(sc >= a.sharp * sc.max())
                i = int(ok[np.argmin(np.abs(pts[ok] - target))])
                exact = int(np.argmin(np.abs(pts - target)))
                t = float(pts[i])
                name = f"fight{f:02d}_cam{cam}_{round(t * 1000):06d}ms.png"
                png = img_dir / name
                subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{t - 0.25 / fps:.6f}",
                                "-i", str(video), "-frames:v", "1", "-update", "1", "-pix_fmt", "rgb24",
                                str(png)], check=True)
                saved = small_grey(png).astype(int)
                d = {j: float(np.abs(saved - frames[j].astype(int)).mean())
                     for j in (i - 1, i, i + 1) if 0 <= j < len(frames)}
                if d[i] > 3 or min(d, key=d.get) != i:
                    sys.exit(f"{name}: saved frame does not match the chosen frame (diffs {d})")
                manifest.append(dict(file=name, fight=f, cam=cam, pct=p, target_s=f"{target:.3f}",
                                     chosen_s=f"{t:.3f}", shift_ms=round((t - target) * 1000),
                                     sharp=round(sc[i], 1), sharp_exact=round(sc[exact], 1),
                                     sharp_window_max=round(sc.max(), 1)))
                print(f"  {p:>4.0f}% cam{cam}  {name}  shift {manifest[-1]['shift_ms']:+5d} ms"
                      f"  sharp {sc[i]:.0f} (exact {sc[exact]:.0f}, max {sc.max():.0f})")
    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    print(f"[extract] {len(manifest)} PNGs -> {img_dir}")


# ---------------------------------------------------------------- check + zip

def cmd_check(a) -> None:
    img_dir = Path(a.images)
    pngs = sorted(p.name for p in img_dir.glob("*.png"))
    root = ET.parse(a.xml).getroot()
    images = root.findall("image")
    names = [Path(i.get("name")).name for i in images]
    labels = {l.findtext("name") for l in root.iter("label")}
    problems = []
    if len(pngs) != a.expect:
        problems.append(f"{len(pngs)} PNGs in {img_dir}, expected {a.expect}")
    if sorted(names) != pngs:
        problems.append(f"XML image names differ from PNGs: only in XML {sorted(set(names) - set(pngs))}, "
                        f"only in images/ {sorted(set(pngs) - set(names))}")
    allowed = set(a.labels.split(","))
    counts: dict[str, int] = {}
    point_states: dict[str, dict[str, int]] = {}
    for img in images:
        n = img.get("name")
        groups: dict[str, list[str]] = {}
        for sh in img:
            lab = sh.get("label")
            counts[lab] = counts.get(lab, 0) + 1
            if lab not in allowed:
                problems.append(f"{n}: unexpected label {lab!r}")
            g = sh.get("group_id")
            for pt in sh.findall("points"):
                state = "outside" if pt.get("outside") == "1" else "occluded" if pt.get("occluded") == "1" else "visible"
                point_states.setdefault(pt.get("label"), {}).setdefault(state, 0)
                point_states[pt.get("label")][state] += 1
            if sh.tag == "box" or sh.tag == "skeleton":
                if not g:
                    problems.append(f"{n}: {sh.tag} {lab!r} is not grouped")
                else:
                    groups.setdefault(g, []).append(lab)
        for g, labs in groups.items():
            boxes = [l for l in labs if l in ("sword", "axe", "mace", "shield")]
            skels = [l for l in labs if l.endswith("_points")]
            want = "shield_points" if boxes == ["shield"] else "weapon_points"
            if len(boxes) != 1 or skels != [want]:
                problems.append(f"{n}: group {g} has {labs}, expected one box + {want}")
    print(f"[check] {len(images)} images in XML, {len(pngs)} PNGs; shapes per label: {counts}")
    print(f"[check] labels defined in task: {sorted(labels)}")
    print("[check] points:")
    for lab, st in sorted(point_states.items()):
        print(f"   {lab:<13} " + "  ".join(f"{s} {st.get(s, 0):>3}" for s in ("visible", "occluded", "outside")))
    if problems:
        print("[check] PROBLEMS:")
        for p in problems:
            print("   -", p)
        sys.exit(1)
    zp = Path(a.zip)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(a.xml, "annotations.xml")
        for n in pngs:
            z.write(img_dir / n, f"images/{n}")
    print(f"[check] OK -> {zp}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract", help="grab frames at percentages of each fight")
    ex.add_argument("--cam", action="append", required=True, help="NAME=video.mp4 (repeat per camera)")
    ex.add_argument("--cutlist", required=True, help="cut list the videos were rendered from")
    ex.add_argument("--fights", default="1-5", help="fight range, e.g. 1-5")
    ex.add_argument("--pcts", default="10,30,50,70,90", help="percentages of elapsed fight time")
    ex.add_argument("--pad", type=float, default=1.0, help="--pad used at render time (default 1)")
    ex.add_argument("--search", type=float, default=0.5, help="max seconds to move off a blurred frame")
    ex.add_argument("--sharp", type=float, default=0.85,
                    help="frame counts as sharp at this fraction of the sharpest frame in the window")
    ex.add_argument("--out", required=True, help="output folder (images/ and manifest.csv)")
    ck = sub.add_parser("check", help="validate a CVAT for images 1.1 export and build the ZIP")
    ck.add_argument("--xml", required=True, help="annotations.xml from the CVAT export")
    ck.add_argument("--images", required=True, help="folder with the extracted PNGs")
    ck.add_argument("--expect", type=int, default=50)
    ck.add_argument("--labels", default="sword,axe,mace,shield,weapon_points,shield_points")
    ck.add_argument("--zip", required=True, help="output ZIP path")
    a = ap.parse_args()
    {"extract": cmd_extract, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
