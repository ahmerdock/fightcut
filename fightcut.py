#!/usr/bin/env python3
"""fightcut — cut an Armored MMA event broadcast down to just the fighting rounds.

Two steps, with a human check in between:

  1. analyze:  python fightcut.py analyze EVENT.mp4
       -> work/<name>/cutlist.csv      (one row per detected round; edit freely)
       -> work/<name>/review/*.png     (frames around every cut, to eyeball)

  2. render:   python fightcut.py render EVENT.mp4
       -> work/<name>/<name> - rounds only.mp4

How detection works (no AI model, just the broadcast's own grammar):
  * Live rounds are shot with close, handheld, constantly-cut cameras -> sustained
    high frame-to-frame motion.
  * Between rounds the director parks on one locked-off wide shot of the cage ->
    near-identical frames that recur all event long ("wide shot").
  * Entrances are dark stage shots; intros/interviews are static close shots.
  A round = a long stretch of sustained motion that is not the wide shot and not
  dark. Rounds of one fight are always separated by a wide-shot break; a new fight
  starts after an entrance (dark) or a long static stretch (interview/intros).
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

FPS = 2                  # analysis sample rate (frames per second)
TW, TH = 64, 36          # thumbnail size used for analysis

# Tunables (seconds unless stated). Defaults were fitted on the Portland 2026 event.
P = dict(
    motion_th=10.0,      # median frame-diff above this = "active"
    motion_win=10.0,     # window for the median motion filter
    min_active=30.0,     # a round needs at least this much active footage
    min_clangs=6,        # ... and at least this many weapon clangs
    wide_sim=0.6,        # correlation with a wide-shot reference to count as "wide"
    dark_frac=0.85,      # fraction of dark pixels to count as an entrance shot
    stoppage=45.0,       # wide-shot gaps shorter than this are ref stoppages (merged)
    min_round=20.0,      # drop anything shorter than this
    intro_max=60.0,      # short pieces running straight into the next segment = intros
    break_wide=0.3,      # a between-rounds gap is at least this fraction wide shot
    new_fight_gap=240.0, # a gap longer than this between rounds => new fight
    tail=45.0,           # max seconds kept after the last action before the cut-out
    walkin=15.0,         # if the first clang burst is this late, the start is a walk-in
    burst_n=3,           # >= this many clangs ...
    burst_win=6.0,       # ... within this many seconds = fighting has started
    burst_lead=2.0,      # start the round this long before the first clang of the burst
    # "Lull" = a break shown without cutting to the wide shot (the director stays on a
    # cage-side camera while the corners work). Still picture + no clangs, sustained.
    lull_win=10.0,       # look this many seconds either side for clangs
    lull_clangs=1,       # at most this many clangs in that window = quiet
    lull_motion=6.0,     # median motion below this = still picture
    lull_min=20.0,       # a lull must last at least this long to count as a break
)


# --------------------------------------------------------------------------- utils
def fmt(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def parse_t(s: str) -> float:
    parts = [float(p) for p in s.strip().split(":")]
    t = 0.0
    for p in parts:
        t = t * 60 + p
    return t


def runs(mask: np.ndarray) -> list[list[int]]:
    """[start, end) index pairs of True runs."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append([i, j])
            i = j
        else:
            i += 1
    return out


def workdir(video: Path) -> Path:
    d = Path("work") / video.stem
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------- feature pass
def extract_thumbs(video: Path, out: Path) -> np.ndarray:
    if out.exists():
        return np.load(out)
    print(f"[analyze] decoding {video.name} at {FPS} fps (one-time, cached)…", flush=True)
    cmd = ["ffmpeg", "-nostdin", "-v", "quiet", "-i", str(video),
           "-vf", f"fps={FPS},scale={TW}:{TH}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size, frames = TW * TH * 3, []
    while True:
        b = p.stdout.read(size)
        if len(b) < size:
            break
        frames.append(np.frombuffer(b, np.uint8).reshape(TH, TW, 3))
        if len(frames) % (FPS * 600) == 0:
            print(f"  … {len(frames) // FPS // 60} min", flush=True)
    p.wait()
    arr = np.array(frames)
    np.save(out, arr)
    return arr


def clang_times(video: Path, out: Path) -> np.ndarray:
    """Times (s) of weapon-on-armour clangs: sharp broadband transients in 5–8 kHz.
    Processed in 10-minute chunks so a 3-hour event fits comfortably in memory."""
    if out.exists():
        return np.load(out)
    from scipy.io import wavfile
    from scipy.signal import find_peaks, stft
    wav = out.with_suffix(".wav")
    print("[analyze] extracting audio (one-time, cached)…", flush=True)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "quiet", "-y", "-i", str(video), "-vn",
                    "-ac", "1", "-ar", "16000", str(wav)], check=False)
    sr, a = wavfile.read(wav, mmap=True)
    chunk, times = sr * 600, []
    for off in range(0, len(a), chunk):
        x = np.asarray(a[max(0, off - sr * 2):off + chunk], dtype=np.float32) / 32768
        f, t, Z = stft(x, sr, nperseg=256, noverlap=128)
        lb = np.log(np.abs(Z[(f > 5000) & (f < 7900)]).mean(0) + 1e-7)
        score = lb - median_filter(lb, 125)
        pk, _ = find_peaks(score, height=2.0, distance=8)
        tt = t[pk] + max(0, off - sr * 2) / sr
        times.extend(tt[tt >= off / sr])
    arr = np.array(sorted(times))
    np.save(out, arr)
    wav.unlink(missing_ok=True)
    return arr


def features(thumbs: np.ndarray) -> dict:
    rgb = thumbs.astype(np.float32)
    g = rgb.mean(3)
    lum = g
    sat = rgb.max(3) - rgb.min(3)
    lower = slice(TH // 3, TH)
    white = (lum[:, lower] > 170) & (sat[:, lower] < 40)
    red = (rgb[:, lower, :, 0] > 150) & (rgb[:, lower, :, 1] < 90)
    mat = (white | red).mean((1, 2))
    dark = (lum < 35).mean((1, 2))
    diff = np.abs(np.diff(g, axis=0, prepend=g[:1])).mean((1, 2))

    # z-normalised grey frames for shot matching
    z = g.reshape(len(g), -1)
    z = (z - z.mean(1, keepdims=True)) / (z.std(1, keepdims=True) + 1e-6)

    k = int(P["motion_win"] * FPS) | 1
    mm = median_filter(diff, k)
    wide_sim = wide_similarity(z, mm, mat, dark)
    return dict(mm=mm, diff=diff, mat=mat, dark=median_filter(dark, 9),
                sim=median_filter(wide_sim, 9), raw_sim=wide_sim)


def wide_similarity(z, mm, mat, dark) -> np.ndarray:
    """Find the recurring locked-off wide shot automatically and score every frame
    against it. Candidates: static frames where the mat is a small ellipse with
    darkness around it. The director re-frames that camera during the night, so one
    reference is picked per part of the event (the candidate that recurs most there)."""
    n = len(z)
    cand_all = np.where((mm < 4) & (mat < 0.08) & (dark > 0.35) & (dark < 0.85))[0]
    if len(cand_all) == 0:
        print("[analyze] warning: no wide-shot candidates found", file=sys.stderr)
        return np.zeros(n)
    refs = []
    blocks = max(1, int(n / FPS / 1800))                 # one block per ~30 min
    for blk in np.array_split(np.arange(n), blocks):
        cand = cand_all[(cand_all >= blk[0]) & (cand_all <= blk[-1])]
        if len(cand) < 20:
            continue
        cand = cand[:: max(1, len(cand) // 200)]
        score = ((z[cand] @ z[cand].T) / z.shape[1] > 0.8).sum(1)
        refs.append(cand[np.argmax(score)])
    if not refs:
        refs = [cand_all[0]]
    print("[analyze] wide-shot references at " + ", ".join(fmt(r / FPS) for r in refs))
    return np.max([z @ z[r] / z.shape[1] for r in refs], axis=0)


# ----------------------------------------------------------------------- detection
def lulls(mm: np.ndarray, clangs: np.ndarray) -> np.ndarray:
    """Frames inside a sustained still + silent stretch (a break with no wide shot)."""
    n = len(mm)
    per = np.bincount(np.clip((clangs * FPS).astype(int), 0, n - 1), minlength=n)
    w = int(P["lull_win"] * FPS)
    near = np.convolve(per, np.ones(2 * w + 1), mode="same")
    m = (near <= P["lull_clangs"]) & (mm < P["lull_motion"])
    m = median_filter(m.astype(np.uint8), 9).astype(bool)
    out = np.zeros(n, bool)
    for a, b in runs(m):
        if (b - a) / FPS >= P["lull_min"]:
            out[a:b] = True
    return out


def detect(F: dict) -> list[dict]:
    """Rounds = stretches between break shots (wide / dark) that contain sustained
    action AND weapon clangs. Edges come from the director's cuts to/from the wide
    shot, so clinches and ground work (low motion) stay inside the round."""
    mm, sim, dark, mat = F["mm"], F["sim"], F["dark"], F["mat"]
    clangs = F.get("clangs", np.array([]))
    is_dark = dark > P["dark_frac"]
    static_wide = (mm < 3) & (mat < 0.05) & (dark > 0.45) & (dark < 0.8)
    wide = median_filter(((sim > P["wide_sim"]) | static_wide).astype(np.uint8), 9).astype(bool)
    lull = lulls(mm, clangs)                             # breaks that never cut to wide
    close = ~wide & ~lull & ~is_dark

    cands = []
    for a, b in runs(close):
        t0, t1 = a / FPS, b / FPS
        act = np.where(mm[a:b] > P["motion_th"])[0]
        if len(act) / FPS < P["min_active"]:
            continue
        c = clangs[(clangs >= t0) & (clangs < t1)]
        if len(c) < P["min_clangs"]:
            continue                                     # interview / intros / crowd
        # start: cut-in from the wide shot, unless a long walk-in precedes the first
        # burst of clangs (round 1 after an entrance) -> anchor on the burst
        start = t0
        for x in c:
            if ((c >= x) & (c < x + P["burst_win"])).sum() >= P["burst_n"]:
                if x - t0 > P["walkin"]:
                    start = x - P["burst_lead"]
                break
        # end: cut-out to the wide shot, but don't run on forever after the action
        last_action = max((a + act[-1]) / FPS, c[-1])
        end = min(t1, last_action + P["tail"])
        cands.append([start, end])

    # Short wide-shot gaps are ref stoppages inside a round, not round breaks.
    merged: list[list[float]] = []
    for s, e in cands:
        if merged and s - merged[-1][1] < P["stoppage"] and \
                not is_dark[int(merged[-1][1] * FPS):int(s * FPS)].any():
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # Broadcast style. Some directors always cut to the wide shot between rounds
    # (then a gap without it is intros/interviews = new fight). Others often stay on
    # a cage-side camera (then a still, silent gap is a round break too, and fights
    # are told apart by the entrance between them).
    def frac(mask, pe, s):
        g = slice(int(pe * FPS), int(s * FPS))
        return mask[g].mean() if s > pe else 1.0

    plain = [(pe, s) for (_, pe), (s, _) in zip(merged, merged[1:])
             if not is_dark[int(pe * FPS):int(s * FPS)].any()]
    wide_style = not plain or np.median([frac(wide, pe, s) for pe, s in plain]) >= 0.5
    brk = wide if wide_style else (wide | lull)
    print(f"[analyze] broadcast style: {'wide shot between rounds' if wide_style else 'cage-side breaks (no wide shot)'}")

    def gap_info(pe: float, s: float) -> tuple[float, bool]:
        return frac(brk, pe, s), bool(is_dark[int(pe * FPS):int(s * FPS)].any())

    # Walk-outs / intros: a short piece that runs into the next segment with no
    # break shot and no entrance between them is pre-fight footage, not a round.
    kept = []
    for i, (s, e) in enumerate(merged):
        if e - s < P["min_round"]:
            continue
        if i + 1 < len(merged) and e - s < P["intro_max"]:
            wf, dk = gap_info(e, merged[i + 1][0])
            if wf < P["break_wide"] and not dk and merged[i + 1][0] - e < P["new_fight_gap"]:
                continue
        kept.append([s, e])

    # Fights: between rounds the director sits on the wide shot; between fights
    # there are interviews, intros or an entrance instead.
    out, fight, rnd = [], 0, 0
    for i, (s, e) in enumerate(kept):
        if i == 0:
            fight, rnd = 1, 1
        else:
            ps, pe = kept[i - 1]
            wf, dk = gap_info(pe, s)
            if dk or s - pe > P["new_fight_gap"] or (wide_style and wf < P["break_wide"]):
                fight, rnd = fight + 1, 1
            else:
                rnd += 1
        note = []
        if e - s > 200:
            note.append("long")
        if rnd > 3:
            note.append("round>3 (5-round fight?)")
        out.append(dict(fight=fight, round=rnd, start=s, end=e, note=" ".join(note)))
    return out


# ---------------------------------------------------------------------- review pngs
def review_sheets(video: Path, rows: list[dict], outdir: Path) -> None:
    outdir.mkdir(exist_ok=True)
    for old in outdir.glob("*.png"):
        old.unlink()
    for r in rows:
        name = outdir / f"F{r['fight']}R{r['round']}.png"
        # 6 frames around start, 6 frames around end (2 s apart)
        ts = [r["start"] + d for d in (-6, -4, -2, 0, 2, 4)] + [r["end"] + d for d in (-4, -2, 0, 2, 4, 6)]
        tiles = []
        for i, t in enumerate(ts):
            tile = outdir / f".t{i}.png"
            subprocess.run(["ffmpeg", "-nostdin", "-v", "quiet", "-y", "-ss", f"{max(0, t):.2f}", "-i", str(video),
                            "-frames:v", "1", "-vf", "scale=240:-1", str(tile)])
            tiles.append(tile)
        inputs = sum((["-i", str(t)] for t in tiles), [])
        subprocess.run(["ffmpeg", "-nostdin", "-v", "quiet", "-y", *inputs, "-filter_complex",
                        "".join(f"[{i}]" for i in range(12)) + "xstack=inputs=12:grid=6x2",
                        str(name)])
        for t in tiles:
            t.unlink(missing_ok=True)


# ------------------------------------------------------------------------ commands
def cmd_analyze(a) -> None:
    video = Path(a.video)
    wd = workdir(video)
    st = video.stat()
    stamp, stamp_file = f"{st.st_size} {int(st.st_mtime)}", wd / "source.txt"
    if not stamp_file.exists() or stamp_file.read_text() != stamp:
        for c in ("thumbs.npy", "clangs.npy"):      # video changed (e.g. re-downloaded)
            (wd / c).unlink(missing_ok=True)
        stamp_file.write_text(stamp)
    thumbs = extract_thumbs(video, wd / "thumbs.npy")
    print(f"[analyze] {len(thumbs) / FPS / 60:.1f} min of decodable video")
    F = features(thumbs)
    F["clangs"] = clang_times(video, wd / "clangs.npy")
    rows = detect(F)
    csv_path = wd / "cutlist.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fight", "round", "start", "end", "duration_s", "keep", "note"])
        for r in rows:
            w.writerow([r["fight"], r["round"], fmt(r["start"]), fmt(r["end"]),
                        round(r["end"] - r["start"]), "yes", r["note"]])
    print(f"\n{'fight':>5} {'rnd':>3}  {'start':>11}  {'end':>11}  dur")
    for r in rows:
        print(f"{r['fight']:>5} {r['round']:>3}  {fmt(r['start'])}  {fmt(r['end'])}  {r['end'] - r['start']:4.0f}s {r['note']}")
    print(f"\n[analyze] cut list: {csv_path}")
    if not a.no_review:
        review_sheets(video, rows, wd / "review")
        print(f"[analyze] review sheets: {wd / 'review'}  (top row = around start, bottom = around end)")
    print("[analyze] edit the CSV if needed (change times, set keep=no), then run `render`.")


def cmd_render(a) -> None:
    video = Path(a.video)
    wd = workdir(video)
    csv_path = Path(a.cutlist) if a.cutlist else wd / "cutlist.csv"
    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f) if r.get("keep", "yes").strip().lower() not in ("no", "n", "0")]
    if not rows:
        sys.exit("nothing to render")
    seg_dir = wd / "segments"
    seg_dir.mkdir(exist_ok=True)
    vcodec = ["-c:v", "h264_videotoolbox", "-b:v", a.bitrate] if a.encoder == "hw" else \
             ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]
    listing = []
    total = 0.0
    for i, r in enumerate(rows):
        s = max(0.0, parse_t(r["start"]) - a.pad)
        e = parse_t(r["end"]) + a.pad
        total += e - s
        seg = seg_dir / f"{i:03d}_F{r['fight']}R{r['round']}.mp4"
        print(f"[render] F{r['fight']} R{r['round']}  {fmt(s)} -> {fmt(e)}", flush=True)
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{s:.3f}", "-i", str(video),
                        "-t", f"{e - s:.3f}", *vcodec, "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                        "-movflags", "+faststart", str(seg)], check=True)
        listing.append(f"file '{seg.resolve()}'")
    lst = seg_dir / "concat.txt"
    lst.write_text("\n".join(listing) + "\n")
    out = Path(a.output) if a.output else wd / f"{video.stem} - rounds only.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
    if not a.keep_segments:
        for f in seg_dir.iterdir():
            f.unlink()
        seg_dir.rmdir()
    print(f"[render] done: {out}  ({total / 60:.1f} min from {len(rows)} rounds)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    an = sub.add_parser("analyze", help="detect rounds, write cutlist.csv + review sheets")
    an.add_argument("video")
    an.add_argument("--no-review", action="store_true", help="skip review contact sheets")
    rn = sub.add_parser("render", help="cut and join the rounds from cutlist.csv")
    rn.add_argument("video")
    rn.add_argument("--cutlist", help="path to cutlist.csv (default: work/<name>/cutlist.csv)")
    rn.add_argument("-o", "--output")
    rn.add_argument("--pad", type=float, default=1.0, help="seconds of padding on each side (default 1)")
    rn.add_argument("--encoder", choices=["hw", "x264"], default="hw",
                    help="hw = Apple VideoToolbox (fast), x264 = slower, smaller files")
    rn.add_argument("--bitrate", default="10M", help="bitrate for hw encoder (default 10M)")
    rn.add_argument("--keep-segments", action="store_true", help="keep per-round clips in work/<name>/segments")
    a = ap.parse_args()
    {"analyze": cmd_analyze, "render": cmd_render}[a.cmd](a)


if __name__ == "__main__":
    main()
