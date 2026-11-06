# fightcut

Turns a full Armored MMA event broadcast into a **rounds-only** video. Every round of every fight is played back to back. Commentary, entrances, introductions, breaks between rounds and interviews are all cut out.

```
Fight 1: R1 R2 R3 → Fight 2: R1 R2 R3 → … → Main event: R1 … R5
```

It runs offline on your Mac using Python and ffmpeg. No AI model or internet connection is needed.

## Requirements

- macOS (built and tested on an Apple Silicon M1)
- [ffmpeg](https://ffmpeg.org/): `brew install ffmpeg`
- Python 3.9 or newer

## Setup (once)

```bash
git clone https://github.com/ahmerdock/fightcut.git
cd fightcut
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

Put the event video in this folder, then run the two commands below. Replace `"video name.mp4"` with your file name, and keep the quotes if the name contains spaces.

### 1. Analyze: find the rounds

```bash
.venv/bin/python fightcut.py analyze "video name.mp4"
```

On a 3-hour event this takes about 10 minutes. It prints a table of the fights and rounds it found and writes these files to `work/<video name>/`:

| File | What it is |
|---|---|
| `cutlist.csv` | One row per round: fight, round, start, end, duration, keep, note. **You can edit this file.** |
| `review/F<n>R<n>.png` | Frames around each cut. The top row is around the round start, the bottom row is around the round end. |
| `thumbs.npy`, `clangs.npy` | Cached analysis data. If the video file changes (size or date), these are rebuilt automatically. |

### 2. Check the cut list (optional, recommended)

Open the `review/` images. Each round should start as the referee signals the start and end as the broadcast cuts to the wide shot of the cage. If a row is wrong, fix it in `cutlist.csv`:

- Change `start` or `end`, in the format `H:MM:SS.ss`.
- Set `keep` to `no` to leave a round out.
- Pay particular attention to rows with a `note`, such as `long` or `round>3 (5-round fight?)`.

### 3. Render: build the video

```bash
.venv/bin/python fightcut.py render "video name.mp4"
```

The output is saved as `work/<video name>/<video name> - rounds only.mp4`. Rendering roughly 75 minutes of rounds takes about 15 minutes.

Render options:

| Flag | Default | Meaning |
|---|---|---|
| `--pad 1` | `1` | Seconds of extra footage before and after each round (`0` for tight cuts) |
| `--encoder hw` / `x264` | `hw` | `hw` uses Apple's hardware encoder (fast). `x264` is slower and gives smaller files. |
| `--bitrate 10M` | `10M` | Video bitrate for the `hw` encoder |
| `-o out.mp4` | (see above) | Where to save the output |
| `--cutlist file.csv` | `work/<name>/cutlist.csv` | Use a different cut list |
| `--keep-segments` | off | Keep the individual round clips in `work/<name>/segments/` |

## How it works

The broadcast follows the same pattern all night, and the script uses that pattern:

| Part of the event | What the broadcast shows | Signal the script uses |
|---|---|---|
| **Round** | Close, handheld, quickly switching cameras | Sustained high motion and weapon-on-armour **clangs** in the audio |
| **Break between rounds** | One locked-off **wide shot** of the whole cage | Frames match the wide-shot references, which are found automatically |
| **Entrance** | Dark stage with coloured lights | Mostly dark frames |
| **Introductions and interviews** | Close shots, little action, no clangs | Rejected because they have no clangs |

Rules applied on top of those signals:

- A round runs from the cut away from the wide shot to the cut back to it. Clinches and ground fighting stay in.
- A wide shot shorter than 45 s is a referee stoppage, so the round continues.
- Round 1 has no wide shot before it, so it starts about 2 s before the first burst of clangs. This skips the walk-out and the face-off.
- A new fight starts after an entrance, or after a gap that is mostly introductions or interviews rather than the wide shot.
- Some broadcasts don't cut to the wide shot between rounds. They stay on a cage-side camera while the corners work. A still, silent stretch of 20 s or more (no clangs) also counts as a break. `analyze` prints which style it detected. With cage-side breaks, fights are separated by entrances only.

The detection settings are in the `P = dict(...)` block at the top of `fightcut.py`. Each one has a short comment. They were tuned on the Portland, June 27 2026 event, so a production with a different look may need small adjustments.

## Troubleshooting

- **The video stops partway through, or the minute count in `analyze` is too short.** The download is probably incomplete. A complete 3-hour file at this quality is about 14–15 GB. Download it again and re-run `analyze`.
- **A round is missing or split in two.** Fix the times in `cutlist.csv` and render again. Paste the `analyze` table into an issue or chat so the settings can be adjusted.
- **Rounds run on into post-fight celebrations or interviews.** Applause can register as clangs, so the end of a fight's last round is the row most likely to need trimming. Events with cage-side breaks (`analyze` reports the style) need a closer review than events that cut to the wide shot.
- **A second camera angle of the same event.** If the file has the same duration and audio as one already done, reuse that event's reviewed cut list: `render "camera 2.mp4" --cutlist "work/<camera 1 name>/cutlist.csv"`.
- **The fight or round labels are off but the footage is right.** Labels only affect the CSV and the review image names. The rendered video plays the kept rows in order either way.
