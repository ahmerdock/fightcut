# Weapon labelling spec (pilot: octopi_weapons_pilot_50)

Rules agreed with the client for the 50-frame pilot. Apply them to every frame.

## Frames

- Source: Portland OR, June 27 2026, rounds-only videos. Camera 1 = `camA`, Camera 2 = `camB` (synced).
- Fights 1–5. For each fight and camera, 5 frames at 10, 30, 50, 70 and 90% of elapsed fight time = 50 PNGs.
- If the exact frame is motion-blurred, the nearest sharp frame within 0.5 s is used.
- File name: `fight01_camA_001250ms.png`. The ms value is the frame's time in the processed video.
- Extract with:
  ```bash
  .venv/bin/python framegrab.py extract \
    --cam "A=work/<camera 1>/<camera 1> - rounds only.mp4" \
    --cam "B=work/<camera 2>/<camera 2> - rounds only.mp4" \
    --cutlist "work/<camera 1>/cutlist.csv" --fights 1-5 --out work/octopi_weapons_pilot_50
  ```
- If a camera's video was rendered from its own cut list (e.g. Sioux Falls Camera 2 covers only fights 5–8), give each camera its cut list. Fights missing from a camera's cut list are skipped for that camera:
  ```bash
  .venv/bin/python framegrab.py extract \
    --cam "A=work/<camera 1>/<camera 1> - rounds only.mp4" \
    --cam "B=work/<camera 2>/<camera 2> - rounds only.mp4" \
    --cutlist "A=work/<camera 1>/cutlist.csv" --cutlist "B=work/<camera 2>/cutlist.csv" \
    --fights 1-8 --out work/octopi_weapons_sioux_falls_60
  ```
  Check with `--expect <frame count>` when the total is not 50.

## Labels (`cvat_labels.json`)

| Label | Type | Grouped with |
|---|---|---|
| `sword`, `axe`, `mace` | rectangle | a `weapon_points` skeleton: `handle_end`, `striking_end` |
| `shield` | rectangle | a `shield_points` skeleton: `top`, `bottom`, `left`, `right` (bones top–bottom, left–right) |

Select each rectangle and its skeleton and press **G** to group them.

## What to label

- Every identifiable sword, axe, mace and shield, including dropped ones.
- Not labelled: other weapon types (polearms, spears, …) and the referee's yellow-and-black stick.
- If no weapon is identifiable, leave the frame empty.

## Boxes

- One rectangle per weapon. Zoom in and place each side against the outermost visible part, with zero intentional padding.
- Include the guard or the axe head.
- If the fighter hides the middle, one box still covers both visible ends.

## Points

- `handle_end`: the centre of the handle's butt face. On a sword, the outer end of the pommel.
- `striking_end`: the sword tip, the midpoint of the axe's cutting edge, or the farthest point of the mace head along the shaft's centreline.
- Shield `top`/`bottom`: the ends of the long axis. `top` is always the shield's real top, so on an upside-down shield `top` sits below `bottom`.
- Heater shields: the point is `bottom`, and the middle of the opposite (flat) edge is `top`.
- Shield `left`/`right`: the ends of the widest line across. With the shield turned upright in your head, `left` is the one on the left as seen in the frame.
- Round shields: use a fixed marking such as a crest to find the top. A round buckler with no markings keeps its rectangle with all four points Outside.
- If the middle is hidden but both ends are visible, place both points normally.

Every point is in one of three states:

| State | When | In CVAT |
|---|---|---|
| Visible | The end can be seen | Placed |
| Occluded | Hidden (by a hand, the other fighter, a shield) but its position is obvious, e.g. a pommel just past the gauntlet in line with the blade | Placed, **occluded** toggled (Q) |
| Outside | Cannot be located, or out of frame | **Outside** |

## Delivery

Export the task as **CVAT for images 1.1** and unzip `annotations.xml`, then:

```bash
.venv/bin/python framegrab.py check --xml path/to/annotations.xml \
  --images work/octopi_weapons_pilot_50/images --zip work/octopi_weapons_pilot_50/octopi_weapons_pilot_50.zip
```

This checks there are 50 images, the file names match, every box is grouped with the right skeleton, and no unexpected labels are used. It then builds `octopi_weapons_pilot_50.zip` with `annotations.xml` and `images/`.
