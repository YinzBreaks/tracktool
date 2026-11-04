# Case studies: AARO releases DOW-UAP-PR34 and PR35 (Greece, Oct 2023)

These are two real released clips. They exposed most of the failure modes the tool now handles, and they
show how far each video actually supports the conclusions drawn from it. The project files here hold every
keyframe, mask, exclusion and parameter. Results are recomputed from the video, and the video's SHA-256
is checked.

| File used | SHA-256 |
|---|---|
| `pr34_small.mp4` (960×540, 30 fps container, 177 s) | `8d164a44fabc97da1e8cc057ce1140a7d83827dbb6760b4dfce23d9cb7d38bd3` |
| `DOD_111689022-1920x1080-6000k.mp4` (PR35, 1920×1080, 10 fps, 24.5 s) | `e19937160930b13ae6a363ec6972e96553a3df97836a22a2e09f9bef966e7ce9` |

Reproduce (any OS; on Windows, put each command on one line):

```bash
python -m uapka analyze pr34_small.mp4 --project examples/pr34_project.json --out out/pr34
python -m uapka analyze DOD_111689022-1920x1080-6000k.mp4 --project examples/pr35_project.json --out out/pr35
```

Neither project sets a field of view (the HFOV isn't published), so results are in pixels. Add
`--hfov <degrees>` to get deg/s. Every angular value then scales linearly with that number.

## PR34 (sea background), 0:04–0:48

**What the footage is.** The 30 fps file carries ~10 fps of real content: 62% of frames are
near-duplicates from frame-rate conversion, with a cadence of one new frame per three. There are also
47 blank/dropout frames in this segment, plus burned-in cyan symbology and black redaction blocks. If
duplicates are kept, the object appears to stop and then jump every third frame, which creates fake
acceleration. The tool detects the cadence and uses only new-content frames.

**Tracking.** The object is a 1–2 px bright speck (SNR 20–90). It was tracked in point mode from one
click per object:

- `obj1`: 4.0 s to 15.8 s. At 15.1 s it stops at (266, 324) and fades.
- `obj2`: appears at 15.1 s next to it, at (302, 318), and is tracked to 47.4 s. This matches the
  "2nd object" marks at (320, 317) and (348, 314) in the user-supplied plot. It becomes the long-lived
  track.

**Camera motion: not measurable.** The sea texture at this bitrate (~480 kb/s) decorrelates within about
1 s. Frame to frame, registration reports roughly zero motion. That is indistinguishable from content
frozen by the encoder, and every long-baseline check (42 of 42) is *unverifiable*. The tool therefore
reports **0 of 299 frames with trusted camera compensation**. On-screen paths, speeds and turns describe
the image, not the object.

**On-screen results [MEASURED, not camera-compensated].**

| Object | Median speed | Peak speed |
|---|---|---|
| obj1 | 74 px/s | 191 ± 5 px/s |
| obj2 | 50 px/s | 165 ± 5 px/s |

On-screen heading changes of obj2 occur at 17.2, 23.5, 27.1, 30.2, 32.2, 35.4, 39.1, 40.1, 41.9 and
45.9 s. The plot's annotated turns at 30.4 s and 35.3 s are reproduced (30.2 s, 35.4 s); the angles
depend on the time span compared. Whether any of these are object manoeuvres can't be decided from this
video. The camera's own slewing produces the same on-screen signature. Attributing the turns to the
object requires the static-camera assumption (`--static-camera`), and the tool then labels every
angular value `MEASURED+FOV+ASSUMED-STATIC-CAMERA`.

## PR35 (sky, clouds, then coastline), 0:02.5–0:24.5

**What the footage is.** The display switches from black-hot to white-hot at 2.4 s, with one blank frame
in between, so the analysis starts at 2.5 s. The camera pans roughly 1400 px right and 930 px down while
tilting toward the coast.

**Tracking.** The target ("dot", 3–5 px, SNR 40–360) was tracked from 2.8 to 16.6 s with 13 keyframes.
Frames 8.5–10.3 s are excluded: the dot crosses a cloud there and can't be separated from cloud texture.
Without keyframes the tracker repeatedly preferred cloud highlights, which move with the background and
therefore agree with the camera-motion prediction. A camera-free tracker (or a human) can make the same
mistake. Past ~17 s the coastline fills that part of the image and the dot can't be identified by this
method.

**Camera compensation, validated with control tracks.**

| Background feature (control) | On-screen speed | After compensation (should be 0) |
|---|---|---|
| `cloud` | 102 px/s | 1.8 px/s (98% removed) |
| `coast` | 42 px/s | 1.3 px/s (97% removed) |

Camera motion is trusted for 40 of 105 target frames, mostly after 9 s. Before that the sky has too
little texture, and the closure check flags it.

**Result [MEASURED].** Relative to the background, the dot moves at a median of **103 px/s** along an
almost straight line (the camera-compensated path). On screen it moves at only 23 px/s over the same
frames, because the camera is following it. The on-screen "turns" at 5.6, 7.5 and 14.8 s don't appear
in the compensated path, so they come from the camera's tracking corrections, not from the object.

**Note on the user-supplied PR35 plot.** In this copy of the file, at half scale, the plotted positions
at 8.7 s and 12 s fall on cloud tops. During 15–22 s the plotted object and "2nd object" move parallel to
each other and to coastline features. Parallel on-screen motion of several features is what camera
motion looks like. Running those positions as control tracks would show directly whether they are fixed
to the background.

## Caveats that apply to both

- **Motion "relative to the background" is not the object's own velocity.** If the object is much closer
  than the background, the aircraft's own motion produces apparent motion (parallax). This needs an
  observer-velocity assumption (`--observer-v`), see the README.
- **No range information in either clip.** Speeds in m/s and accelerations in g are available only as
  explicit scenario sweeps.

## GOFAST (Navy ATFLIR, Jan 2015, NAVAIR FOIA release 2020)

**Why this clip.** It is the only released UAP video where the sensor's zoom mode is readable on the
HUD ("NAR, Z 1.0"), the mode has a published field of view, and the HUD also shows slant range,
closure, altitude, Mach, heading and gimbal angles. Every step of the chain from pixels to m/s can
therefore be labelled with what it rests on. Files: `gofast_project.json` (masks, 789 object
positions, FOV estimate, scenario), `gofast_hud_readouts.json` (the HUD values, read by eye),
`gofast_parallax.py` (the per-frame parallax correction).

```bash
python -m uapka analyze "3 - GOFAST.mp4" --project examples/gofast_project.json --out out/gofast
python examples/gofast_parallax.py out/gofast/per_frame.csv out/gofast
```

**What the footage is.** 640×480, 29.97 fps, 34.4 s, no duplicates. The sensor image is a 427×428
inset; the symbology is white (not coloured), so it is masked by brightness per frame in the project.
IR polarity flips to white-hot for ~3 s at 24 s. The pod is not locked for the first 12 s and its
own designator box hides the object there; the analysis uses 12–34 s.

**FOV.** 0.7° from the HUD label through a lookup table, `DERIVED-FOV(hud)`, σ 20%: the 0.7° NAR
figure is quoted by Forecast International and Metabunk; one simulator wiki says 1.5°. Every angular
and speed value below scales with it.

**Measured.** The pod holds the object on screen to 0.35 px/s. Relative to the sea background the
line of sight moves at **0.555°/s** (5th–95th percentile 0.50–0.61) along a straight line; angular
acceleration is at the noise floor (0.014°/s²). Camera motion is trusted for 577 of 672 frames
(median confidence 0.71) after the adaptive-threshold change this clip prompted. Two controls on the
sea are not possible (nothing on the sea is fixed), so the long-baseline closure check is the
verification here: 132 intervals consistent, 47 corrected, 13 unverifiable.

**Range: HUD readout, not a sweep.** RNG 4.4 → 3.4 NM between 13 s and 32 s; a linear fit gives
−96 m/s range rate, matching the HUD closure (Vc 210 → 150 kt). The aircraft's along-LOS speed from
Mach 0.61 and the gimbal angles is 85–143 m/s, so the object also has a small radial component
away from the aircraft (tens of m/s).

**Parallax.** The naive conversion `v = R·ω` gives 57–74 m/s (112–144 kt) *relative to the sea*. The
F/A-18 is doing ~189 m/s true, the LOS is 35–57° off the nose and 22–34° down, so the aircraft's
velocity across the LOS is ~130–160 m/s, and the sea it is registered against is at 14–20 km while
the object is at 6–8 km: they shift by different amounts. With `v_obj = R·ω + v_ac⊥·(1 − R/R_sea)`
(`background_range_m` in the scenario spec; the per-frame version in `gofast_parallax.py` uses the
HUD angles and range at each time):

| Quantity | Value | Rests on |
|---|---|---|
| LOS rate relative to the sea | 0.555°/s | MEASURED + FOV(hud) |
| Naive speed across the LOS (background at infinity) | 67 m/s (130 kt) median | + HUD range |
| Stationary-object prediction of the sea-relative rate | 0.66°/s | HUD Mach, altitude, gimbal angles |
| **Parallax-corrected speed across the LOS** | **~26 m/s (50 kt)** median, 18–36 m/s over ±20% in FOV or sea range | + HUD Mach/altitude/angles, wings level ASSUMED |
| Direction check | sea-relative motion within ~16° of anti-parallel to the aircraft's lateral velocity | — |
| Lateral acceleration | peak ~1 g ± 0.5 g, floor 0.2 g, not significant | all of the above |

The direction test is the point of the exercise: if the object were stationary, its motion relative
to the sea must be exactly opposite the aircraft's lateral velocity in the image. It is within
~16°, which an unknown bank angle alone can produce. The remaining ~26 m/s is consistent with a
wind-blown object, in line with the NASA panel's reading of this clip, and is an order of magnitude
below the naive number. Neither number is a measurement: both rest on the HUD range, and the
corrected one additionally on true airspeed from Mach, the sea's slant range from altitude and
depression angle, and a wings-level, no-sideslip assumption for the image orientation.

### Middle East Object: parallax under the MQ-9 cruise assumptions

`middle_east_parallax.py` applies the finite-background correction with Bellingcat's assumptions
(25,000 ft, 150–170 kt, flying toward the scene). Nothing in the HUD survives redaction, so the
ground scale comes from Bellingcat's geolocated aim points (0.165 m/px) and the object's height above
ground is swept. In ground terms the correction is `v_obj = (1 − h/A)·s_rel − v_ac⊥·(h/A)`.

Measured input: the object's LOS speed relative to the ground texture, **45 m/s median** (p5–p95
12–89 m/s over the 28 frames with trusted camera motion). Direction check: the ground-relative motion
is ~21° from straight "down" the image, i.e. toward the near side, which is what a stationary object
below a forward-flying drone produces.

| Object height AGL | 0 m | 1 km | 2 km | 3 km | 4 km | stationary at |
|---|---|---|---|---|---|---|
| Object speed, 45° depression, 160 kt | 45 m/s | 32 | 18 | 5 | −9 | **~3.3 km** |
| 30° / 75° depression | 45 | 34 / 29 | 23 / 13 | 11 / −4 | 0 / −21 | 4.0 / 2.8 km |

A stationary object (a balloon) reproduces the measured motion at roughly 3 km above ground under
every combination of speed and depression tried; a lower object must be moving, up to 45 m/s if it
were on the ground. The video cannot pick the height. The depression angle is itself unmeasured here
and fixes the implied FOV (0.35–0.67°); the plane-resection method would measure both from four or
more ground coordinates in one frame.
