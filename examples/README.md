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

### Middle East Object: viewing geometry from a satellite map, and parallax

Files: `middle_east_project.json` (segment 3.2–24.7 s, side panels and reticle masked, the object's
positions taken from the sensor's own track gate, one control on a roof), `middle_east_geometry.py` (the
geometry below), `middle_east_parallax.py` (the earlier sweep that assumed an isotropic 0.165 m/px and swept
the depression angle). Video `DOD_109584445.mp4`, SHA-256
`030f45a791b5e8d3fe2c28c96fd65dbd5fc318fcd390493061d426f8296fecba`.

```bash
python -m uapka analyze DOD_109584445.mp4 --project examples/middle_east_project.json --out out/me
python examples/middle_east_geometry.py out/me/per_frame.csv
```

**Why the plane resection does not give the FOV here.** The drone is ~10 km from a scene a few hundred
metres across, so the ground maps to the image by an affine transform: four or six ground coordinates fit
a homography with no usable perspective, the focal length is undetermined, and `estimate_ground` says so
(`FOCAL LENGTH POORLY DETERMINED`). A first attempt, four corners of a 45 × 50 m rectangle, failed
for a second reason as well: the pins did not outline the structure in the video. What the affine map still
carries is reported in the new `weak_perspective` diagnostics: the orientation, the metres per pixel across
and along the look direction, and their ratio, which is the sine of the look-down angle.

**How the correspondence was established.** The HUD's red arrow is not a north arrow (treating it as one
sends every pairing to the wrong compound). The camera track was used to build a mosaic of frames 270–430,
on which the big walled courtyard, its tree, the trench along its east berm, the compounds north of it and
the plowed fields are all recognisable; five correspondences to a north-up Google Earth screenshot
(georeferenced from five pins read in Google Earth, consistent to 1 px) fixed the affine map, which was
then refined by gradient correlation. The video is roughly south-up (North 193° clockwise from image-up).
The structure under the reticle at 9 s is the shed just west of the road at Bellingcat's aim point A; the
flat-roofed building north of it sits under a redaction box.

| Quantity | Value | Rests on |
|---|---|---|
| Metres per pixel across / along the look direction | 0.156 / 0.247 | MEASURED (map ↔ mosaic affine, square pixels assumed) |
| Look-down angle | **39° ± 5°** | MEASURED (foreshortening ratio 0.63) |
| Line toward the aircraft in the image | 34° left of image-up, which is exactly the HUD's red arrow | MEASURED; the arrow is an aircraft pointer |
| Ground heading of that line | 326° or 146° (affine cannot tell near from far) | MEASURED |
| Slant range, HFOV of the 558-px inset / full 1280-px frame | 11.7 km, 0.43° / 0.98° at 25,000 ft; 9.3 km, 0.54° / 1.23° at 20,000 ft | + ASSUMED altitude |
| Aim displacement 7.57 → 23.59 s | 872 m at heading 7° from the camera chain, vs Bellingcat's 751 m at 45° | unresolved; the chain's confidence is low over the fields |

**Parallax, re-run with the measured geometry.** The object's motion is along the image vertical, where the
scale is 0.25 m/px rather than the isotropic 0.165 m/px used before, so its ground-relative speed is
**64 m/s median** (p5–p95 29–87 m/s, 28 trusted frames), heading ≈ 351°. With the look-down angle measured
(39°) and the altitude assumed (25,000 ft), `v_obj = (1 − h/A)·s_rel − v_ac⊥·(h/A)`:

| Object height AGL | 0 m | 1 km | 2 km | 3 km | 4 km | stationary at |
|---|---|---|---|---|---|---|
| 150–170 kt, LOS along the direction of flight | 64 m/s | 48–49 | 32–34 | 16–19 | 0–3 | **4.0–4.2 km** |
| 150–170 kt, LOS 90° off the heading | 64 | 44–45 | 23–26 | 3–7 | −18 to −12 | 3.1–3.4 km |

A stationary object (a balloon) still reproduces the measurement, now at 3–4 km above ground; an object on
the ground would be doing 64 m/s. The stationary case also needs the aircraft's lateral motion anti-parallel
to the object's ground-relative motion, i.e. an aircraft heading near 171°; nothing in the video gives the
heading, and Bellingcat's "flying toward the scene" is one guess among several. The altitude remains the
one assumption that scales the ranges and speeds; the look-down angle no longer has to be guessed.
