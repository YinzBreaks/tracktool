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
