# UAP Kinematics Analyzer (`uapka`)

An offline desktop tool that takes any UAP/UFO video and extracts what the footage can actually
support: the object's on-screen path and its **angular** velocity and acceleration, with camera
pan/tilt/roll/zoom removed. It converts to metres per second and g **only** under assumptions you
enter, and it labels every number with where it came from.

Built for teaching and OSINT-style analysis. It does not identify objects. It shows which
conclusions the pixels support and which depend on an assumption.

- Python, OpenCV, NumPy, SciPy, matplotlib, PySide6 (Qt) GUI
- Runs locally on Windows (also Linux/macOS). There are no network calls, no server and no telemetry.
- Includes a headless CLI and a synthetic ground-truth generator for validation and classroom demos

---

## Why every value has a label

| Label | Meaning | Example |
|---|---|---|
| **MEASURED** | Comes only from pixel positions and frame timing | on-screen path (px), camera-compensated speed (px/s), camera-motion confidence |
| **MEASURED+FOV** | Measured motion converted to angles using the **horizontal field of view you enter**. Scales linearly with the FOV. | LOS angle (deg), angular velocity (deg/s), angular acceleration (deg/s²) |
| **ASSUMED** | Also depends on a scenario assumption (slant range, object size, or altitude). Scales linearly with the assumed range. **Not a measurement.** | speed (m/s, kt), lateral acceleration (g) |

These labels are embedded in every CSV column name (e.g. `omega_deg_s [MEASURED+FOV]`), in plot
titles, in the annotated video overlay and in the report. The per-frame CSV contains no ASSUMED
values. Scenario results go in separate files, and each row states the assumption that produced it.
The tool never shows a single g-force figure on its own. You always get a table across the
assumptions you entered, each peak with its 1σ and the noise floor, plus a "peak > 3σ?" column.

---

## Install and run (Windows)

1. Install 64-bit Python 3.10+ from python.org (tick "Add python.exe to PATH").
2. Double-click **`run_uapka.bat`**. The first run creates `.venv` next to it and installs the
   dependencies. This is the only step that needs internet. After it, everything runs offline.
   The same script runs the command line: `run_uapka.bat analyze clip.mp4 --out results ...`
   (any `uapka` command; see the CLI section). If a run fails, the window stays open with the error.

Manual install, any OS:

```bat
py -3 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m uapka            REM GUI
.venv\Scripts\python -m pytest           REM tests (about 50 s)
```

Optional: if `ffprobe` (FFmpeg) is on `PATH`, the metadata panel also shows the codec profile, pixel
format, field order (interlacing) and sample aspect ratio. Decoding never needs it.

**Self-contained folder (no Python on the target machine):** run `build_windows.bat` (it creates the
venv if needed, installs PyInstaller and runs the versioned `uapka.spec`; by hand:
`pyinstaller --noconfirm --clean uapka.spec`). The result is `dist\UAPKA\` with two executables
sharing one `_internal\` folder:

- `UAPKA.exe`: the GUI, windowed (no console). Double-click, or pass a video / `project.json` path.
- `uapka-cli.exe`: the command line (`info`, `synth`, `analyze`, `check-env`). A windowed executable
  has no console output on Windows, which is why the CLI is a second, console-mode exe.

Zip `dist\UAPKA\` to distribute it. Because `UAPKA.exe` has no console, anything it would print
(including a crash traceback) goes to `%LOCALAPPDATA%\uapka\uapka.log`.

Frame cache: clips are decoded once to grayscale. They stay in RAM up to a budget (2 GiB, or half of
the RAM currently available if that is less; override with `UAPKA_RAM_BUDGET_MB`), beyond that they go
to a memory-mapped file in `%TEMP%` (override with `UAPKA_CACHE_DIR`; free space is checked first, and
the file is deleted when the video is closed or at exit).

> `opencv-python-headless` is intentional: the regular `opencv-python` wheel bundles its own Qt
> plugins, and those conflict with PySide6. The spec refuses to build if `cv2` ships Qt plugins.

### Troubleshooting

First run the self-check and read its `FAIL`/`WARN` lines:

```bat
.venv\Scripts\python tools\check_env.py      REM source install
dist\UAPKA\uapka-cli.exe check-env            REM frozen build
```

It reports Python/Qt/OpenCV versions, whether the FFV1 / HFYU / mp4v / MJPG writers open, whether a Qt
platform plugin loads, ffprobe, console encoding, free disk in the cache folder and available RAM.

- **The GUI does not appear, or "no Qt platform plugin could be initialized"**: install the Microsoft
  Visual C++ Redistributable (x64); make sure neither `PyQt5`/`PyQt6` nor the non-headless
  `opencv-python` is installed in the venv (`check_env` reports both); `set QT_DEBUG_PLUGINS=1`
  shows which plugin failed. For `UAPKA.exe`, look in `%LOCALAPPDATA%\uapka\uapka.log`.
- **`UnicodeEncodeError` or garbled `±`/`σ` in a console**: the CLI switches its own output to UTF-8;
  for other Python scripts (`tools\validate.py` does it itself) set `PYTHONUTF8=1`, as
  `run_uapka.bat` does. Exported CSV/Markdown/HTML files are always UTF-8.
- **"The clip needs N GB frame cache on disk"**: the segment is too long for RAM and `%TEMP%` is
  short of space. Analyse a segment (start/end seconds), or point `UAPKA_CACHE_DIR` at a drive with
  room. Leftover `uapka_*.u8` files in that folder come from crashed runs; they are removed
  automatically after 24 h and are safe to delete.
- **"OpenCV could not open video"**: try a path without non-ASCII characters (OpenCV's FFmpeg backend
  on Windows fails on some); unusual containers can be re-muxed losslessly with
  `ffmpeg -i in.ext -c copy out.mp4`. `uapka info clip` shows what OpenCV sees.
- **Annotated video came out as `.avi`**, or "No video writer could be opened": the `mp4v` writer is
  missing from the OpenCV build, so MJPG/AVI was used (or export with the video off). Synthetic clips
  fall back from FFV1 to HuffYUV, then MJPG, and say so.
- **First start takes a minute**: matplotlib builds its font cache once; antivirus scanning of a fresh
  PyInstaller folder adds to it. SmartScreen warns about unsigned executables: "More info, Run anyway".
- **Closing the window during a long run**: the job is cancelled and the program waits for the worker
  thread (up to 30 s) before exiting, so the frame cache file can be deleted.

---

## GUI choice: PySide6 (Qt) rather than a local web UI

- **Offline by construction.** A local web UI needs an HTTP server on localhost. That opens a port,
  which brings DNS-rebinding and CSRF exposure from any web page the user visits, plus a browser
  dependency. A native Qt app has no listening sockets, and that's easier to defend in a security
  course.
- **Frame-accurate interaction.** Marking a 2–3 px blob needs pixel-exact zoom and pan, sub-pixel
  click mapping and instant scrubbing over thousands of frames. In Qt that's a NumPy array painted
  directly. A browser would need frames encoded and shipped to it.
- **One process, one language.** The GUI calls the same library as the CLI and tests. matplotlib
  figures embed natively (with zoom/pan toolbar pop-outs), and PyInstaller produces a single
  Windows folder or executable.
- **License.** PySide6 is LGPL, the official Qt for Python. PyQt is GPL or commercial.

---

## Workflow (GUI)

1. **Video.** Open a file (or drag and drop it). Optionally set a **segment** (start/end seconds)
   to analyse only part of a long release. The tab shows resolution, fps, duration, codec, SHA-256,
   and decode warnings: variable frame rate, interlacing, frame-count mismatch, frame-rate-converted
   content (duplicate cadence), blank/dropout frames, and coloured symbology.
2. **Masks.** Drag rectangles over HUD text, reticles, watermarks and redaction blocks, or click
   *Suggest static overlays*. That finds pixels that never change while the background moves.
   Masked pixels are excluded from camera registration and blob detection, and are neutralised
   inside tracking templates.
3. **Track.** Drag a box around the object once, or use *Auto-detect blobs* (bright/dark, with sky,
   sea or terrain presets) and double-click a candidate. Objects of a few pixels are tracked in
   **point mode** (matched filter). Larger ones use template matching. If the track drifts, go to
   that frame, press *Correct position* and click the object. That frame becomes a keyframe, and
   only frames from there to the next keyframe are re-tracked. *Toggle exclude frame* marks occluded
   or torn frames. *Start/End track here* bound an object that appears, splits or leaves. The
   timeline under the video shows track status (green OK, cyan manual, yellow re-acquired, red
   lost/weak, orange ambiguous) and camera confidence (dark = low).
   **Several objects** can be tracked (*Add object*), each as a **target** or a **control**. A control
   is a feature fixed to the background (a cloud, a coastline point, a surface feature). Its
   camera-compensated motion should be zero, so whatever remains is a *measured* error bar for the
   camera compensation. It's also the direct test of whether a "path" is just camera motion.
4. **Run full analysis** (Ctrl+R). This tracks, estimates camera motion with the object excluded,
   re-tracks with camera-motion prediction, then computes kinematics.
5. **Kinematics.** Enter the horizontal FOV of the first analysed frame (and the pixel aspect ratio
   if the source is non-square, e.g. 720×480 DVD captures). You get angular velocity and
   acceleration with ±1σ bands, the paths, and camera confidence.
6. **Scenarios.** Pick *range*, *size* or *altitude* mode and enter a list or a sweep
   (`1km, 5km, 10nmi`, `2km:40km:2km`, `log:1km:100km:9`; units m/km/ft/kft/nmi/mi). Optionally
   enter an assumed observer velocity across the line of sight to remove platform parallax. You
   get a table and a chart of speed and lateral g against the assumption.
7. **Export.** Writes the annotated video, CSVs, plots, a Markdown + self-contained HTML report,
   and `project.json`. The project file records masks, keyframes, every parameter and the video's
   SHA-256 so the analysis can be reproduced and checked.

## CLI (headless, scriptable)

```bash
uapka info clip.mp4                               # metadata + SHA-256
uapka synth demo.avi --fpn 6 --texture 3          # synthetic clip + demo.avi.truth.json
uapka analyze clip.mp4 --out results \
      --roi 412,233,14,14 --frame 0 \             # object centre x,y and box w,h at frame 0
      --hfov 0.7 \                                # FOV in degrees (MEASURED+FOV basis)
      --mask 0,0,220,40:HUD --mask 300,220,40,40:RETICLE --auto-masks \
      --ranges "log:1km:100km:9" \                # ASSUMED range sweep
      --compare-naive                             # also print what naive phase correlation gets
```

`--sizes "1m,5m,15m"` and `--altitudes "5kft:25kft:5kft" --cam-alt 25000ft --elevation -26` are the
other scenario modes. `--project results/project.json` re-runs a saved analysis.

Several objects, a segment of a long file, and controls:

```bash
uapka analyze release.mp4 --out results --start 4 --end 48 --auto-masks \
      --track "dot=865,641,25,25@4.5" --track "dot=1083,617,25,25@8.0" \   # target, 2 keyframes (x,y,w,h@seconds)
      --exclude "dot@8.5:10.3" --track-end "dot@16.6" \                   # occluded span; object ends
      --track "cloud/control=897,601,15,15@7.5" \                          # background feature = control
      --active dot
```

[`examples/README.md`](examples/README.md) has two worked case studies on real released footage
(AARO PR34 and PR35), with project files that reproduce them.

---

## Method

### 1. Ingest
The video (or a chosen segment of it) is decoded once, sequentially, into an exact grayscale frame
cache (RAM, or a disk memmap for long clips). OpenCV seeking is not frame-accurate for many codecs,
and a correction at "frame N" must land on frame N. Frame numbers and times in all outputs refer
to the original file. During decoding the tool also:

- records container timestamps;
- builds a **per-frame mask of coloured pixels**. Symbology burned into IR video is usually
  coloured and often moves (heading markers, track boxes), so static masks can't cover it;
- detects **blank/dropout frames**;
- detects **duplicate frames**. Frame-rate conversion produces a cadence (e.g. 10 fps content in a
  30 fps file: one new frame, then two near-copies). Re-encoded copies aren't bit-exact and are
  usually the *majority*, so duplicates are found by clustering the frame differences and, when a
  cadence exists, keeping one new frame per period.

Left in, duplicates make the object appear to stop and then jump, which creates fake acceleration.
Duplicates and dropouts are stepped over by tracking and registration and excluded from all fits.
Derivative windows count new-content frames.

### 2. Tracking (`tracker.py`, `detect.py`)
The tracker predicts from camera motion plus the object's own recent motion. Two prediction
hypotheses are kept: "continues" and "stopped/reversed". Constant velocity alone overshoots exactly
at the turns an analyst cares about. Matching then depends on the object's size, decided once per
object:

- **Extended objects:** NCC template matching, parabolic sub-pixel peak, intensity-weighted centroid.
- **Point targets** (a few pixels, typical of real footage): a template of a 2 px speck is mostly
  background and drifts, so a matched (difference-of-Gaussians) filter is used instead.
  - Candidates above an SNR threshold are scored by SNR, distance to the hypotheses, and similarity
    to the object's recent brightness.
  - If two candidates are equally plausible, the frame is marked AMBIGUOUS.
  - If only clutter-level detections exist, it's marked WEAK and the clutter isn't followed.
  - Re-acquisition widens the search but requires object-like brightness (much brighter detections
    are usually terrain or cloud texture).
  - Detections already claimed by another tracked object are off-limits.

Polarity (bright/dark) comes from the strongest response in the marked box. Unusable frames are
never used for kinematics, and **derivatives are never fitted across a track break** (a
re-acquisition or a lost span). Per-frame position uncertainty comes from match quality or SNR.
Manual keyframes carry their own σ (snapped clicks 0.5 px, otherwise 1 px).

### 3. Camera motion (`fpn.py`, `camera.py`), including the known failure
On low-texture IR backgrounds, **fixed-pattern noise** (per-pixel offsets, column stripes, hot
pixels) can carry more high-frequency energy than the scene. FPN is fixed to the sensor, so global
phase correlation finds its strongest peak at zero shift and reports "camera static". All the
apparent motion then gets attributed to the object. The tool counters this in seven steps:

1. **FPN estimation and removal.** Take the per-pixel temporal median of high-passed frames, flag
   defect pixels, remove per-frame column/row stripes, then apply a band-pass (σ≈2 px smoothing)
   that suppresses pixel-scale sensor residue while keeping the low-frequency structure of clouds
   and sea. A *static fraction* diagnostic warns when the "FPN" may actually be a static scene
   (locked-off camera).
2. **Two-pass, scene-based FPN refinement.** When FPN dominates, the first motion estimate is used
   to build a scene mosaic. Subtracting it from each frame isolates the sensor pattern, and
   registration runs again. This removes the residual pull toward zero motion when the camera
   hasn't panned far.
3. **Feature-based registration.** Grid-distributed Shi-Tomasi corners, pyramidal Lucas-Kanade
   tracking with a forward-backward check, and a RANSAC similarity fit (pan/tilt, roll, zoom).
   Masks, the tracked object and defect pixels are excluded. A second RANSAC on the outliers
   detects a competing motion cluster. If one cluster is near-identity (sensor-locked FPN or
   overlay residue) and the other moves, the moving one is preferred and the frame is flagged.
4. **Compression-lag detection and correction.** This was found during development. On a faint
   background with strong FPN, a lossy encoder's own motion search matches the static FPN. The
   **decoded** P-frames therefore genuinely contain a background that lags or freezes, then
   catches up at the next I-frame. No frame-to-frame method can recover motion the encoder
   discarded. The tool detects this (camera-step jumps that coincide with frame-difference spikes),
   registers I-frame to I-frame directly, and interpolates the camera path between them. On the
   validation clip this cut accumulated pan error from 41% to 3%.
   Lag is accepted only when the keyframes register to each other reliably **and** the chain is
   either grossly off or *systematically short along the direction of travel*. A chain that agrees
   with the keyframes is kept.
5. **Zoom and roll only when supported.** Per-step scale and rotation on weak backgrounds are noise,
   and chaining them produces fake zoom and roll. They are kept only where a windowed median shows
   a significant, consistent trend; elsewhere the model is translation-only.
6. **Long-baseline closure check.** Frame-to-frame registration can be *confidently wrong*: locked
   onto sensor-fixed or encoder-frozen content, it agrees with itself. Every ~10 new frames the
   tool registers directly across the interval, starting from the chained estimate.
   - **Agreement:** confidence is kept.
   - **Disagreement:** confidence is capped hard.
   - **Scene can't be registered across the interval** (e.g. sea texture that decorrelates within a
     second): the camera motion is **unverifiable** and capped below the trust threshold.
7. **Confidence score (0–1) per frame pair.** Built from inlier count and ratio, residual RMS,
   spatial coverage of inliers, agreement with phase correlation on the cleaned frames,
   competing-cluster ambiguity and the closure check. Fallbacks (phase correlation only, or no
   estimate) get low confidence. Each registration step also carries a 1σ in pixels.
   Derivatives whose fit window contains an untrusted step are kept in the CSV (flagged) but
   excluded from headline numbers and scenario tables. If nothing is trusted, the report says the
   camera motion could not be measured, and that on-screen paths and turns describe the image,
   not the object.

The flag `RAW_PHASECORR_LOCKED_TO_SENSOR` marks frames where naive phase correlation on the raw
frames would have reported zero motion while the cleaned feature registration measured real
motion. The report counts these flags, so students can see the failure that was avoided.

### 4. Angular kinematics (`kinematics.py`)
The LOS angle is the camera pointing (accumulated pan/tilt in angle) plus the object's in-frame
offset (exact `atan`, rotated and scaled by the measured roll and zoom). Derivatives come from
**local weighted polynomial fits on the real frame times**. This handles gaps, duplicates and
variable frame rate. Samples without data on both sides get no derivative, because one-sided edge
fits invent manoeuvres.

1σ uncertainties combine two parts:
- tracking noise, treated as white position noise and propagated through the fit;
- camera-registration error, modelled as a **random walk** because chained steps accumulate it.
  That is white noise in velocity, which a white-position model would understate by about √N.

A reduced-χ² inflation is applied where the local model fits poorly. The validation table below
shows the resulting error bars are honest to conservative.

The rates reported are |ω|, |α|, and the **lateral** (path-turning) component of α. On-screen
(not camera-compensated) speed and angular rate are reported separately and labelled as such.
**Heading changes ("turns")** are detected on both the on-screen and the camera-compensated path.
A turn that appears only on screen comes from camera motion.

With compensation switched off (`--static-camera`), all on-screen motion is attributed to the
object. Every affected value is then labelled `MEASURED+FOV+ASSUMED-STATIC-CAMERA`.

### Deriving the FOV (`fov.py`)
Released footage rarely states its field of view, so `uapka fov <method>` derives the HFOV from
things an analyst can know, with a 1σ and the inputs recorded. The geometry is an exact pinhole:
the angle between two clicked points is the angle between their rays, and the focal length is
solved so that it matches the known true angle (the small-angle shortcut θ·W/d is only reported as
a sanity check; it is several percent off for off-centre points at wide FOVs).

| Method | Known inputs | Label |
|---|---|---|
| `separation` | two points and the true angle between them (± σ) | `DERIVED-FOV(separation)` |
| `size` | two points spanning an **assumed** size at an **assumed** range (± σ each) | `DERIVED-FOV(size)` |
| `geo` | camera lat/lon/alt and ≥ 2 geo-located landmarks clicked in the frame (WGS-84, local ENU); ≥ 3 adds a consistency residual | `DERIVED-FOV(geo)` |
| `hud` | a user-supplied table of HUD zoom labels → HFOV | `DERIVED-FOV(hud)` |

Click uncertainty (±1 px per point by default) and every input σ are propagated by finite
differences and combined in quadrature. `--save fov.json` stores the estimate;
`analyze --fov-json fov.json` applies it and the report states the method, inputs and HFOV ± σ
(`--hfov` wins if both are given, and is reported as "user input, no uncertainty given"). All
angular rates and scenario values scale linearly with the HFOV, so its relative σ carries straight
through to them.

### 5. Scenarios (`scenarios.py`): ASSUMED only
For an assumed range R (held constant):

- transverse speed v = R·ω, plus the assumed observer velocity
- acceleration a = R·α
- lateral acceleration a_lat = R·α_lateral

Range can come from:

- **range**: entered directly
- **size**: R = L / apparent angular size. In IR the apparent size is inflated by blur, glare and
  blooming, so these ranges are biased short and the speeds biased low. The table says so.
- **altitude**: R = Δh / sin(elevation), flat earth

Each row lists the median, 95th-percentile and peak speed (±1σ), peak total and lateral g (±1σ),
the noise floor, and whether the peak clears 3σ. Speeds are lower bounds: radial motion is
invisible to a single camera.

## Error sources
Every report lists these: tracking-centroid error (glare/bloom/PSF, glare rotating with the
gimbal), compression artefacts, frame timing (duplicates, VFR, interlacing), FOV calibration
(digital vs optical zoom labels, pixel aspect ratio), lens model (pinhole, similarity motion
model), camera-motion estimate (low texture, FPN, overlays, rolling shutter), parallax of near
backgrounds when the platform translates, accumulated drift, derivative noise amplification and
peak bias, the range assumption, unobservable radial motion, observer-platform motion, and the
constant-range approximation. Data-driven warnings (low-confidence frames, lost frames,
duplicates, missing FOV, compression lag) are added per analysis.

---

## Validation

`python tools/validate.py` generates five 640×480, 6-second clips. Each has strong FPN
(σ = 6 grey levels plus column stripes and hot pixels) over a faint background (texture σ = 3–6),
a HUD, a reticle and a redaction block, and a known object trajectory. HFOV is 2°.

| Clip | Naive PC pan error | Our pan error | Median cam. conf. | ω true | ω measured (median) | ω z-spread* | Lateral α true | Lateral α measured |
|---|---|---|---|---|---|---|---|---|
| lossless (FFV1), FPN 6, texture 3 | 98.3% | 0.5% | 0.77 | 0.1581 | 0.1573 | 0.44 | 0 | 0.0079 |
| lossy mp4v, FPN 6, texture 3 | 99.3% | 3.0% | 0.43 | 0.1581 | 0.1578 | 0.69 | 0 | 0.0073 |
| lossy mp4v, FPN 6, texture 6 | 97.2% | 0.8% | 0.62 | 0.1581 | 0.1579 | 0.55 | 0 | 0.0101 |
| lossless, zoom 1.0→1.5 | 98.9% | 0.3% | 0.47 | 0.1581 | 0.1578 | 0.29 | 0 | 0.0121 |
| lossless, turning object 90°/s | 97.8% | 0.3% | 0.80 | 0.1580 | 0.1533 | 0.38 | 0.2484 | 0.2433 |

\*Standard deviation of (measured − true)/σ over frames with camera confidence ≥ 0.3.
1.0 means honest error bars; below 1 means conservative. Angular units are deg/s and deg/s². The
"lateral α measured" values on straight-line clips (0.007–0.013 deg/s²) are well below the
acceleration noise floor (median 1σ ≈ 0.04 deg/s²), so they are indistinguishable from zero.

Naive phase correlation reports almost no camera motion on every clip, which is the FPN lock. The
tracked angular rate stays within 0.5% of truth (3% on the turning clip, where the 21-frame
window slightly smooths a curving path), even on the heavily compressed clip. There the lower
median confidence (0.43) correctly signals a degraded estimate.

The test suite (`pytest`) repeats these checks on small clips. It also covers Monte-Carlo
calibration of the derivative σ, the random-walk camera error model, duplicate-frame detection,
overlay suggestion, scenario linearity, the size and altitude modes, and the rule that every CSV
column carries a provenance tag. A further set of tests reproduces real-footage conditions:

- frame-rate cadence with dropouts;
- segment offsets;
- tracking a 2 px target from a sloppy click;
- turn timing;
- multi-object start/end round trips;
- letterbox detection.

### Known limitations
- The motion model is a 2-D similarity: accurate for narrow FOVs, approximate toward the edges of
  wide-angle footage. Strong lens distortion and rolling shutter aren't modelled.
- Residual sensor-locked content biases chained estimates toward zero motion. That's about 1.5% on
  short 320×240 clips with heavy FPN, and below 0.5% on the validation clips above. The confidence
  score and the RAW/scale flags indicate when this is happening.
- Parallax isn't modelled. If the background is near (sea or terrain from a moving aircraft) and
  the platform translates, background registration doesn't give pure camera rotation. Likewise,
  motion relative to the background isn't the object's own velocity when the object is much closer
  than the background.
- Some backgrounds can't be registered at all: heavily compressed sea, for example (see PR34 in
  `examples/`). The tool then says so rather than guessing, and only on-screen results remain.
- A faint point target crossing textured clutter (clouds, terrain) needs keyframes. Clutter that
  moves with the background agrees with the camera-motion prediction, so the tracker (or a person)
  can follow it instead of the object. Check the annotated video. Control tracks expose such
  mix-ups.
- Single camera, so no triangulation.

---

## Code layout

```
uapka/
  video.py        metadata, SHA-256, exact (segment) frame cache, duplicate cadence, dropouts, colour overlays
  masks.py        mask regions, static-overlay and letterbox suggestion
  fpn.py          fixed-pattern-noise estimate, scene-based refinement, band-pass cleaning
  camera.py       feature registration, confidence, closure check, zoom/roll trend test, GOP-lag anchoring
  detect.py       bright/dark blob detection (sky / sea / terrain presets)
  tracker.py      point (matched filter) / template tracker, keyframes, start/end, multi-object
  kinematics.py   LOS angles, local polynomial derivatives, uncertainty propagation
  scenarios.py    range / size / altitude assumptions -> speed and g tables
  provenance.py   labels and the error-source catalogue
  plots.py        figures (provenance in every title)
  export.py       CSVs, annotated video, Markdown/HTML report
  project.py      project file + pipeline orchestration
  synth.py        synthetic ground-truth clips (FPN, compression, zoom, turns, HUD)
  cli.py          command line
  gui/            PySide6 application
tests/            pytest suite against synthetic ground truth
tools/validate.py validation table above
examples/         real-footage case studies (AARO PR34, PR35) with reproducible project files
```

## Security notes (for the classroom)
- No network code anywhere in the package (`grep` for it). Optional `ffprobe` is called as a local
  subprocess with an argument list (no shell).
- Video decoding goes through OpenCV's bundled FFmpeg, which parses untrusted input. Keep
  `opencv-python-headless` updated, and treat downloaded footage like any other untrusted file.
- The SHA-256 of the analysed file is recorded in the report and the project file, as a basic
  chain-of-custody step. Re-opening a project warns if the hash changed.
