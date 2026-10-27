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

1. Install Python 3.10+ from python.org (tick "Add to PATH").
2. Double-click **`run_uapka.bat`**. The first run creates `.venv` and installs the dependencies.
   This is the only step that needs internet. After it, everything runs offline.

Manual install, any OS:

```bat
py -3 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m uapka            REM GUI
.venv\Scripts\python -m pytest           REM tests (about 40 s)
```

Optional: if `ffprobe` (FFmpeg) is on `PATH`, the metadata panel also shows the codec profile, pixel
format, field order (interlacing) and sample aspect ratio. Decoding never needs it.

Single-folder executable: `pip install pyinstaller` then
`pyinstaller -w -n UAPKA --collect-submodules uapka uapka_launcher.py`.

> `opencv-python-headless` is intentional: the regular `opencv-python` wheel bundles its own Qt
> plugins, and those conflict with PySide6.

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

1. **Video.** Open a file (or drag and drop it). The tab shows resolution, fps, duration, codec,
   SHA-256, decode warnings (variable frame rate, interlacing, frame-count mismatch) and duplicate
   frames.
2. **Masks.** Drag rectangles over HUD text, reticles, watermarks and redaction blocks, or click
   *Suggest static overlays*. That finds pixels that never change while the background moves.
   Masked pixels are excluded from camera registration and blob detection, and are neutralised
   inside tracking templates.
3. **Track.** Drag a box around the object once, or use *Auto-detect blobs* (bright/dark, with sky,
   sea or terrain presets) and double-click a candidate. If the track drifts, go to that frame,
   press *Correct position* and click the object. That frame becomes a keyframe, and only frames
   from there to the next keyframe are re-tracked. *Toggle exclude frame* marks occluded or torn
   frames. The timeline under the video shows track status (green OK, cyan manual, yellow
   re-acquired, red lost) and camera confidence (dark = low).
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

---

## Method

### 1. Ingest
The video is decoded once, sequentially, into an exact grayscale frame cache (RAM, or a disk
memmap for long clips). OpenCV seeking is not frame-accurate for many codecs, and a correction at
"frame N" must land on frame N. The tool also records container timestamps and detects duplicate
frames (frame-rate conversion inserts them; left in, the object appears to stop and then jump,
which creates fake acceleration spikes). Duplicates are excluded from all fits.

### 2. Tracking (`tracker.py`, `detect.py`)
The tracker predicts from camera motion plus the object's own recent motion, matches the template
with normalised cross-correlation, finds the parabolic sub-pixel peak, then refines to an
intensity-weighted centroid above the local background. A weak match (NCC below the threshold)
triggers re-acquisition by blob detection near the prediction. If that fails, the frame is marked
LOST and isn't used. Per-frame position uncertainty comes from match quality. Manual keyframes
carry their own σ (snapped clicks 0.5 px, otherwise 1 px).

### 3. Camera motion (`fpn.py`, `camera.py`), including the known failure
On low-texture IR backgrounds, **fixed-pattern noise** (per-pixel offsets, column stripes, hot
pixels) can carry more high-frequency energy than the scene. FPN is fixed to the sensor, so global
phase correlation finds its strongest peak at zero shift and reports "camera static". All the
apparent motion then gets attributed to the object. The tool counters this in five steps:

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
5. **Confidence score (0–1) per frame pair.** Built from inlier count and ratio, residual RMS,
   spatial coverage of inliers, agreement with phase correlation on the cleaned frames, and
   competing-cluster ambiguity. Fallbacks (phase correlation only, or no estimate) get low
   confidence. Each registration step also carries a 1σ in pixels.

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

The rates reported are |ω|, |α|, and the **lateral** (path-turning) component of α.

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
| lossless (FFV1), FPN 6, texture 3 | 98.3% | 0.3% | 0.77 | 0.1581 | 0.1578 | 0.45 | 0 | 0.0135 |
| lossy mp4v, FPN 6, texture 3 | 99.3% | 2.9% | 0.33 | 0.1581 | 0.1575 | 0.49 | 0 | 0.0118 |
| lossy mp4v, FPN 6, texture 6 | 97.2% | 0.7% | 0.62 | 0.1581 | 0.1581 | 0.58 | 0 | 0.0121 |
| lossless, zoom 1.0→1.5 | 98.9% | 0.3% | 0.46 | 0.1581 | 0.1574 | 0.31 | 0 | 0.0149 |
| lossless, turning object 90°/s | 97.8% | 0.4% | 0.80 | 0.1580 | 0.1527 | 0.65 | 0.2484 | 0.2424 |

\*Standard deviation of (measured − true)/σ over frames with camera confidence ≥ 0.3.
1.0 means honest error bars; below 1 means conservative. Angular units are deg/s and deg/s². The
"lateral α measured" values on straight-line clips (0.012–0.015 deg/s²) are well below the
acceleration noise floor (median 1σ ≈ 0.04 deg/s²), so they are indistinguishable from zero.

Naive phase correlation reports almost no camera motion on every clip, which is the FPN lock. The
tracked angular rate stays within 0.5% of truth (3.4% on the turning clip, where the 21-frame
window slightly smooths a curving path), even on the heavily compressed clip. There the low
median confidence (0.33) correctly signals a degraded estimate.

The test suite (`pytest`) repeats these checks on small clips. It also covers Monte-Carlo
calibration of the derivative σ, the random-walk camera error model, duplicate-frame detection,
overlay suggestion, scenario linearity, the size and altitude modes, and the rule that every CSV
column carries a provenance tag.

### Known limitations
- The motion model is a 2-D similarity: accurate for narrow FOVs, approximate toward the edges of
  wide-angle footage. Strong lens distortion and rolling shutter aren't modelled.
- Residual sensor-locked content biases chained estimates toward zero motion. That's about 1.5% on
  short 320×240 clips with heavy FPN, and below 0.5% on the validation clips above. The confidence
  score and the RAW/scale flags indicate when this is happening.
- Parallax isn't modelled. If the background is near (sea or terrain from a moving aircraft) and
  the platform translates, background registration doesn't give pure camera rotation.
- Single object, single camera. No triangulation.

---

## Code layout

```
uapka/
  video.py        metadata, SHA-256, exact frame cache, duplicate/VFR/interlace checks
  masks.py        mask regions, static-overlay suggestion
  fpn.py          fixed-pattern-noise estimate, scene-based refinement, band-pass cleaning
  camera.py       feature registration, confidence, GOP-lag detection + keyframe anchoring
  detect.py       bright/dark blob detection (sky / sea / terrain presets)
  tracker.py      template + centroid tracker, keyframes, re-acquisition
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
```

## Security notes (for the classroom)
- No network code anywhere in the package (`grep` for it). Optional `ffprobe` is called as a local
  subprocess with an argument list (no shell).
- Video decoding goes through OpenCV's bundled FFmpeg, which parses untrusted input. Keep
  `opencv-python-headless` updated, and treat downloaded footage like any other untrusted file.
- The SHA-256 of the analysed file is recorded in the report and the project file, as a basic
  chain-of-custody step. Re-opening a project warns if the hash changed.
