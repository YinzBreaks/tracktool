"""Provenance labels and the error-source catalogue.

Every number this tool outputs carries one of these labels. The point is that
a reader can never mistake a scenario-dependent number (which scales with an
assumed range) for something that was measured from the pixels.
"""

from __future__ import annotations

from enum import Enum


class Prov(str, Enum):
    # Directly from pixels and frame timing. Depends only on the video itself.
    MEASURED = "MEASURED"
    # Measured, but converted to angles with a user-supplied field of view /
    # pixel aspect ratio. Wrong FOV => wrong value, linearly.
    MEASURED_FOV = "MEASURED+FOV"
    # Requires a scenario assumption (slant range, object size or altitude).
    # Scales linearly with that assumption; not a measurement.
    ASSUMED = "ASSUMED"
    # Angles of the object computed WITHOUT camera-motion compensation, under the explicit assumption that the
    # camera was not moving. Used when the user switches compensation off (e.g. camera motion unmeasurable).
    STATIC_CAM = "MEASURED+FOV+ASSUMED-STATIC-CAMERA"
    # Same assumption, pixel quantities (no FOV involved).
    STATIC_CAM_PX = "MEASURED+ASSUMED-STATIC-CAMERA"
    # A field of view that was not typed in but derived (fov.py) from reference information: two points with a
    # known angle, an assumed object size/range, geo-located landmarks, or a HUD-label table. Carries a 1-sigma.
    # Reported as "DERIVED-FOV(<method>)", see derived_fov_label().
    DERIVED_FOV = "DERIVED-FOV"


def derived_fov_label(method: str) -> str:
    """Provenance label of a derived field of view, e.g. 'DERIVED-FOV(geo)'."""
    return f"{Prov.DERIVED_FOV.value}({method})"


PROV_DESCRIPTIONS = {
    Prov.MEASURED: "Computed only from pixel positions and frame timing in the video.",
    Prov.MEASURED_FOV: (
        "Measured pixel motion converted to angles using the user-supplied horizontal "
        "field of view and pixel aspect ratio. Scales linearly with the FOV value."
    ),
    Prov.STATIC_CAM: (
        "Measured image motion converted to angles with the FOV, attributing ALL of it to the object because the "
        "camera is ASSUMED static (camera-motion compensation switched off). Wrong if the camera moved."
    ),
    Prov.STATIC_CAM_PX: (
        "Measured image motion in pixels, attributed entirely to the object because the camera is ASSUMED static "
        "(camera-motion compensation switched off). Wrong if the camera moved."
    ),
    Prov.ASSUMED: (
        "Depends on a scenario assumption (slant range, object size, or altitude). "
        "Scales linearly with the assumed range. Not a measurement."
    ),
    Prov.DERIVED_FOV: (
        "A horizontal field of view derived from reference information rather than typed in: DERIVED-FOV(separation) "
        "two points with a known angle; DERIVED-FOV(size) two points spanning an ASSUMED size at an ASSUMED range; "
        "DERIVED-FOV(geo) geo-located landmarks and camera position; DERIVED-FOV(hud) a user-supplied HUD-label "
        "table. Each carries a 1-sigma; every MEASURED+FOV and ASSUMED value scales linearly with it."
    ),
}


def col(name: str, prov: Prov) -> str:
    """Column name with its provenance tag embedded, e.g. 'omega_deg_s [MEASURED+FOV]'."""
    return f"{name} [{prov.value}]"


# Error sources that always apply. Data-driven warnings (low camera-motion
# confidence, duplicate frames, lost track...) are appended at analysis time.
ERROR_SOURCES: list[tuple[str, str, str]] = [
    # (affects, source, note)
    ("position", "Tracking centroid error",
     "IR glare, blooming and the optical point-spread function make the blob larger than the object "
     "and can shift its centroid. Glare artefacts can also rotate with the gimbal, not with the object."),
    ("position", "Compression artefacts",
     "Block/ringing artefacts of re-encoded footage move small blobs by fractions of a pixel per frame."),
    ("timing", "Frame timing",
     "Nominal fps is used unless container timestamps are selected. Re-encoded footage often has "
     "duplicated, dropped or interlaced frames; duplicates are detected and excluded."),
    ("angle", "Field-of-view calibration",
     "All angular values scale linearly with the HFOV entered. HUD zoom labels (e.g. 'NAR', '2.0') "
     "are often digital zoom levels, not optical FOVs. Pixel aspect ratio must match the source "
     "(e.g. 720x480 DVD-era captures are not square-pixel)."),
    ("angle", "Lens model",
     "A pinhole camera without distortion is assumed. Camera motion is modelled as a 2-D similarity "
     "(pan/tilt translation, roll, zoom); this is accurate for narrow FOVs (< ~10 deg) and degrades "
     "towards the frame edges for wide-angle footage."),
    ("camera", "Camera-motion estimate",
     "Background registration fails on low-texture backgrounds (clear sky, IR sea), when fixed-pattern "
     "sensor noise or burned-in overlays dominate, or with rolling-shutter skew. See the per-frame "
     "confidence score."),
    ("camera", "Parallax",
     "Registration assumes the background is at infinity or the camera only rotates. When the platform "
     "translates and the background is near (sea, terrain), background motion includes parallax and the "
     "compensated object motion is biased."),
    ("camera", "Accumulated drift",
     "Frame-to-frame camera estimates are chained; errors accumulate as a random walk. Rates are affected "
     "less than absolute positions."),
    ("derivative", "Noise amplification",
     "Velocity and especially acceleration are derivatives of noisy positions. The smoothing window trades "
     "noise for time resolution; short, sharp manoeuvres are attenuated by wide windows, and peak values "
     "of noisy derivatives are biased upwards."),
    ("scenario", "Range assumption",
     "Speed and acceleration scale linearly with the assumed slant range. The range sweep makes this "
     "explicit; no single value is more 'correct' than the assumption that produced it."),
    ("scenario", "Unobservable radial motion",
     "A single camera measures only motion across the line of sight. Motion towards/away from the camera "
     "is invisible, so speeds are lower bounds on the true speed relative to the camera."),
    ("scenario", "Observer platform motion",
     "Velocities are relative to the camera platform. An aircraft-mounted camera's own velocity "
     "(often 100-250 m/s) appears as apparent object motion via parallax unless an observer velocity "
     "is supplied. Constant platform velocity does not create apparent acceleration, but platform turns do."),
    ("scenario", "Constant-range approximation",
     "Within a scenario the range is held constant, so range-rate (Coriolis) terms in the acceleration "
     "are ignored."),
]
