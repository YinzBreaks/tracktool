"""Project state and the analysis pipeline.

A project file (JSON) records everything needed to reproduce an analysis: the
video's SHA-256, masks, user keyframes and exclusions, and every parameter.
Results are recomputed from it; the hash guards against analysing a different
file than the one the project was made for.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from . import __version__
from .camera import CameraParams, CameraTrack, estimate_camera_motion
from .kinematics import KinParams, Kinematics, compute_kinematics
from .masks import Mask
from .scenarios import ScenarioResult, ScenarioSpec, compute_scenarios
from .tracker import Track, TrackParams, run_tracking
from .video import FrameCache, VideoMeta, load_frames, probe


@dataclass
class Project:
    video_path: str = ""
    video_sha256: str = ""
    masks: list = field(default_factory=list)
    camera_params: CameraParams = field(default_factory=CameraParams)
    track_params: TrackParams = field(default_factory=TrackParams)
    kin_params: KinParams = field(default_factory=KinParams)
    scenario: ScenarioSpec = field(default_factory=ScenarioSpec)
    notes: str = ""
    # runtime (not serialised directly)
    meta: Optional[VideoMeta] = None
    cache: Optional[FrameCache] = None
    track: Optional[Track] = None
    camera: Optional[CameraTrack] = None
    kin: Optional[Kinematics] = None
    scen: Optional[ScenarioResult] = None
    log: list = field(default_factory=list)

    # ---------------- persistence
    def to_dict(self) -> dict:
        return {
            "tool": "uapka",
            "version": __version__,
            "video_path": self.video_path,
            "video_sha256": self.video_sha256,
            "masks": [m.to_dict() for m in self.masks],
            "camera_params": asdict(self.camera_params),
            "track_params": asdict(self.track_params),
            "kin_params": asdict(self.kin_params),
            "scenario": self.scenario.to_dict(),
            "track": self.track.to_dict() if self.track else None,
            "notes": self.notes,
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)

    @staticmethod
    def load(path: str) -> "Project":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        p = Project(
            video_path=d.get("video_path", ""),
            video_sha256=d.get("video_sha256", ""),
            masks=[Mask.from_dict(m) for m in d.get("masks", [])],
            camera_params=CameraParams(**d.get("camera_params", {})),
            track_params=TrackParams(**d.get("track_params", {})),
            kin_params=KinParams(**d.get("kin_params", {})),
            scenario=ScenarioSpec(**d.get("scenario", {})),
            notes=d.get("notes", ""),
        )
        if d.get("track"):
            p.track = Track.from_dict(d["track"])
        return p

    # ---------------- pipeline steps
    def open_video(self, path: str, progress=None, cancel=None) -> list[str]:
        meta = probe(path)
        warnings = []
        if self.video_sha256 and self.video_sha256 != meta.sha256:
            warnings.append("SHA-256 of the video differs from the one recorded in the project file!")
        cache = load_frames(meta, progress=progress, cancel=cancel)
        cache.detect_duplicates()
        if self.cache is not None:
            self.cache.close()
        self.meta, self.cache, self.video_path, self.video_sha256 = meta, cache, meta.path, meta.sha256
        if self.track is None or self.track.n != len(cache):
            if self.track is not None:
                warnings.append("Saved track does not match the frame count; track reset.")
            self.track = Track(len(cache))
        self.camera = self.kin = self.scen = None
        warnings += cache.warnings
        if cache.duplicate.any():
            warnings.append(f"{int(cache.duplicate.sum())} duplicate frames detected (excluded from kinematics).")
        self.log += warnings
        return warnings

    def frame_times(self) -> np.ndarray:
        if self.kin_params.timing == "container":
            ct = self.cache.container_times()
            if ct is not None:
                return ct
        return self.cache.nominal_times()

    def run_camera(self, progress=None, cancel=None) -> CameraTrack:
        boxes = self.track.box if (self.track is not None and self.track.usable().any()) else self._keyframe_boxes
        self.camera = estimate_camera_motion(self.cache.frames, self.masks, self.camera_params,
                                             object_boxes=boxes, progress=progress, cancel=cancel)
        self.log += [self.camera.fpn_info, self.camera.gop_info]
        return self.camera

    def _keyframe_boxes(self, k: int):
        if self.track is None or not self.track.keyframes:
            return None
        kf = self.track.keyframes
        nearest = min(kf, key=lambda j: abs(j - k))
        cx, cy, w, h = kf[nearest]
        grow = 1 + abs(nearest - k) * 0.05
        return (cx, cy, w * grow + 40, h * grow + 40) if abs(nearest - k) < 30 else None

    def run_tracking(self, progress=None, cancel=None) -> Track:
        run_tracking(self.cache.frames, self.track, self.masks, self.camera, self.track_params, progress, cancel)
        return self.track

    def run_kinematics(self) -> Kinematics:
        self.kin = compute_kinematics(self.track, self.camera, self.frame_times(), self.cache.duplicate,
                                      self.cache[0].shape, self.kin_params)
        return self.kin

    def run_scenarios(self) -> ScenarioResult:
        if self.kin is None:
            self.run_kinematics()
        self.scen = compute_scenarios(self.kin, self.scenario)
        return self.scen

    def run_all(self, progress=None, cancel=None, with_camera: bool = True) -> None:
        """Track -> camera (excluding the track) -> re-track with camera prediction -> kinematics -> scenarios."""
        def sub(a, b):
            return (lambda f, m: progress(a + (b - a) * f, m)) if progress else None
        self.camera = None
        self.run_tracking(sub(0.0, 0.1), cancel)
        if with_camera:
            self.run_camera(sub(0.1, 0.85), cancel)
            self.run_tracking(sub(0.85, 0.95), cancel)
        self.run_kinematics()
        if self.kin.has_angles():
            try:
                self.run_scenarios()
            except ValueError as e:
                self.log.append(f"Scenarios not computed: {e}")
        if progress:
            progress(1.0, "Analysis complete")
