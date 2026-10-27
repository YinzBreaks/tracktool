"""Project state and the analysis pipeline.

A project file (JSON) records everything needed to reproduce an analysis: the
video's SHA-256, the analysed segment, masks, every object's keyframes and
exclusions, and every parameter.

A project can hold several tracked objects. Each is a "target" (under
investigation) or a "control" (a feature fixed to the background, used to
measure how well camera motion was removed). One target is "active": it drives
the scenario table and the headline numbers.
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
    segment_start_s: float = 0.0       # analyse only this part of the file (0 / 0 = whole file)
    segment_end_s: float = 0.0
    masks: list = field(default_factory=list)
    camera_params: CameraParams = field(default_factory=CameraParams)
    track_params: TrackParams = field(default_factory=TrackParams)
    kin_params: KinParams = field(default_factory=KinParams)
    scenario: ScenarioSpec = field(default_factory=ScenarioSpec)
    notes: str = ""
    tracks: list = field(default_factory=list)
    active: int = 0
    # runtime (not serialised directly)
    meta: Optional[VideoMeta] = None
    cache: Optional[FrameCache] = None
    camera: Optional[CameraTrack] = None
    kins: list = field(default_factory=list)
    scen: Optional[ScenarioResult] = None
    log: list = field(default_factory=list)
    scen_error: str = ""

    # ---------------- objects
    @property
    def track(self) -> Optional[Track]:
        return self.tracks[self.active] if self.tracks else None

    @track.setter
    def track(self, t: Track) -> None:
        if not self.tracks:
            self.tracks = [t]
            self.active = 0
        else:
            self.tracks[self.active] = t

    @property
    def kin(self) -> Optional[Kinematics]:
        return self.kins[self.active] if self.kins and self.active < len(self.kins) else None

    @kin.setter
    def kin(self, k: Optional[Kinematics]) -> None:
        if k is None:
            self.kins = []
        else:
            while len(self.kins) <= self.active:
                self.kins.append(None)
            self.kins[self.active] = k

    def add_track(self, name: str, role: str = "target") -> Track:
        n = len(self.cache) if self.cache is not None else (self.tracks[0].n if self.tracks else 0)
        t = Track(n, name=name, role=role)
        self.tracks.append(t)
        return t

    def get_track(self, name: str) -> Optional[Track]:
        return next((t for t in self.tracks if t.name == name), None)

    # ---------------- persistence
    def to_dict(self) -> dict:
        return {
            "tool": "uapka",
            "version": __version__,
            "video_path": self.video_path,
            "video_sha256": self.video_sha256,
            "segment_start_s": self.segment_start_s,
            "segment_end_s": self.segment_end_s,
            "masks": [m.to_dict() for m in self.masks],
            "camera_params": asdict(self.camera_params),
            "track_params": asdict(self.track_params),
            "kin_params": asdict(self.kin_params),
            "scenario": self.scenario.to_dict(),
            "tracks": [t.to_dict() for t in self.tracks],
            "active": self.active,
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
            segment_start_s=float(d.get("segment_start_s", 0.0)),
            segment_end_s=float(d.get("segment_end_s", 0.0)),
            masks=[Mask.from_dict(m) for m in d.get("masks", [])],
            camera_params=CameraParams(**d.get("camera_params", {})),
            track_params=TrackParams(**d.get("track_params", {})),
            kin_params=KinParams(**d.get("kin_params", {})),
            scenario=ScenarioSpec(**d.get("scenario", {})),
            notes=d.get("notes", ""),
        )
        if d.get("tracks"):
            p.tracks = [Track.from_dict(t) for t in d["tracks"]]
        elif d.get("track"):          # single-object projects from earlier versions
            p.tracks = [Track.from_dict(d["track"])]
        p.active = min(int(d.get("active", 0)), max(len(p.tracks) - 1, 0))
        return p

    # ---------------- pipeline steps
    def open_video(self, path: str, progress=None, cancel=None) -> list[str]:
        meta = probe(path)
        warnings = []
        if self.video_sha256 and self.video_sha256 != meta.sha256:
            warnings.append("SHA-256 of the video differs from the one recorded in the project file!")
        cache = load_frames(meta, progress=progress, cancel=cancel, start_s=self.segment_start_s,
                            end_s=self.segment_end_s or None)
        cache.detect_blank()
        cache.detect_duplicates()
        if self.cache is not None:
            self.cache.close()
        self.meta, self.cache, self.video_path, self.video_sha256 = meta, cache, meta.path, meta.sha256
        if any(t.n != len(cache) for t in self.tracks):
            warnings.append("Saved tracks do not match the analysed frame count; tracks reset.")
            self.tracks = []
        if not self.tracks:
            self.tracks = [Track(len(cache))]
            self.active = 0
        self.camera = self.scen = None
        self.kins = []
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

    def _target_boxes(self, k: int) -> list:
        """Boxes of all TARGET objects at frame k (excluded from background registration). Controls are part of
        the background and stay in."""
        out = []
        for t in self.tracks:
            if t.role != "target":
                continue
            if t.usable().any():
                b = t.box(k)
            elif t.keyframes:
                nearest = min(t.keyframes, key=lambda j: abs(j - k))
                cx, cy, w, h = t.keyframes[nearest]
                grow = 1 + abs(nearest - k) * 0.05
                b = (cx, cy, w * grow + 40, h * grow + 40) if abs(nearest - k) < 30 else None
            else:
                b = None
            if b is not None:
                out.append(b)
        return out

    def run_camera(self, progress=None, cancel=None) -> CameraTrack:
        self.camera = estimate_camera_motion(self.cache.frames, self.masks, self.camera_params,
                                             object_boxes=self._target_boxes, progress=progress, cancel=cancel,
                                             skip=self.cache.skip(), overlay=self._overlay())
        self.log += [self.camera.fpn_info, self.camera.model_info, self.camera.gop_info, self.camera.closure_info]
        return self.camera

    def run_tracking(self, progress=None, cancel=None, only_active: bool = False) -> Track:
        todo = [self.track] if only_active else [t for t in self.tracks if t.keyframes]
        for i, t in enumerate(todo):
            sub = (lambda f, m, i=i: progress((i + f) / len(todo), f"[{t.name}] {m}")) if progress else None
            # objects listed earlier keep their detections; a later object may not claim them
            others = [o for o in self.tracks[:self.tracks.index(t)] if o.keyframes]
            run_tracking(self.cache.frames, t, self.masks, self.camera, self.track_params, sub, cancel,
                         dup=self.cache.duplicate, blank=self.cache.blank, overlay=self._overlay(), others=others)
        return self.track

    def _overlay(self):
        return self.cache.overlay if self.cache.has_overlays() else None

    def file_frame(self, k: int) -> int:
        """Cache index -> frame number in the video file."""
        return k + (self.cache.offset if self.cache is not None else 0)

    def cache_index(self, frame: int) -> int:
        return frame - (self.cache.offset if self.cache is not None else 0)

    def run_kinematics(self) -> Kinematics:
        self.kins = [
            compute_kinematics(t, self.camera, self.frame_times(), self.cache.duplicate, self.cache[0].shape,
                               self.kin_params, frame_offset=self.cache.offset, blank=self.cache.blank)
            if t.usable().any() else None
            for t in self.tracks]
        return self.kin

    def run_scenarios(self) -> ScenarioResult:
        if self.kin is None:
            self.run_kinematics()
        self.scen_error = ""
        self.scen = compute_scenarios(self.kin, self.scenario)
        return self.scen

    def run_all(self, progress=None, cancel=None, with_camera: bool = True) -> None:
        """Track -> camera (excluding targets) -> re-track with camera prediction -> kinematics -> scenarios."""
        def sub(a, b):
            return (lambda f, m: progress(a + (b - a) * f, m)) if progress else None
        self.camera = None
        if not with_camera:
            self.kin_params.use_camera = False
        self.run_tracking(sub(0.0, 0.1), cancel)
        if with_camera:
            self.run_camera(sub(0.1, 0.85), cancel)
            self.run_tracking(sub(0.85, 0.95), cancel)
        self.run_kinematics()
        self.scen = None
        if self.kin is not None and self.kin.has_angles():
            try:
                self.run_scenarios()
            except ValueError as e:
                self.scen = None
                self.scen_error = str(e)
                self.log.append(f"Scenarios not computed: {e}")
        if progress:
            progress(1.0, "Analysis complete")
