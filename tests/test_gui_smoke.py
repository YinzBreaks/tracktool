"""Headless smoke test of the PySide6 GUI: every user action is driven programmatically on a small synthetic
clip. Any exception reaching the Qt event loop, any blocking dialog, and any stale display after an action is a
failure. Runs with QT_QPA_PLATFORM=offscreen and skips if a QApplication cannot be created."""

import json
import os
import sys
import time
import traceback

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

try:
    _app = QApplication.instance() or QApplication([])
except Exception as e:  # pragma: no cover - no usable Qt platform
    pytest.skip(f"cannot create a QApplication: {e}", allow_module_level=True)

from uapka.gui import app as gui_app  # noqa: E402
from uapka.gui.frameview import STATUS_COLORS  # noqa: E402
from uapka.masks import Mask  # noqa: E402
from uapka import tracker  # noqa: E402


class Gui:
    """One MainWindow plus helpers: wait for jobs, record notifications and event-loop exceptions."""

    def __init__(self):
        self.msgs = []
        self.errors = []
        self._old_hook = sys.excepthook
        sys.excepthook = self._hook
        self.w = gui_app.MainWindow()
        self.w.resize(1400, 900)
        self.w.show()
        self.pump()

    def _hook(self, t, v, tb):
        self.errors.append("".join(traceback.format_exception(t, v, tb)))

    def record(self, kind, title, text, detail=""):
        self.msgs.append((kind, title, text))
        self.w.statusBar().showMessage(text.splitlines()[0], 10000)

    def pump(self):
        for _ in range(3):
            _app.processEvents()

    def wait(self, timeout=90.0):
        t0 = time.time()
        while self.w.job is not None:
            _app.processEvents()
            time.sleep(0.01)
            assert time.time() - t0 < timeout, "GUI job did not finish in time"
        self.pump()
        assert not self.errors, "exception reached the Qt event loop:\n" + "\n".join(self.errors)

    def pop_msgs(self):
        m, self.msgs = self.msgs, []
        return m

    def status(self):
        return self.w.statusBar().currentMessage()


@pytest.fixture(scope="module")
def gui(lossless_clip):
    mp = pytest.MonkeyPatch()
    g = None
    mp.setattr(gui_app.MainWindow, "_msg", lambda self, *a, **k: g.record(*a, **k))
    g = Gui()
    yield g
    g.w.close()
    g.pump()
    mp.undo()
    sys.excepthook = g._old_hook
    assert not g.errors, "\n".join(g.errors)


@pytest.fixture(scope="module")
def clip(lossless_clip):
    path, truth = lossless_clip
    return path, truth, truth["object_screen_px"]


def ensure_video(gui, clip):
    """The tests build on each other in file order; run alone, they open the clip themselves."""
    if gui.w.p.cache is None or not os.path.samefile(gui.w.p.video_path, clip[0]):
        gui.w.open_video(clip[0])
        gui.wait()
        gui.pop_msgs()


# ----------------------------------------------------------------------------- video
def test_actions_without_video_give_feedback(gui):
    w = gui.w
    assert not w.slider.isEnabled() and not w.btn_run.isEnabled()
    for fn in (w.run_all, w.run_tracking, w.run_kinematics, w.run_scenarios, w.run_camera, w._detect,
               w._suggest_masks, w._add_object, w.reload_segment, w._start_here, w._toggle_exclude):
        fn()
        assert gui.status(), fn.__name__
    assert w.export_to("/nonexistent/dir") is False
    assert w.save_project("/nonexistent/dir/p.json") is False
    gui.wait()
    gui.pop_msgs()


def test_open_video(gui, clip):
    path, truth, gt = clip
    w = gui.w
    assert w.open_video(path)
    gui.wait()
    n = len(w.p.cache)
    assert n == len(truth["t"])
    assert w.slider.maximum() == n - 1 == w.spin_frame.maximum()
    assert w.sp_start.maximum() == n - 1 and w.sp_end.maximum() == n - 1
    assert w.slider.isEnabled() and w.btn_run.isEnabled() and w.btn_mark.isEnabled()
    assert "frames" in gui.status() and "Resolution" in w.txt_meta.toPlainText() or w.txt_meta.toPlainText()
    assert w.k == 0 and w.view._img is not None
    assert w.cmb_obj.count() == 1 and w.p.track is not None
    assert w.txt_kin.toPlainText() == "" and w.fig_kin.fig is None and w.tbl_scen.rowCount() == 0
    assert not gui.msgs


def test_open_path_rejects_missing_file(gui):
    w = gui.w
    assert w.open_video("/nonexistent/clip.mp4") is False
    kind, _, text = gui.pop_msgs()[-1]
    assert kind == "warning" and "not found" in text


# ----------------------------------------------------------------------------- masks
def test_masks(gui, clip):
    ensure_video(gui, clip)
    w = gui.w
    w.view.maskDrawn.emit(0, 0, 200, 34)
    w.cmb_mask_kind.setCurrentText("RETICLE")
    w.view.maskDrawn.emit(128, 88, 64, 64)
    assert len(w.p.masks) == 2 and w.lst_masks.count() == 2 and len(w.view.masks) == 2
    assert w.p.masks[1].kind == "RETICLE"
    w._delete_mask()                                    # nothing selected
    assert len(w.p.masks) == 2 and "Select" in gui.status()
    w.lst_masks.setCurrentRow(1)
    w._delete_mask()
    assert len(w.p.masks) == 1 and w.lst_masks.count() == 1
    w._suggest_masks()
    assert len(w.p.masks) > 1 and "suggested" in gui.status()
    w._clear_masks()
    assert not w.p.masks and w.lst_masks.count() == 0
    w._clear_masks()
    assert "No masks" in gui.status()
    w.p.masks = [Mask.from_dict(d) for d in clip[1]["hud_masks"]]
    w._refresh_masks()
    assert w.lst_masks.count() == len(w.p.masks)


# ----------------------------------------------------------------------------- tracking
def test_detect_and_pick_candidate(gui, clip):
    ensure_video(gui, clip)
    _, _, gt = clip
    w = gui.w
    w.goto(0)
    w._detect()
    assert w.lst_cand.count() >= 1 and len(w.view.candidates) == w.lst_cand.count()
    w.goto(1)
    assert w.view.candidates == []                     # candidates belong to the frame they were found in
    w._pick_candidate(w.lst_cand.item(0))
    gui.wait()
    assert w.k == 0 and 0 in w.p.track.keyframes
    cx, cy = w.p.track.keyframes[0][:2]
    assert abs(cx - gt[0][0]) < 2 and abs(cy - gt[0][1]) < 2
    assert not w.btn_mark.isChecked() and w.view.mode == "none"
    assert "[0]" in w.lbl_kf.text()
    assert w.timeline.keyframes == [0]
    assert w.view.box is not None and w.view.box[4] == "MANUAL" and w.view.keyframe_here


def test_mark_object_and_track_only(gui, clip):
    ensure_video(gui, clip)
    _, _, gt = clip
    w = gui.w
    w.p.track.keyframes.clear()
    w.goto(0)
    w.btn_mark.setChecked(True)
    assert w.view.mode == "roi"
    w.view.roiSelected.emit(gt[0][0], gt[0][1], 12, 12)
    assert w.p.track.keyframes == {0: (gt[0][0], gt[0][1], 12, 12)}
    assert "marked" in gui.status()
    w.run_tracking()
    gui.wait()
    tr = w.p.track
    assert tr.usable().mean() > 0.9
    assert "OK" in w.lbl_track.text() and tr.name in w.lbl_track.text()
    assert w.timeline.n == tr.n and w.timeline.status is tr.status
    assert "usable" in gui.status()
    w.goto(30)
    assert w.view.box[4] == "OK" and w.view.trail is not None and len(w.view.trail) > 10
    assert "track OK" in w.lbl_time.text()


def test_correct_position_and_keyframes(gui, clip):
    _, _, gt = clip
    w = gui.w
    tr = w.p.track
    w.goto(45)
    w.btn_correct.setChecked(True)
    assert w.view.mode == "point"
    w.view.pointClicked.emit(gt[45][0] + 0.3, gt[45][1] - 0.3)
    gui.wait()
    assert sorted(tr.keyframes) == [0, 45] and tr.status[45] == "MANUAL"
    assert w.timeline.keyframes == [0, 45] and "45" in w.lbl_kf.text()
    assert w.view.box[4] == "MANUAL" and w.view.keyframe_here
    w.btn_correct.setChecked(False)
    # keyframe on an excluded frame makes the frame usable again
    w.goto(60)
    w._toggle_exclude()
    gui.wait()
    assert 60 in tr.excluded and tr.status[60] == "EXCLUDED"
    w.view.pointClicked.emit(gt[60][0], gt[60][1])
    gui.wait()
    assert 60 not in tr.excluded and tr.status[60] == "MANUAL"
    w._remove_keyframe()
    gui.wait()
    assert 60 not in tr.keyframes and tr.status[60] == "OK"
    w._remove_keyframe()                               # no keyframe here any more
    assert "No keyframe" in gui.status()
    # excluding a keyframe frame is refused with an explanation
    w.goto(45)
    w._toggle_exclude()
    assert 45 not in tr.excluded and gui.pop_msgs()[-1][0] == "warning"
    w._remove_keyframe()
    gui.wait()
    assert sorted(tr.keyframes) == [0]


def test_start_end_exclude(gui):
    w = gui.w
    tr = w.p.track
    w.goto(5)
    w._start_here()                                    # keyframe 0 would fall outside -> refused
    assert tr.start == -1 and gui.pop_msgs()[-1][0] == "warning"
    w.goto(80)
    w._end_here()
    gui.wait()
    assert tr.end == 80 and (tr.status[81:] == "NONE").all() and tr.status[80] != "NONE"
    assert "ends at" in gui.status() or "usable" in gui.status()
    assert "Range" in w.lbl_kf.text()
    w._end_here()                                      # toggle off
    gui.wait()
    assert tr.end == -1 and tr.status[85] != "NONE"
    w.goto(40)
    w._toggle_exclude()
    gui.wait()
    assert tr.status[40] == "EXCLUDED" and "Excluded: [40]" in w.lbl_kf.text()
    w._toggle_exclude()
    gui.wait()
    assert tr.status[40] in ("OK", "REACQ") and not tr.excluded
    w.goto(0)


# ----------------------------------------------------------------------------- full analysis, one object
def test_full_analysis_single_object(gui, clip):
    ensure_video(gui, clip)
    _, truth, _ = clip
    w = gui.w
    w.sp_hfov.setValue(0.0)
    w.run_all()
    gui.wait()
    assert w.p.camera is not None and w.p.kin is not None
    assert w.tabs.currentIndex() == 4 and "FOV" in gui.status()
    assert w.txt_kin.toPlainText().startswith("[obj1, target]") and "pixel units" in w.txt_kin.toPlainText()
    assert w.fig_kin.fig is not None and w.fig_path.fig is not None and w.fig_cam.fig is not None
    assert w.txt_cam.toPlainText().startswith("FPN") and "Confidence" in w.txt_cam.toPlainText()
    assert w.timeline.conf is not None and np.isfinite(w.timeline.conf[1:]).all()
    assert "cam " in w.lbl_time.text()
    assert w.p.scen is None and w.tbl_scen.rowCount() == 0
    w.run_scenarios()
    assert w.tbl_scen.rowCount() == 0 and "FOV" in gui.pop_msgs()[-1][2]
    w.sp_hfov.setValue(truth["config"]["hfov_deg"])
    w.run_kinematics()
    gui.pump()
    txt = w.txt_kin.toPlainText()
    assert "deg/s" in txt and "Angular velocity" in txt
    s = w.p.kin.summary()
    assert abs(s["omega_median_deg_s"] - 0.158) < 0.02
    assert not gui.errors


def test_kinematics_parameters(gui):
    w = gui.w
    w.sp_win.setValue(7)
    w.sp_order.setValue(3)
    w.cmb_timing.setCurrentText("container")
    w.sp_start.setValue(10)
    w.sp_end.setValue(70)
    w.chk_usecam.setChecked(False)
    w.run_kinematics()
    kp = w.p.kin_params
    assert (kp.window_frames, kp.poly_order, kp.timing, kp.start_frame, kp.end_frame, kp.use_camera) == \
        (7, 3, "container", 10, 70, False)
    assert w.p.kin.summary()["n_frames_used"] <= 61
    assert "ASSUMED" in w.txt_kin.toPlainText()        # static camera assumption is announced
    w.chk_usecam.setChecked(True)
    w.sp_win.setValue(21)
    w.sp_order.setValue(2)
    w.cmb_timing.setCurrentText("nominal")
    w.sp_start.setValue(0)
    w.sp_end.setValue(-1)
    w.run_kinematics()
    assert w.p.kin.params.use_camera and w.p.kin.summary()["n_frames_used"] > 80


def test_scenarios_all_modes_and_errors(gui):
    w = gui.w
    w.cmb_mode.setCurrentText("range")
    w.ed_values.setText("1km, 5km, 10nmi")
    w.run_scenarios()
    assert w.tbl_scen.rowCount() == 3 and w.tbl_scen.columnCount() > 5 and w.fig_sweep.fig is not None
    assert "[obj1]" in w.txt_scen.toPlainText()
    w.cmb_mode.setCurrentText("size")
    assert w.lbl_values.text() == "Object sizes" and "m" in w.ed_values.text()
    w.run_scenarios()
    assert w.tbl_scen.rowCount() == 5 and "apparent" in w.txt_scen.toPlainText()
    w.cmb_mode.setCurrentText("altitude")
    w.ed_camalt.setText("1000m")
    w.sp_elev.setValue(10.0)
    w.run_scenarios()
    assert w.tbl_scen.rowCount() == 5 and "flat-earth" in w.txt_scen.toPlainText()
    w.sp_elev.setValue(0.0)
    w.run_scenarios()                                  # elevation 0 -> ValueError shown, table cleared
    assert "Elevation" in gui.pop_msgs()[-1][2] and w.tbl_scen.rowCount() == 0 and w.fig_sweep.fig is None
    w.cmb_mode.setCurrentText("range")
    for bad in ("abc", "1km:2km", "", "5xx"):
        w.ed_values.setText(bad)
        w.run_scenarios()
        kind, _, text = gui.pop_msgs()[-1]
        assert kind == "warning" and "Accepted forms" in text, bad
    w.ed_values.setText("2km:6km:2km")
    w.sp_ovx.setValue(10.0)
    w.run_scenarios()
    assert w.tbl_scen.rowCount() == 3 and "observer velocity" in w.txt_scen.toPlainText()
    w.sp_ovx.setValue(0.0)
    # no trusted camera frames -> the explanatory ValueError must be shown, not raised
    w.p.kin.trusted[:] = False
    w.run_scenarios()
    assert "trusted camera-motion" in gui.pop_msgs()[-1][2] and w.tbl_scen.rowCount() == 0
    assert "Not computed" in w.txt_scen.toPlainText()
    w.run_kinematics()
    gui.pump()
    assert w.p.scen is None
    w.run_scenarios()
    assert w.tbl_scen.rowCount() == 3


def test_figure_popout_restores_embedded_figure(gui):
    w = gui.w
    holder = w.fig_kin
    fig = holder.fig
    assert fig is not None and holder.btn.isEnabled()
    holder._pop()
    gui.pump()
    dlg = holder.popup
    assert isinstance(dlg, QDialog) and dlg.isVisible()
    assert holder.scroll.widget().__class__.__name__ == "QLabel"       # placeholder while popped out
    holder._pop()                                                      # second click: no second window
    assert holder.popup is dlg
    dlg.close()
    gui.pump()
    assert holder.popup is None and holder.fig is fig
    assert holder.scroll.widget().__class__.__name__ == "FigureCanvasQTAgg"
    # a figure replaced while the pop-out is open stays after the pop-out closes
    holder._pop()
    gui.pump()
    dlg = holder.popup
    w.run_kinematics()
    gui.pump()
    new = holder.fig
    assert new is not fig
    dlg.close()
    gui.pump()
    assert holder.fig is new and holder.scroll.widget().__class__.__name__ == "FigureCanvasQTAgg"
    assert not gui.errors


def test_camera_alone_and_mask_invalidation(gui):
    w = gui.w
    w.run_camera()
    gui.wait()
    assert w.p.camera is not None and w.fig_cam.fig is not None and w.p.kin is not None
    assert "Camera motion estimated" in gui.status()
    w.view.maskDrawn.emit(300, 200, 10, 10)
    assert w.p.camera is None and w.p.kin is None and w.p.scen is None
    assert w.fig_cam.fig is None and w.fig_kin.fig is None and w.tbl_scen.rowCount() == 0
    assert "Masks changed" in w.txt_kin.toPlainText() and w.timeline.conf is None
    w.lst_masks.setCurrentRow(w.lst_masks.count() - 1)
    w._delete_mask()
    assert len(w.p.masks) == 4


# ----------------------------------------------------------------------------- several objects
def test_multi_object_analysis_and_active_switch(gui, clip):
    ensure_video(gui, clip)
    _, truth, gt = clip
    w = gui.w
    if not w.p.track.keyframes:
        w.p.track.keyframes[0] = (gt[0][0], gt[0][1], 12, 12)
        w.sp_hfov.setValue(truth["config"]["hfov_deg"])
    w.run_all()
    gui.wait()
    assert w.p.kin is not None
    w._add_object()
    assert [t.name for t in w.p.tracks] == ["obj1", "obj2"] and w.p.active == 1
    assert w.cmb_obj.currentText() == "obj2 (target)" and w.btn_mark.isChecked()
    assert w.p.kins[0] is not None and w.p.kins[1] is None          # obj1's results survive
    assert "No kinematics for obj2" in w.txt_kin.toPlainText() and w.tbl_scen.rowCount() == 0
    w.goto(0)
    w.view.roiSelected.emit(gt[0][0] + 30, gt[0][1] + 30, 10, 10)    # a second (fake) target in the background
    w._add_object()
    w.cmb_role.setCurrentText("control")
    assert w.p.tracks[2].role == "control" and w.cmb_obj.currentText() == "obj3 (control)"
    assert "control" in gui.status()
    w.view.roiSelected.emit(60, 150, 12, 12)
    w.run_all()
    gui.wait()
    assert [k.role if k else None for k in w.p.kins] == ["target", "target", "control"]
    assert w.txt_kin.toPlainText().startswith("[obj3, control]")
    assert w.p.scen is not None and w.tbl_scen.rowCount() > 0 and "[obj3]" in w.txt_scen.toPlainText()
    assert {o[2] for o in w.view.others} == {"obj1", "obj2"}
    w.cmb_obj.setCurrentIndex(0)
    gui.pump()
    assert w.p.active == 0 and w.txt_kin.toPlainText().startswith("[obj1, target]")
    assert "[obj1]" in w.txt_scen.toPlainText() and w.tbl_scen.rowCount() > 0
    assert {o[2] for o in w.view.others} == {"obj2", "obj3 (control)"}
    assert "obj1" in w.lbl_kf.text() and w.cmb_role.currentText() == "target"
    assert w.fig_kin.fig is not None and w.fig_path.fig is not None
    # role change updates labels and overlays
    w.cmb_obj.setCurrentIndex(1)
    gui.pump()
    w.cmb_role.setCurrentText("control")
    assert w.p.tracks[1].role == "control" and w.cmb_obj.currentText() == "obj2 (control)"
    assert w.txt_kin.toPlainText().startswith("[obj2, control]")
    w.cmb_role.setCurrentText("target")
    # delete the active object: results of the others stay, the display follows the new active object
    w.cmb_obj.setCurrentIndex(2)
    gui.pump()
    w._delete_object()
    assert [t.name for t in w.p.tracks] == ["obj1", "obj2"] and len(w.p.kins) == 2
    assert w.p.active == 1 and w.txt_kin.toPlainText().startswith("[obj2, target]")
    assert w.cmb_obj.count() == 2
    w._delete_object()
    assert [t.name for t in w.p.tracks] == ["obj1"] and w.p.active == 0
    assert w.txt_kin.toPlainText().startswith("[obj1, target]")
    w._delete_object()
    assert len(w.p.tracks) == 1 and "cannot be deleted" in gui.status()
    assert w.view.others == []
    assert not gui.errors


# ----------------------------------------------------------------------------- export and project files
def test_export_and_project_roundtrip(gui, clip, tmp_path):
    ensure_video(gui, clip)
    path, truth, _ = clip
    w = gui.w
    out = tmp_path / "export"
    assert w.export_to(str(out), video=True)
    gui.wait()
    files = [w.lst_files.item(i).text() for i in range(w.lst_files.count())]
    names = {os.path.basename(f) for f in files}
    assert {"report.html", "report.md", "project.json", "per_frame.csv", "annotated.mp4"} <= names
    assert all(os.path.exists(f) for f in files)
    assert w.tabs.currentIndex() == 6 and "Exported" in gui.status()
    # every widget round-trips through the project file
    w.sp_radius.setValue(33)
    w.sp_lost.setValue(0.6)
    w.sp_tupd.setValue(0.1)
    w.chk_centroid.setChecked(False)
    w.chk_reacq.setChecked(False)
    w.cmb_tmode.setCurrentText("template")
    w.sp_snr.setValue(6.5)
    w.cmb_pol.setCurrentText("dark")
    w.cmb_preset.setCurrentText(w.cmb_preset.itemText(w.cmb_preset.count() - 1))
    w.cmb_fpn.setCurrentText("off")
    w.chk_destripe.setChecked(False)
    w.chk_refine.setChecked(False)
    w.sp_smooth.setValue(3.5)
    w.cmb_gop.setCurrentText("off")
    w.sp_maxw.setValue(640)
    w.sp_hfov.setValue(1.25)
    w.sp_par.setValue(1.1)
    w.sp_win.setValue(11)
    w.sp_order.setValue(1)
    w.cmb_timing.setCurrentText("container")
    w.sp_start.setValue(3)
    w.sp_end.setValue(70)
    w.chk_usecam.setChecked(False)
    w.cmb_mode.setCurrentText("altitude")
    w.ed_values.setText("3kft, 9kft")
    w.ed_camalt.setText("25000ft")
    w.sp_elev.setValue(-26.0)
    w.sp_ovx.setValue(12.5)
    w.sp_ovy.setValue(-3.0)
    proj = tmp_path / "p.json"
    assert w.save_project(str(proj))
    saved = json.load(open(proj))
    expected = {k: v for k, v in saved.items() if k in ("track_params", "camera_params", "kin_params", "scenario")}
    assert saved["tracks"][0]["keyframes"] and saved["masks"]
    # scramble the widgets, then reload the project: everything must come back
    w.sp_radius.setValue(24)
    w.sp_hfov.setValue(0.0)
    w.chk_usecam.setChecked(True)
    w.cmb_mode.setCurrentText("range")
    w.sp_start.setValue(0)
    w.sp_end.setValue(-1)
    w.cmb_tmode.setCurrentText("auto")
    w.p.masks.clear()
    w.p.track.keyframes.clear()
    assert w.open_project(str(proj))
    gui.wait()
    w._sync_params()
    got = {k: v for k, v in w.p.to_dict().items() if k in expected}
    assert got == expected
    assert w.p.track.keyframes and len(w.p.masks) == len(saved["masks"])
    assert w.sp_seg0.value() == 0.0 and w.cmb_role.currentText() == "target"
    assert "Keyframes present" in gui.status()
    assert w.txt_kin.toPlainText() == "" and w.fig_kin.fig is None     # no results until re-run
    assert w.lst_files.count() == 0
    # open_path dispatches .json to the project loader
    assert w.open_path(str(proj)) is None and w.job is not None
    gui.wait()
    # a project whose video is missing: locate callback (abort, then success), no dialog
    d = json.load(open(proj))
    d["video_path"] = "/nonexistent/clip.avi"
    p2 = tmp_path / "p2.json"
    json.dump(d, open(p2, "w"))
    asked = []
    assert w.open_project(str(p2), locate=lambda m: (asked.append(m), "")[1]) is False
    assert asked == ["/nonexistent/clip.avi"] and "not located" in gui.status()
    assert w.open_project(str(p2), locate=lambda m: path)
    gui.wait()
    assert w.p.video_path == path and w.p.track.keyframes
    # a project next to its (renamed) video is found without asking
    d["video_path"] = "elsewhere/" + os.path.basename(path)
    p3 = tmp_path / "p3.json"
    json.dump(d, open(p3, "w"))
    os.symlink(path, tmp_path / os.path.basename(path))
    assert w.open_project(str(p3), locate=lambda m: pytest.fail("should not ask"))
    gui.wait()
    # broken project files
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert w.open_project(str(bad)) is False
    assert w.open_project(str(tmp_path / "missing.json")) is False
    assert all(m[0] == "warning" for m in gui.pop_msgs()[-2:])
    # restore the defaults used by the remaining tests
    w.sp_hfov.setValue(truth["config"]["hfov_deg"])
    w.chk_usecam.setChecked(True)
    w.cmb_tmode.setCurrentText("auto")
    w.cmb_pol.setCurrentText("auto")
    w.sp_start.setValue(0)
    w.sp_end.setValue(-1)
    w.cmb_mode.setCurrentText("range")
    w.cmb_fpn.setCurrentText("auto")


def test_reload_segment_remaps_keyframes(gui, clip):
    ensure_video(gui, clip)
    path, _, gt = clip
    w = gui.w
    fps = w.p.meta.fps
    w.p.track.keyframes = {0: (gt[0][0], gt[0][1], 12, 12), 45: (gt[45][0], gt[45][1], 12, 12)}
    w.goto(50)
    w._toggle_exclude()
    w.run_tracking()
    gui.wait()
    w.run_kinematics()
    gui.pump()
    assert w.p.kin is not None
    w.sp_seg0.setValue(1.0)
    w.sp_seg1.setValue(2.5)
    w.reload_segment()
    gui.wait()
    n = len(w.p.cache)
    assert n == round(1.5 * fps) + 1 or abs(n - 1.5 * fps) <= 2
    off = w.p.cache.offset
    assert off == round(1.0 * fps)
    assert w.slider.maximum() == n - 1 and w.k == 0
    assert w.sp_start.maximum() == off + n - 1
    assert sorted(w.p.track.keyframes) == [45 - off] and w.p.track.excluded == {50 - off}
    assert "remapped" in w.txt_meta.toPlainText()
    assert w.p.camera is None and w.p.kins == [] and w.p.scen is None
    assert w.txt_kin.toPlainText() == "" and w.fig_kin.fig is None and w.timeline.n == n
    assert "file frame 30" in w.lbl_time.text()
    w.run_all()
    gui.wait()
    assert w.p.kin is not None and int(w.p.kin.frame[0]) == off
    # a segment with no keyframes inside: objects reset, explained in the warnings
    w.sp_seg0.setValue(0.0)
    w.sp_seg1.setValue(0.4)
    w.reload_segment()
    gui.wait()
    assert not w.p.track.keyframes and "no keyframes" in w.txt_meta.toPlainText()
    # an impossible segment is refused
    w.sp_seg0.setValue(2.0)
    w.sp_seg1.setValue(1.0)
    assert w.reload_segment() is None and gui.pop_msgs()[-1][0] == "warning"
    w.sp_seg0.setValue(0.0)
    w.sp_seg1.setValue(0.0)
    w.reload_segment()
    gui.wait()
    assert len(w.p.cache) == len(gt)


def test_bad_video_keeps_previous_state(gui, clip):
    ensure_video(gui, clip)
    path, _, gt = clip
    w = gui.w
    w.p.track.keyframes = {0: (gt[0][0], gt[0][1], 12, 12)}
    n = len(w.p.cache)
    assert w.open_video(__file__)                       # not a video: decoding fails in the worker
    gui.wait()
    kind, title, text = gui.pop_msgs()[-1]
    assert kind == "error" and "could not open" in text.lower()
    assert w.p.cache is not None and len(w.p.cache) == n and os.path.samefile(w.p.video_path, path)
    assert w.p.track.keyframes and w.slider.isEnabled()
    # opening a second video replaces the project but keeps the parameters
    w.sp_radius.setValue(31)
    w.open_video(path)
    gui.wait()
    assert w.p.track_params.search_radius_px == 31 and not w.p.track.keyframes and not w.p.masks
    assert w.lst_masks.count() == 0 and w.cmb_obj.count() == 1


# ----------------------------------------------------------------------------- input handling
def test_keyboard_mouse_and_drop(gui, clip):
    ensure_video(gui, clip)
    path, _, _ = clip
    w, v = gui.w, gui.w.view
    w.goto(0)
    QTest.keyClick(w, Qt.Key_Right)
    assert w.k == 1
    QTest.keyClick(w, Qt.Key_Left)
    assert w.k == 0
    QTest.keyClick(w, Qt.Key_End)
    assert w.k == len(w.p.cache) - 1
    QTest.keyClick(w, Qt.Key_Home)
    assert w.k == 0
    QTest.keyClick(w, Qt.Key_Space)
    assert w.play_timer.isActive() and w.btn_play.text() != "▶"
    QTest.keyClick(w, Qt.Key_Space)
    assert not w.play_timer.isActive()
    v.setFocus()
    QTest.keyClick(v, Qt.Key_1)
    assert v.zoom == 1.0
    QTest.keyClick(v, Qt.Key_F)
    assert v.zoom == 0.0
    QTest.keyClick(v, Qt.Key_Right)                    # unhandled keys propagate to the window
    assert w.k == 1
    w.btn_mark.setChecked(True)
    QTest.keyClick(v, Qt.Key_Escape)
    assert v.mode == "none" and not w.btn_mark.isChecked()
    # wheel zoom keeps the image point under the cursor fixed
    pos = QPointF(v.width() / 2 + 50, v.height() / 2 + 20)
    before = v.to_img(pos)
    ev = QWheelEvent(pos, v.mapToGlobal(pos.toPoint()), QPoint(0, 0), QPoint(0, 120), Qt.NoButton, Qt.NoModifier,
                     Qt.ScrollPhase.NoScrollPhase, False)
    _app.sendEvent(v, ev)
    after = v.to_img(pos)
    assert v.zoom > 0 and abs(before.x() - after.x()) < 1e-6 and abs(before.y() - after.y()) < 1e-6
    # right-drag pans
    c0 = QPointF(v.center)
    QTest.mousePress(v, Qt.RightButton, Qt.NoModifier, pos.toPoint())
    QTest.mouseMove(v, (pos + QPointF(30, 10)).toPoint())
    QTest.mouseRelease(v, Qt.RightButton, Qt.NoModifier, (pos + QPointF(30, 10)).toPoint())
    assert v.center.x() < c0.x() and v.center.y() < c0.y() and v._pan_start is None
    v.zoom_fit()
    # left-drag draws a box (roi) or a mask
    w.p.track.keyframes.clear()
    def drag(x0, y0, x1, y1):                           # image coordinates -> left-button drag in the widget
        a, b = v.to_widget(x0, y0).toPoint(), v.to_widget(x1, y1).toPoint()
        QTest.mousePress(v, Qt.LeftButton, Qt.NoModifier, a)
        QTest.mouseMove(v, b)
        QTest.mouseRelease(v, Qt.LeftButton, Qt.NoModifier, b)

    w.btn_mark.setChecked(True)
    drag(100, 100, 114, 112)
    gui.wait()
    assert w.k in w.p.track.keyframes and not w.btn_mark.isChecked()
    cx, cy, bw, bh = w.p.track.keyframes[w.k]
    assert abs(cx - 107) < 1 and abs(cy - 106) < 1 and abs(bw - 14) < 1 and abs(bh - 12) < 1
    nm = len(w.p.masks)
    w.btn_draw_mask.setChecked(True)
    drag(-5, -5, 40, 30)                                # starts outside the image: clipped to it
    assert len(w.p.masks) == nm + 1 and w.p.masks[-1].to_dict()["x"] == 0 and w.p.masks[-1].to_dict()["w"] == 40
    w.btn_draw_mask.setChecked(False)
    # a tiny drag is ignored
    w.btn_mark.setChecked(True)
    drag(200, 200, 201, 201)
    assert len(w.p.track.keyframes) == 1
    w.btn_mark.setChecked(False)
    # timeline click jumps
    w.timeline.frameClicked.emit(40)
    assert w.k == 40
    QTest.mouseClick(w.timeline, Qt.LeftButton, Qt.NoModifier, QPoint(w.timeline.width() // 2, 5))
    assert abs(w.k - len(w.p.cache) // 2) <= 1
    # drag and drop of a video file

    class Drop:
        def __init__(self, p):
            self.md = QMimeData()
            self.md.setUrls([QUrl.fromLocalFile(p)])
            self.accepted = False

        def mimeData(self):
            return self.md

        def acceptProposedAction(self):
            self.accepted = True

    d = Drop(path)
    w.dragEnterEvent(d)
    assert d.accepted
    w.dropEvent(d)
    assert w.job is not None
    gui.wait()
    assert w.p.video_path == path and not w.p.track.keyframes
    w.grab()                                            # paints view, timeline and overlays
    assert not gui.errors


def test_timeline_paints_every_status(gui):
    w = gui.w
    statuses = [tracker.OK, tracker.MANUAL, tracker.REACQ, tracker.LOST, tracker.WEAK, tracker.AMBIG,
                tracker.EXCLUDED, tracker.DUP, tracker.BLANK]
    assert all(s in STATUS_COLORS for s in statuses)
    n = len(w.p.cache)
    st = np.array([statuses[k % len(statuses)] for k in range(n)], dtype=object)
    conf = np.linspace(0, 1, n)
    conf[3] = np.nan
    w.timeline.set_data(n, st, conf, [0, 5, n - 1])
    w.timeline.grab()
    for k, s in enumerate(statuses):
        w.view.box = (50.0, 50.0, 10.0, 10.0, s)
        w.view.keyframe_here = k % 2 == 0
        w.view.grab()
    w.timeline.set_data(0)
    assert w.timeline.status is None
    w._refresh_track_info()
    assert not gui.errors


# ----------------------------------------------------------------------------- jobs
def test_busy_feedback_cancel_and_close(gui, clip):
    ensure_video(gui, clip)
    _, _, gt = clip
    w = gui.w
    w.p.track.keyframes = {0: (gt[0][0], gt[0][1], 12, 12)}
    w.run_all()
    assert w.job is not None and w.progress.isVisible() and w.btn_cancel.isVisible() and not w.tabs.isEnabled()
    assert w.run_camera() is None and "Still busy" in gui.status()      # second click is answered, not swallowed
    assert w._run_job("x", lambda p, c: None, lambda r: None) is False
    w.cancel_job()
    gui.wait()
    assert w.job is None and "Cancelled" in gui.status() and w.tabs.isEnabled() and not w.progress.isVisible()
    assert not [m for m in gui.pop_msgs() if m[0] == "error"]
    w.run_tracking()                                    # back to back after a cancel
    gui.wait()
    assert w.p.track.usable().any()
    w.run_all()
    assert w.job is not None
    w.close()                                           # close while a job runs: cancels, waits, closes the cache
    gui.pump()
    assert w.job is None and w.p.cache is None
    assert not gui.errors
