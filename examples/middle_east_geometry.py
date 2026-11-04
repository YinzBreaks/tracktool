"""Middle East Object (MQ-9, 2022): viewing geometry from a satellite map registered to the video mosaic.

What this measures, and how. The plane-resection method (uapka.fov.estimate_ground) cannot determine the FOV
here: the drone is ~10 km from a scene a few hundred metres across, so the ground maps to the image by an
affine transform (no usable perspective). That affine map still carries three things: the orientation of the
image (where North points), the metres per pixel across and along the look direction, and their ratio, which
is sin(look-down angle). The FOV and the slant range then follow from an ASSUMED altitude.

Inputs (all in this file):
  * five correspondences between the video, in the camera track's reference pixels (the mosaic of frames
    270-430 built from `middle_east_project.json`'s camera track), and a north-up Google Earth screenshot:
    the courtyard's tree, the two ends of the trench along its east berm, and the two ends of its north berm;
  * the screenshot's georeference: an affine fit through five pins whose coordinates were read in Google
    Earth (self-consistent to 1 px; 2.01 screenshot px per metre), origin at Bellingcat's aim point A
    (35.369189, 40.332815), which the first frames' reticle sits on;
  * the camera track's cumulative pan between Bellingcat's timestamps (7.57 s and 23.59 s).
The five correspondences were refined by maximising the gradient correlation between the warped map and the
mosaic; the refined 2x3 map (metres -> reference px) is embedded below. Everything printed is labelled.
"""
import csv, math, sys
import numpy as np

# metres east/north of Bellingcat's point A  ->  reference pixels (x right, y down) of the camera track
J = np.array([[-5.67690, -1.13255], [1.05708, 4.78686]])   # d(ref px)/d(E, N)       [MEASURED: map<->mosaic]
# (J = M[:, :2] @ diag(pxm, -pxm) with the 8.webp screenshot at pxm = 2.0104 px/m; see the session notes)
RETICLE_270_REF = np.array([1338.4 - 731.5, 216.2 + 116.8])  # not used numerically; frame-270 reticle, ref px
PAN_A_TO_B_REF = np.array([-1564.4, 4254.2])  # aim displacement, ref px, 7.57 s -> 23.59 s  [MEASURED: camera track]
BELLINGCAT_AB_M, BELLINGCAT_AB_HDG = 750.9, 44.98       # [EXTERNAL GEOLOCATION]
GROUND_ALT_M = 229.0                                     # Google Earth terrain at the scene
INSET_W, INSET_H = 558, 720                              # live sensor area of the 1280x720 release

U, S, Vt = np.linalg.svd(J)
g_across, g_along = 1.0 / S[0], 1.0 / S[1]
ratio = S[1] / S[0]
elev = math.degrees(math.asin(ratio))
ax = U[:, 1]                                             # most compressed image axis = line toward the camera
ax_img = math.degrees(math.atan2(ax[0], -ax[1])) % 360
Ji = np.linalg.inv(J)
g = Ji @ ax
look_hdg = math.degrees(math.atan2(g[0], g[1])) % 360
north = J @ np.array([0.0, 1.0]); east = J @ np.array([1.0, 0.0])
north_img = math.degrees(math.atan2(north[0], -north[1])) % 360
east_img = math.degrees(math.atan2(east[0], -east[1])) % 360

print("VIEWING GEOMETRY, frame 270 (t = 9.0 s)  [MEASURED from the map-to-mosaic affine; square pixels assumed]")
print(f"  metres per reference pixel: {g_across:.4f} across the look direction, {g_along:.4f} along it")
print(f"  foreshortening ratio {ratio:.3f}  ->  look-down angle {elev:.1f} deg (estimated +/- 5 deg from the fit)")
print(f"  compressed image axis {ax_img:.0f} deg clockwise from image-up (= the HUD's red arrow direction, "
      f"{(360 - ax_img + 180) % 360:.0f} deg left of up): the line toward the aircraft")
print(f"  ground heading of that line {look_hdg:.0f} or {(look_hdg + 180) % 360:.0f} deg (affine: near/far ambiguous)")
print(f"  North points {north_img:.0f} deg, East {east_img:.0f} deg clockwise from image-up: the video is roughly south-up")
# pan check against Bellingcat
d = Ji @ PAN_A_TO_B_REF
print(f"\nPAN CHECK: aim displacement 7.57 s -> 23.59 s: {np.linalg.norm(PAN_A_TO_B_REF):.0f} ref px = "
      f"{np.linalg.norm(d):.0f} m at heading {math.degrees(math.atan2(d[0], d[1])) % 360:.0f} deg "
      f"[camera track x affine]; Bellingcat: {BELLINGCAT_AB_M:.0f} m at {BELLINGCAT_AB_HDG:.0f} deg. "
      "The chain's confidence is low after 15 s (fields): the 38 deg / 16% disagreement is unresolved.")
print("\nFOV AND RANGE vs ASSUMED ALTITUDE (the affine fixes everything except the overall distance):")
print(f"  {'altitude MSL':>14} {'AGL':>7} {'slant range':>12} {'ground dist':>12} {'HFOV inset':>11} {'HFOV 1280px':>12} {'VFOV':>7}")
for ft in (15000, 20000, 25000, 30000):
    agl = ft * 0.3048 - GROUND_ALT_M
    R = agl / math.sin(math.radians(elev))
    hf = math.degrees(2 * math.atan(0.5 * INSET_W * g_across / R)); hf_full = math.degrees(2 * math.atan(0.5 * 1280 * g_across / R))
    vf = math.degrees(2 * math.atan(0.5 * INSET_H * g_along * math.sin(math.radians(elev)) / R))
    print(f"  {ft:>11,d} ft {agl:>6.0f} m {R / 1e3:>9.2f} km {R * math.cos(math.radians(elev)) / 1e3:>9.2f} km "
          f"{hf:>10.3f}° {hf_full:>11.3f}° {vf:>6.3f}°")

if len(sys.argv) > 1:
    rows = list(csv.DictReader(open(sys.argv[1])))
    col = lambda n: next(c for c in rows[0] if c.split(" ")[0] == n)
    t = np.array([float(r[col("t_s")]) for r in rows])
    tr = np.array([r["in_summaries"] == "1" for r in rows])
    sx = np.array([float(r[col("stab_x_refpx")] or "nan") for r in rows])
    sy = np.array([float(r[col("stab_y_refpx")] or "nan") for r in rows])
    sp = np.array([float(r[col("stab_speed_refpx_s")] or "nan") for r in rows])
    ok = tr & np.isfinite(sp) & np.isfinite(sx)
    val = np.isfinite(sx)
    vel = []
    for i in np.flatnonzero(ok):                       # direction from a local linear fit, +/- 0.3 s
        m = val & (np.abs(t - t[i]) <= 0.3)
        if m.sum() < 4:
            continue
        A = np.column_stack([t[m] - t[i], np.ones(m.sum())])
        vx = np.linalg.lstsq(A, sx[m], rcond=None)[0][0]; vy = np.linalg.lstsq(A, sy[m], rcond=None)[0][0]
        vel.append(Ji @ np.array([vx, vy]))
    vel = np.array(vel)
    spd = np.linalg.norm(vel, axis=1); hdg = np.degrees(np.arctan2(vel[:, 0], vel[:, 1])) % 360
    s_med = float(np.median(spd))
    print(f"\nOBJECT, ground-relative, {len(spd)} trusted frames [MEASURED px/s x measured anisotropic scale]:")
    print(f"  speed median {s_med:.1f} m/s (p5-p95 {np.percentile(spd, 5):.1f}-{np.percentile(spd, 95):.1f}); "
          f"heading median {np.median(hdg):.0f} deg; the isotropic 0.165 m/px shortcut gave {0.165 * np.nanmedian(sp[ok]):.1f} m/s")
    print("  (the motion is along the image vertical, where the scale is 0.25 m/px, not 0.165)")
    el = elev; agl = 25000 * 0.3048 - GROUND_ALT_M; R_g = agl / math.sin(math.radians(el))
    print(f"\nPARALLAX at the measured look-down angle {el:.1f} deg, 25,000 ft ASSUMED (R_ground {R_g / 1e3:.1f} km, "
          f"LOS rate {math.degrees(s_med / R_g):.3f} deg/s):")
    H = [0, 500, 1000, 2000, 3000, 4000, 5000]
    print("  " + " ".join(f"{h:>6d} m" for h in H) + "  | stationary at (AGL)")
    for v_kt in (150, 170):
        v = v_kt * 0.5144
        for lab, vp in (("LOS in the direction of flight", v * math.sin(math.radians(el))),
                        ("LOS 90 deg off the heading", v)):
            vals = [(1 - h / agl) * s_med - vp * (h / agl) for h in H]
            print("  " + " ".join(f"{x:>+8.0f}" for x in vals) + f"  | {agl * s_med / (s_med + vp):>5.0f} m   {v_kt} kt, {lab}")
    print("  A stationary object needs the aircraft's lateral motion anti-parallel to the object's image motion: the")
    print(f"  object moves {np.median(hdg):.0f} deg on the ground, so that requires an aircraft heading near "
          f"{(np.median(hdg) + 180) % 360:.0f} deg. The heading is not in the video; Bellingcat's 'toward the scene' is one guess.")
