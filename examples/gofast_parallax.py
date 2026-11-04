"""GOFAST parallax correction from HUD readouts (read by eye from the frames) and the tool's sea-relative LOS rates."""
import sys, csv, numpy as np
rows = list(csv.DictReader(open(sys.argv[1])))
def col(name): return next(c for c in rows[0] if c.split(" ")[0] == name)
g = lambda n: np.array([float(r[col(n)] or "nan") for r in rows])
t = g("t_s"); wx = np.radians(g("omega_x_deg_s")); wy = np.radians(g("omega_y_deg_s")); sw = np.radians(g("omega_sigma_deg_s"))
trusted = np.array([r["in_summaries"] == "1" for r in rows])
# --- HUD readouts (gimbal azimuth left of nose, elevation below horizon, laser/radar range), by eye
T_az = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 24, 28, 32]
AZ = [35, 36, 36, 37, 37, 39, 39, 39, 40, 41, 42, 42, 43, 44, 45, 46, 47, 49, 51, 54, 57]
EL = [22, 22, 23, 23, 23, 24, 24, 24, 24, 25, 25, 26, 26, 26, 27, 27, 27, 29, 31, 33, 34]
T_rng = [13, 14, 15, 16, 20, 24, 28, 32]; RNG = [4.4, 4.3, 4.3, 4.2, 4.0, 3.8, 3.6, 3.4]
az = np.radians(np.interp(t, T_az, AZ)); el = np.radians(np.interp(t, T_az, EL))
rng_fit = np.polyfit(T_rng, RNG, 1); R = np.polyval(rng_fit, t) * 1852.0        # linear fit, extrapolated before lock
alt = 25000 * 0.3048; a_sound = 20.05 * np.sqrt(288.15 - 0.0065 * alt); v = 0.61 * a_sound
R_sea = alt / np.sin(el)
v_right, v_up, v_along = v * np.sin(az), v * np.cos(az) * np.sin(el), v * np.cos(az) * np.cos(el)
print(f"TAS {v:.0f} m/s; along-LOS component {v_along.min():.0f}-{v_along.max():.0f} m/s vs HUD closure 150-210 kt = 77-108 m/s "
      f"(closure = along-LOS speed minus the object's radial motion)")
print(f"range fit: {rng_fit[0]*1852:.1f} m/s range rate ({rng_fit[0]*1852/0.5144:.0f} kt), R(13 s)={np.polyval(rng_fit, 13)*1852:.0f} m; sea slant range {R_sea.min()/1e3:.1f}-{R_sea.max()/1e3:.1f} km")
ok = trusted & np.isfinite(wx) & (t >= 12)
# direction test: stationary object => omega_rel anti-parallel to (v_right, v_up)
ang_w = np.degrees(np.arctan2(wy[ok], wx[ok])); ang_v = np.degrees(np.arctan2(v_up[ok], v_right[ok]))
d = (ang_w - ang_v + 180 + 180) % 360 - 180      # 0 = exactly anti-parallel
print(f"sea-relative LOS motion direction: median {np.median(ang_w):.1f} deg (image, 0=right, 90=up); aircraft lateral velocity direction {np.median(ang_v):.1f} deg;"
      f" deviation from anti-parallel: median {np.median(d):+.1f} deg, p5-p95 {np.percentile(d, 5):+.1f}..{np.percentile(d, 95):+.1f}")
# correction: v_obj_perp = R*omega_rel + v_ac_perp*(1 - R/R_sea)   (all across the LOS, image axes)
bg = 1 - R / R_sea
vox = R * wx + v_right * bg; voy = R * wy + v_up * bg
naive = R * np.hypot(wx, wy); corr = np.hypot(vox, voy)
sig = R * sw
print(f"naive R*omega (background at infinity, no observer motion): median {np.median(naive[ok]):.1f} m/s ({np.median(naive[ok])/0.5144:.0f} kt)")
print(f"parallax-corrected object speed across the LOS: median {np.median(corr[ok]):.1f} m/s ({np.median(corr[ok])/0.5144:.0f} kt), p5-p95 {np.percentile(corr[ok],5):.1f}..{np.percentile(corr[ok],95):.1f}; "
      f"1-sigma from LOS-rate noise alone ~{np.median(sig[ok]):.1f} m/s")
print(f"stationary-object prediction of the sea-relative rate: {np.degrees(np.median((np.hypot(v_right, v_up) * (1/R - 1/R_sea))[ok])):.3f} deg/s vs measured {np.degrees(np.median(np.hypot(wx, wy)[ok])):.3f} deg/s")
# sensitivity: FOV +-20%, sea range +-20%, TAS +-5%
for label, fx in (("FOV +20%", (1.2, 1, 1)), ("FOV -20%", (0.8, 1, 1)), ("sea range +20%", (1, 1.2, 1)), ("sea range -20%", (1, 0.8, 1)), ("TAS +5%", (1, 1, 1.05)), ("TAS -5%", (1, 1, 0.95))):
    f, s_, tv = fx; bg2 = 1 - R / (R_sea * s_)
    c2 = np.hypot(R * wx * f + v_right * tv * bg2, R * wy * f + v_up * tv * bg2)
    print(f"   {label:15s}: corrected median {np.median(c2[ok]):.1f} m/s")
np.save(sys.argv[2] + "/parallax_series.npy", np.c_[t, corr, naive, ok])
