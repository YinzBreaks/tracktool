"""Middle East Object (MQ-9, 12 Jul 2022): parallax correction under the MQ-9 cruise assumptions.

No HUD range, altitude or FOV survives redaction, so this follows Bellingcat's assumptions: cruise at
25,000 ft, 150-170 kt, flying roughly toward the scene it looks at. The object's height above ground is
unknown and is swept. Inputs from the tool: the object's LOS speed relative to the ground texture in
reference pixels (per_frame.csv, camera-compensated 'stab_speed'), and the ground scale 0.165 m/px from
Bellingcat's geolocated aim points A (7.57 s) and B (23.59 s), 750.9 m apart.

Geometry (velocity horizontal, LOS depressed by `dep`, object on the LOS at height h above ground):
  R_g   = A / sin(dep)                      range to the ground along the LOS
  R_obj = R_g - h / sin(dep) = R_g (1 - h/A)
  omega_rel = s_rel / R_g                   measured ground-relative LOS rate
  v_ac_perp = v sin(dep)                    aircraft velocity across the LOS (flying toward the scene)
  v_obj = R_obj omega_rel + v_ac_perp (1 - R_obj/R_g)        (tool's formula, finite background)
        = (1 - h/A) s_rel  -  v_ac_perp (h/A)                (direction: stationary-object hypothesis, i.e.
                                                              the aircraft term anti-parallel to the motion)
A stationary object therefore reproduces s_rel at  h/A = s_rel / (s_rel + v_ac_perp).
"""
import csv, sys
import numpy as np

csv_path = sys.argv[1] if len(sys.argv) > 1 else "out/me/per_frame.csv"
GSD = 0.165          # m/px at the ground, reference frame, from the Bellingcat A-B datum  [EXTERNAL GEOLOCATION]
ALT = 25000 * 0.3048 # m, ASSUMED (MQ-9 operational altitude per Bellingcat)
V_KT = (150.0, 160.0, 170.0)   # ASSUMED cruise speed range

rows = list(csv.DictReader(open(csv_path)))
col = lambda n: next(c for c in rows[0] if c.split(" ")[0] == n)
s_px = np.array([float(r[col("stab_speed_refpx_s")] or "nan") for r in rows])
trusted = np.array([r["in_summaries"] == "1" for r in rows])
s_rel = GSD * s_px[trusted & np.isfinite(s_px)]
s_med, s_lo, s_hi = np.median(s_rel), *np.percentile(s_rel, [5, 95])
print(f"ground-relative LOS speed (MEASURED px/s x external scale): median {s_med:.1f} m/s, p5-p95 {s_lo:.1f}-{s_hi:.1f} m/s, "
      f"{int((trusted & np.isfinite(s_px)).sum())} trusted frames")
print(f"\nassumed: altitude {ALT:.0f} m, cruise {V_KT[0]:.0f}-{V_KT[-1]:.0f} kt, flying toward the scene; object height swept\n")
print(f"{'depression':>10} {'R_ground':>9} {'implied HFOV':>12} {'v_ac_perp':>9} | object speed across the LOS (m/s) for object height AGL:")
H = [0, 500, 1000, 2000, 3000, 4000, 5000]
print(f"{'':>10} {'':>9} {'':>12} {'':>9} | " + " ".join(f"{h:>6d} m" for h in H) + " |  h where stationary")
for dep in (30, 45, 60, 75):
    sd = np.sin(np.radians(dep)); R_g = ALT / sd
    hfov = np.degrees(558 * GSD / R_g)     # 558-px inset at the ground scale -> the FOV this depression implies
    for v_kt in (V_KT[0], V_KT[-1]):
        v = v_kt * 0.5144; vp = v * sd
        vals = [(1 - h / ALT) * s_med - vp * (h / ALT) for h in H]
        h0 = ALT * s_med / (s_med + vp)
        print(f"{dep:>9}° {R_g/1e3:>7.1f}km {hfov:>10.2f}°  {vp:>6.0f}m/s | " + " ".join(f"{x:>+8.0f}" for x in vals) + f" |  {h0:>5.0f} m  ({v_kt:.0f} kt)")
print("\nPositive = object moving in the direction seen on screen; negative = moving with the aircraft's lateral motion.")
print("Every column rests on the assumed altitude, speed, heading-toward-scene and the swept object height; the row's")
print("depression angle also fixes the implied FOV (a plane resection from >= 4 ground coordinates would measure it).")
