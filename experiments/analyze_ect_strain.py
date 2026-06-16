"""EC <-> strain correlation for the Pavia AMIQUAM tensile+ECT experiment.

The lab bonded an eddy-current probe to AM 316L tensile specimens and recorded
the in-/out-of-phase response while loading. This script turns that raw response
into a lift-off-compensated conductivity change and compares it to the scalar
elastoresistive law used in the manuscript.

Pipeline
--------
1. Calibration -> lift-off-invariant proxy. The freq--lift-off calibration grid
   (calibration_datapoints.xlsx) gives (Re, Im) at many (frequency, lift-off)
   points. An RBF inverts it: (Re, Im) -> effective frequency f_eff. Because the
   eddy-current response depends on the product omega*sigma, at the FIXED test
   frequency f0 = 240.4 kHz the effective frequency tracks conductivity:
       f_eff / f0 = sigma / sigma0   =>   Dsigma/sigma0 = Df_eff / f_eff .
   Reading f_eff off the calibration manifold therefore compensates the
   lift-off / contact drift that otherwise dominates the raw (Re, Im).
2. Apply to each tensile-test EC record -> Dsigma/sigma0(t). Samples whose
   (Re, Im) leaves the calibration convex hull are masked (the RBF must not
   extrapolate).
3. Compare to Dsigma/sigma0 = kappa * eps, kappa = -0.326 (316L).

Caveats (see the data, not just this script):
- Only the APRIL campaign EC files share the calibration's acquisition scale
  (Re ~ 1.7, Im ~ 4-6); the November runs are ~100x smaller and need their own
  calibration.
- The axial extensometer was not engaged, so strain is not directly measured;
  this validates the conductivity side (sign + magnitude), not a precise kappa.
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay

ROOT = r"C:\Users\user\OneDrive - Università di Pavia\PROJECTS\AMIQUAM\Experiment"
F0 = 240.4          # test drive frequency [kHz]
KAPPA = -0.326      # scalar elastoresistive coefficient for 316L (manuscript)

# April EC records that share the calibration scale and stay in-hull.
CLEAN_APRIL = ["ex_1", "ex_3", "ex_4", "ex_6", "ex_9", "ex_10"]


def build_calibration():
    """RBF mapping measured (Re, Im) -> effective frequency [kHz]."""
    path = os.path.join(ROOT, "calibration", "calibration_datapoints.xlsx")
    g = pd.read_excel(path).dropna(subset=["real", "imag"])
    pts = g[["real", "imag"]].values
    rbf = RBFInterpolator(pts, g["frequency"].values,
                          kernel="thin_plate_spline", smoothing=1e-3)
    return rbf, Delaunay(pts)


def load_ec(fn):
    d = pd.read_csv(fn)
    d.columns = [c.strip() for c in d.columns]
    return (d["Time (s)"].values,
            d["Signal (in-phase)"].values,
            d["Signal (out-of-phase)"].values)


def conductivity_trajectory(fn, rbf, hull, win=150):
    """Return (t, Dsigma/sigma0 [%], in-hull fraction) for one EC record.

    ``win`` samples ~ 0.4 s at the ~380 Hz acquisition (matches the lab's
    averaging window).
    """
    t, re, im = load_ec(fn)
    box = np.ones(win) / win
    sm = lambda x: np.convolve(x, box, mode="valid")
    rs, ims, ts = sm(re), sm(im), t[win - 1:]
    xy = np.c_[rs, ims]
    inside = hull.find_simplex(xy) >= 0
    f = np.where(inside, rbf(xy), np.nan)
    f0 = np.nanmedian(f[:300])                 # unloaded baseline
    dsigma = (f - f0) / f0 * 100.0             # Dsigma/sigma0 in percent
    return ts, dsigma, float(inside.mean())


def summarize():
    rbf, hull = build_calibration()
    rows = []
    traj = {}
    for name in CLEAN_APRIL:
        fn = os.path.join(ROOT, "tensile_test", "EC_data", name + ".csv")
        ts, ds, frac = conductivity_trajectory(fn, rbf, hull)
        n = len(ds)
        start = np.nanmedian(ds[:n // 8])
        end = np.nanmedian(ds[-n // 8:])
        net = end - start
        rows.append((name, ts[-1] - ts[0], 100 * frac, net))
        traj[name] = (ts, ds)
    return rows, traj


if __name__ == "__main__":
    rows, traj = summarize()
    print(f"{'record':8s} {'dur[s]':>7s} {'in-hull%':>9s} {'Dsigma/sigma0[%]':>17s}")
    nets = []
    for name, dur, frac, net in rows:
        nets.append(net)
        print(f"{name:8s} {dur:7.0f} {frac:9.0f} {net:17.3f}")
    med = float(np.median(nets))
    print(f"\nmedian net Dsigma/sigma0 = {med:+.3f}%  (presentation reported 0.57%)")
    print(f"implied strain at kappa={KAPPA}:  eps = {med/100/KAPPA*100:+.2f}%")
