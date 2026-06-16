"""Export the experimental EC conductivity trajectories to the paper data folder.

Writes paper/data/exp_conductivity.csv: normalised test time, the six clean
April specimens' Dsigma/sigma0(t) [%], and their median. Source of the figure
fig_experiment.tex. Reproducible from the raw lab data via analyze_ect_strain.
"""

import os
import numpy as np

from analyze_ect_strain import (build_calibration, conductivity_trajectory,
                                CLEAN_APRIL, ROOT)

HERE = os.path.dirname(__file__)
OUT = os.path.normpath(os.path.join(HERE, "..", "paper", "data", "exp_conductivity.csv"))


def main():
    rbf, hull = build_calibration()
    grid = np.linspace(0.0, 1.0, 60)
    cols = {}
    for name in CLEAN_APRIL:
        fn = os.path.join(ROOT, "tensile_test", "EC_data", name + ".csv")
        ts, ds, _ = conductivity_trajectory(fn, rbf, hull)
        tn = (ts - ts[0]) / (ts[-1] - ts[0])
        good = np.isfinite(ds)
        cols[name] = np.interp(grid, tn[good], ds[good])
    stack = np.vstack([cols[n] for n in CLEAN_APRIL])
    median = np.median(stack, axis=0)

    header = "tnorm," + ",".join(f"s{i+1}" for i in range(len(CLEAN_APRIL))) + ",median"
    rows = np.column_stack([grid, stack.T, median])
    with open(OUT, "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(",".join(f"{v:.5g}" for v in r) + "\n")
    print("wrote", OUT, rows.shape)
    print("median end Dsigma/sigma0 = %.2f%%" % median[-1])


if __name__ == "__main__":
    main()
