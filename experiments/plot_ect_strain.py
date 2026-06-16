"""Figure: lift-off-compensated EC conductivity change under tensile loading.

Plots Dsigma/sigma0(t) for the clean April specimens (all decreasing -> kappa<0)
and overlays the magnitude implied by the manuscript's kappa = -0.326.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from analyze_ect_strain import (build_calibration, conductivity_trajectory,
                                CLEAN_APRIL, ROOT, KAPPA)

OKABE = ["#0072b2", "#e69f00", "#009e73", "#d55e00", "#56b4e9", "#cc79a7"]


def main():
    rbf, hull = build_calibration()
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    nets = []
    for k, name in enumerate(CLEAN_APRIL):
        fn = os.path.join(ROOT, "tensile_test", "EC_data", name + ".csv")
        ts, ds, _ = conductivity_trajectory(fn, rbf, hull)
        tn = (ts - ts[0]) / (ts[-1] - ts[0])          # normalised time
        ax.plot(tn, ds, color=OKABE[k % len(OKABE)], lw=1.3, label=name)
        n = len(ds)
        nets.append(np.nanmedian(ds[-n // 8:]) - np.nanmedian(ds[:n // 8]))
    med = float(np.median(nets))
    ax.axhline(med, color="0.4", ls="--", lw=1,
               label=f"median end = {med:.2f}%")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("normalised test time  $t/t_{\\mathrm{end}}$")
    ax.set_ylabel(r"$\Delta\sigma/\sigma_0$  (\%)".replace("\\%", "%"))
    ax.set_title("EC conductivity change under tensile load (April campaign, "
                 "lift-off compensated)")
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    eps = med / 100 / KAPPA * 100
    ax.text(0.02, 0.06,
            f"$\\sigma$ decreases under tension ($\\kappa<0$).\n"
            f"median $\\Delta\\sigma/\\sigma_0={med:.2f}\\%$  "
            f"$\\Rightarrow\\ \\varepsilon\\approx{eps:.1f}\\%$ at $\\kappa=-0.326$",
            transform=ax.transAxes, fontsize=8.5, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "ect_conductivity_vs_load.png")
    plt.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
