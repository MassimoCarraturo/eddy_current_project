"""
Demo: sensitivity of strain observability to the elastoresistivity ratio.

The strain map is A = sigma0 * M * K, so the resolvable degrees of freedom and
the conditioning are bounded by rank(K). For the isotropic two-constant model K
becomes rank-deficient as kappa_perp -> kappa_parallel (a purely volumetric
response): no probe configuration can then resolve more than the volumetric
strain. This script sweeps the ratio kappa_perp/kappa_parallel and reports the
condition number and resolvable DOF for an in-plane rosette and a full
(in-plane + tilted) probe set.

It also documents the constants used throughout the paper and their provenance:
kappa_parallel is anchored so that the uniaxial along-load projection reproduces
the thesis scalar kappa = -0.326 (316L), with kappa_perp a second, illustrative
constant (one uniaxial test under-determines the pair).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.tensor_model import (ElastoResistivityModel, direction,
                              evaluate_configuration)

POISSON = 0.30
KAPPA_SCALAR = -0.326          # thesis-calibrated scalar (316L)
KAPPA_PERP_NOMINAL = -0.08     # illustrative transverse constant


def kappa_parallel_from_scalar(kappa_scalar, kappa_perp, nu):
    """kappa_par such that the uniaxial along-load projection equals kappa_scalar.

    Uniaxial-stress along-load projection:  kappa_par - 2*nu*kappa_perp.
    """
    return kappa_scalar + 2.0 * nu * kappa_perp


def main():
    print("=" * 60)
    print("Demo: Observability vs elastoresistivity ratio")
    print("=" * 60)

    kpar = kappa_parallel_from_scalar(KAPPA_SCALAR, KAPPA_PERP_NOMINAL, POISSON)
    print(f"\nProvenance of constants:")
    print(f"  thesis scalar kappa (316L)      = {KAPPA_SCALAR}")
    print(f"  chosen kappa_perp (illustrative)= {KAPPA_PERP_NOMINAL}")
    print(f"  => kappa_parallel (anchored)    = {kpar:.3f}")
    m = ElastoResistivityModel(kappa_long=kpar, kappa_trans=KAPPA_PERP_NOMINAL)
    print(f"  along-load projection           = "
          f"{m.effective_scalar_kappa([1,0,0],[1,0,0],POISSON):.3f}  (= thesis)")
    print(f"  transverse projection           = "
          f"{m.effective_scalar_kappa([1,0,0],[0,1,0],POISSON):.3f}")
    print(f"  nominal ratio kappa_perp/kappa_par = "
          f"{KAPPA_PERP_NOMINAL/kpar:.3f}")

    rosette = [direction(a) for a in (0, 60, 120)]
    full = ([direction(a) for a in (0, 60, 120)]
            + [direction(a, 45) for a in (0, 60, 120)])

    ratios = np.linspace(0.0, 0.98, 60)
    cond_ros, cond_full, rank_ros, rank_full = [], [], [], []
    for r in ratios:
        model = ElastoResistivityModel(kappa_long=kpar, kappa_trans=r * kpar)
        mr = evaluate_configuration(rosette, model)
        mf = evaluate_configuration(full, model)
        cond_ros.append(mr["condition_number"])
        cond_full.append(mf["condition_number"])
        rank_ros.append(mr["rank"])
        rank_full.append(mf["rank"])

    nominal_ratio = KAPPA_PERP_NOMINAL / kpar

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Strain observability vs elastoresistivity ratio "
                 r"$\kappa_\perp/\kappa_\parallel$", fontsize=13)

    ax = axes[0]
    ax.semilogy(ratios, cond_full, "b-", lw=2, label="full set (6 probes)")
    ax.semilogy(ratios, cond_ros, "r-", lw=2, label="in-plane rosette (3)")
    ax.axvline(nominal_ratio, color="k", ls="--", lw=1,
               label=f"nominal ({nominal_ratio:.2f})")
    ax.set_xlabel(r"$\kappa_\perp/\kappa_\parallel$")
    ax.set_ylabel("condition number of strain map")
    ax.set_title("Conditioning collapses toward the volumetric limit")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.plot(ratios, rank_full, "b-", lw=2, label="full set (6 probes)")
    ax.plot(ratios, rank_ros, "r--", lw=2, label="in-plane rosette (3)")
    ax.axvline(nominal_ratio, color="k", ls="--", lw=1)
    ax.set_xlabel(r"$\kappa_\perp/\kappa_\parallel$")
    ax.set_ylabel("resolvable strain DOF")
    ax.set_ylim(0, 6.5)
    ax.set_title("Resolvable DOF (1 at the volumetric limit)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("kappa_sensitivity.png", dpi=150)
    plt.show()
    print("\nSaved: kappa_sensitivity.png")


if __name__ == "__main__":
    main()
