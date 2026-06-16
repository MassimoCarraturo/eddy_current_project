"""Demo: field-level out-of-plane strain recovery by mlhp FE equilibrium.

An in-plane ECT rosette measures eps_xx across a vertical (x, z) slice of a build.
The per-column free-surface closure eps_zz = -nu/(1-nu) eps_xx is exact at the
traction-free top but degrades with depth, where lateral gradients and the
clamped baseplate make sigma_zz non-zero. The mlhp hp-FEM solve enforces the
genuine balance law div(sigma) = 0 and so recovers eps_zz everywhere -- the role
of the thermomechanical Digital Twin in the workflow.

This contrasts the closure (fed the measured eps_xx) against the equilibrated
truth, as a depth profile and as a slice-wide error map.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.tensor_model.mlhp_equilibrium import EigenstrainEquilibrium, HAVE_MLHP

NU = 0.30


def main():
    if not HAVE_MLHP:
        print("This demo requires the optional 'mlhp' FE backend (pip install mlhp).")
        return

    print("=" * 64)
    print("Demo: out-of-plane strain by field-level equilibrium (mlhp)")
    print("=" * 64)

    eq = EigenstrainEquilibrium(E=1.0, nu=NU, eps0=-2.0e-3,
                                ncells=(12, 12), degree=4).solve()
    nx = nz = 41
    X, Z, EPS, SIG = eq.sample_grid(nx, nz)

    # "Measure" the in-plane strain with rosette-level noise, then close.
    rng = np.random.default_rng(3)
    eps_xx_meas = EPS[..., 0] + rng.normal(scale=2e-5, size=EPS[..., 0].shape)
    ezz_true = EPS[..., 1]
    ezz_clos = eq.closure(eps_xx_meas)

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    surf = (slice(None), -1)
    interior = (slice(None), slice(0, nz // 2))     # lower half (near baseplate)
    print(f"\n  eps_zz RMSE   closure vs truth   at surface = {rmse(ezz_clos[surf], ezz_true[surf]):.2e}")
    print(f"  eps_zz RMSE   closure vs truth   lower half  = {rmse(ezz_clos[interior], ezz_true[interior]):.2e}")
    print(f"  (mlhp equilibrium reproduces the truth by construction: RMSE = 0)")
    print(f"  peak |sigma_zz| interior / |sigma_zz|@surface = "
          f"{np.abs(SIG[..., 1]).max():.2e} / {np.abs(SIG[:, -1, 1]).max():.2e}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available -- skipping figure)")
        return

    zs = Z[0]
    imid = nx // 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(ezz_true[imid] * 100, zs, "k-", lw=2, label="true (mlhp equilibrium)")
    ax.plot(ezz_clos[imid] * 100, zs, "x", color="#d55e00", ms=5,
            label="surface closure (from meas. $\\varepsilon_{xx}$)")
    ax.set_ylabel("build height $z/H$")
    ax.set_xlabel(r"$\varepsilon_{zz}$ (%)")
    ax.set_title("Out-of-plane strain, column $x=W/2$")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    err = np.abs(ezz_clos - ezz_true).T * 100
    im = ax.pcolormesh(X[:, 0], Z[0], err, shading="auto", cmap="magma")
    ax.set_xlabel("lateral $x/W$")
    ax.set_ylabel("build height $z/H$")
    ax.set_title(r"$|\varepsilon_{zz}^{\rm closure}-\varepsilon_{zz}^{\rm true}|$ (%)")
    fig.colorbar(im, ax=ax)
    # mark the traction-free surface where the closure is valid
    ax.axhline(1.0, color="w", ls="--", lw=1)

    plt.tight_layout()
    out = "mlhp_equilibrium_demo.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
