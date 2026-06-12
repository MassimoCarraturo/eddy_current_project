"""
Generate all numeric data for the manuscript figures as CSV/DAT files in
paper/data/. Each pgfplots/TikZ figure reads from here, so the figures are
fully reproducible and version-controlled as data + source (no raster blobs).

Run:  python paper/gen_paper_data.py
"""

import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
DATA = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA, exist_ok=True)

from src.utils.common import ECTCoilParams, MaterialParams
from src.inversion.dodd_deeds import DoddDeedsModel
from src.tensor_model import (ElastoResistivityModel, ObservabilityAnalysis,
                              direction, evaluate_configuration,
                              optimize_inplane_rosette, optimize_probe_set,
                              TensorStrainInverter, to_voigt)

COMP = ["xx", "yy", "zz", "yz", "xz", "xy"]


def save_csv(name, header, rows):
    path = os.path.join(DATA, name)
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(f"{v:.6g}" for v in r) + "\n")
    print("wrote", name, f"({len(rows)} rows)")


def save_grid(name, X, Y, Z):
    """Whitespace .dat with blank line between scanlines for pgfplots surf."""
    path = os.path.join(DATA, name)
    with open(path, "w") as f:
        f.write("x y z\n")
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                f.write(f"{X[i, j]:.6g} {Y[i, j]:.6g} {Z[i, j]:.6g}\n")
            f.write("\n")
    print("wrote", name, f"({X.size} pts)")


# ---------------------------------------------------------------------------
# Coil / material (reference coil used for the forward-model study)
# ---------------------------------------------------------------------------
coil = ECTCoilParams(r_inner=0.00404, r_outer=0.01184, length=0.00802,
                     n_turns=1858, liftoff=0.001, frequency=240e3)
material = MaterialParams(sigma=16.2e6)
model_dd = DoddDeedsModel(coil)

# ---- 1. analytical normalised-impedance locus vs conductivity -------------
sig_sweep = np.logspace(4, 8, 45)
z_sweep = model_dd.impedance_vs_sigma(sig_sweep)
save_csv("imp_locus.csv", ["sigma", "re", "im"],
         list(zip(sig_sweep, z_sweep.real, z_sweep.imag)))

# ---- 4a. analytical k_max convergence -------------------------------------
kmax_rows = []
for kmax in (1000, 2000, 3000, 4000, 6000, 8000):
    zn = DoddDeedsModel(coil, k_max=float(kmax)).z_normalized(material)
    kmax_rows.append((kmax, zn.real, zn.imag))
save_csv("kmax_conv.csv", ["kmax", "Rnorm", "Xnorm"], kmax_rows)

# ---- FEM-dependent data (needs NGSolve) -----------------------------------
fem_point = None
try:
    from src.fem_impedance.coil_impedance import FEMCoilImpedance
    from src.fem_impedance.sensitivity import validate_reciprocity, sensitivity_grid

    fem = FEMCoilImpedance(coil=coil, material=material)
    res = fem.setup_and_solve()
    zf = res["impedance_normalized"]
    fem_point = (zf.real, zf.imag)
    save_csv("fem_point.csv", ["re", "im"], [fem_point])

    # 4b. reciprocity ratio vs perturbation
    rec_rows = []
    for frac in (0.02, 0.01, 0.005, 0.002):
        r = validate_reciprocity(coil, material, dsigma_frac=frac)
        rec_rows.append((frac, abs(r["ratio"])))
    save_csv("recip.csv", ["frac", "ratio_abs"], rec_rows)

    # 5a. sensitivity kernel map (near surface, in the specimen)
    rho, z, K = sensitivity_grid(coil, material, n_rho=60, n_z=60,
                                 rho_max=0.025, z_min=-0.0015, z_max=-2e-5)
    RHO, ZZ = np.meshgrid(rho * 1e3, z * 1e3)
    Kfilled = np.nan_to_num(K, nan=0.0)
    save_grid("kernel.dat", RHO, ZZ, Kfilled)

    # 5b. mean sensitivity vs depth at two frequencies
    depth_cols = {}
    zmm_ref = None
    for f in (120e3, 480e3):
        coil_f = ECTCoilParams(r_inner=coil.r_inner, r_outer=coil.r_outer,
                               length=coil.length, n_turns=coil.n_turns,
                               liftoff=coil.liftoff, frequency=f)
        rho2, z2, K2 = sensitivity_grid(coil_f, material, n_rho=50, n_z=120,
                                        rho_max=0.02, z_min=-0.003, z_max=-2e-5)
        prof = np.nanmean(K2, axis=1)
        prof = prof / np.nanmax(prof)
        depth_cols[f] = prof
        zmm_ref = z2 * 1e3
    save_csv("depth_freq.csv", ["depth_mm", "s120k", "s480k"],
             list(zip(zmm_ref, depth_cols[120e3], depth_cols[480e3])))
except Exception as e:  # pragma: no cover
    print("[WARN] FEM data skipped:", e)

# ---------------------------------------------------------------------------
# Elastoresistivity / observability (illustrative kappa = -0.40, -0.08)
# ---------------------------------------------------------------------------
model = ElastoResistivityModel(kappa_long=-0.40, kappa_trans=-0.08)
POISSON = 0.30

# ---- 6. conductivity rosette: kappa_eff vs sensing angle ------------------
load = np.array([1.0, 0.0, 0.0])
eps_uni = (1 + POISSON) * np.outer(load, load) - POISSON * np.eye(3)
dsig_uni = model.delta_sigma_ratio(eps_uni)
ang = np.linspace(0, 180, 181)
keff = [(a, float(np.array([np.cos(np.radians(a)), np.sin(np.radians(a)), 0])
                   @ dsig_uni
                   @ np.array([np.cos(np.radians(a)), np.sin(np.radians(a)), 0])))
        for a in ang]
save_csv("rosette.csv", ["angle_deg", "kappa_eff"], keff)

# ---- 7. observability SVD spectra (strain map) ----------------------------
configs = {
    "normal": ObservabilityAnalysis([np.array([0.5, 0.5, 0, 0, 0, 0])]),
    "rosette": ObservabilityAnalysis.from_directions(
        [direction(a) for a in (0, 60, 120)]),
    "full": ObservabilityAnalysis.from_directions(
        [direction(0), direction(60), direction(120),
         direction(0, 45), direction(60, 45), direction(120, 45)]),
}
spec = {}
for nm, oa in configs.items():
    s = oa.analyse(oa.strain_map(model))["singular_values"]
    spec[nm] = np.pad(s, (0, 6 - len(s)))
save_csv("spectra.csv", ["index", "normal", "rosette", "full"],
         [(i + 1, spec["normal"][i], spec["rosette"][i], spec["full"][i])
          for i in range(6)])

# ---- 8. kappa_perp/kappa_par sensitivity ----------------------------------
kpar = -0.326 + 2 * POISSON * (-0.08)   # anchored so along-load proj = -0.326
ros = [direction(a) for a in (0, 60, 120)]
full = ros + [direction(a, 45) for a in (0, 60, 120)]
krows = []
for r in np.linspace(0.0, 0.98, 50):
    m = ElastoResistivityModel(kappa_long=kpar, kappa_trans=r * kpar)
    mr = evaluate_configuration(ros, m)
    mf = evaluate_configuration(full, m)
    krows.append((r, mr["condition_number"], mf["condition_number"],
                  mr["rank"], mf["rank"]))
save_csv("kappa.csv", ["ratio", "cond_ros", "cond_full", "rank_ros", "rank_full"],
         krows)

# ---- probe-design conditioning numbers (for the table/text) ---------------
cond_naive = evaluate_configuration([direction(a) for a in (0, 45, 90)], model)["condition_number"]
cond_delta = evaluate_configuration([direction(a) for a in (0, 60, 120)], model)["condition_number"]
opt_in = optimize_inplane_rosette(model, n_probes=3, n_restarts=12)
opt_full = optimize_probe_set(model, n_probes=6, n_restarts=12)
save_csv("design_cond.csv", ["config", "cond"],
         [(0, cond_naive), (1, cond_delta), (2, opt_in["condition_number"]),
          (3, opt_full["condition_number"])])
print("  (cond: 0/45/90=%.2f, 0/60/120=%.2f, opt-in=%.2f, full=%.2f)" %
      (cond_naive, cond_delta, opt_in["condition_number"], opt_full["condition_number"]))

# ---------------------------------------------------------------------------
# 9. Tensor inversion along a build-height profile
# ---------------------------------------------------------------------------
def build_profile(n=40):
    z = np.linspace(0, 1, n)
    eps = np.zeros((n, 3, 3))
    eps[:, 0, 0] = -0.004 + 0.002 * z
    eps[:, 1, 1] = -0.003 + 0.0015 * z
    eps[:, 2, 2] = 0.001 + 0.006 * z
    eps[:, 0, 1] = eps[:, 1, 0] = 0.002 * np.sin(np.pi * z)
    eps[:, 0, 2] = eps[:, 2, 0] = 0.0015 * z
    eps[:, 1, 2] = eps[:, 2, 1] = -0.001 * z
    return z, eps


rng = np.random.default_rng(1)
zb, eps_true = build_profile(40)
ev = np.array([to_voigt(e) for e in eps_true])
prior = 0.85 * ev + np.array([0.001, 0.001, 0.0015, 0, 0.0008, 0])
dirs_full = opt_full["directions"]
dirs_ros = [direction(a) for a in (0, 60, 120)]
inv_full = TensorStrainInverter(model, dirs_full, prior_weight=1e-5)
inv_ros = TensorStrainInverter(model, dirs_ros, prior_weight=2e-3)
Yf = np.array([inv_full.forward(e, noise_std=1e-4, rng=rng) for e in eps_true])
Yr = np.array([inv_ros.forward(e, noise_std=1e-4, rng=rng) for e in eps_true])
rec_full = inv_full.invert_field(Yf, prior)
rec_ros = inv_ros.invert_field(Yr, prior)

cols, hdr = [], ["z"]
data_cols = [zb]
for nm, idx in [("xx", 0), ("zz", 2), ("xy", 5), ("xz", 4)]:
    for tag, arr in [("true", ev[:, idx]), ("prior", prior[:, idx]),
                     ("full", rec_full[:, idx]), ("ros", rec_ros[:, idx])]:
        hdr.append(f"{tag}_{nm}")
        data_cols.append(arr * 100)   # percent
save_csv("inv_profiles.csv", hdr, list(zip(*data_cols)))

save_csv("resolution.csv", ["index", "full", "rosette"],
         [(i + 1, inv_full.data_resolved_fraction()[i],
           inv_ros.data_resolved_fraction()[i]) for i in range(6)])


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# case: 0 = prior only, 1 = full probe set, 2 = in-plane rosette
save_csv("rmse.csv", ["case", "rmse"],
         [(0, rmse(prior, ev)), (1, rmse(rec_full, ev)), (2, rmse(rec_ros, ev))])
print("  RMSE  prior=%.3e  full=%.3e  rosette=%.3e"
      % (rmse(prior, ev), rmse(rec_full, ev), rmse(rec_ros, ev)))

# ---------------------------------------------------------------------------
# 10. Pipeline reconstructed strain field (von Mises + volumetric)
# ---------------------------------------------------------------------------
from src.pipeline import UnifiedECTPipeline

nL, nX = 50, 36
zg = np.linspace(0, 1, nL)
xg = np.linspace(0, 1, nX)
fwd = TensorStrainInverter(model, dirs_full, prior_weight=1e-6)
pipe = UnifiedECTPipeline(coil, material)
true_field = np.zeros((nL, nX, 3, 3))
for i, zz in enumerate(zg):
    meas = []
    for j, xx in enumerate(xg):
        e = np.zeros((3, 3))
        e[0, 0] = -0.004 + 0.003 * zz
        e[1, 1] = -0.003 + 0.002 * zz
        e[2, 2] = 0.001 + 0.006 * zz
        e[0, 1] = e[1, 0] = 0.0025 * np.sin(np.pi * xx) * zz
        e[0, 2] = e[2, 0] = 0.0015 * zz
        e[1, 2] = e[2, 1] = -0.001 * zz
        true_field[i, j] = e
        meas.append(fwd.forward(e, noise_std=1e-4, rng=rng))
    pipe.add_directional_layer(i, np.column_stack(meas))
pres = pipe.run_tensor_strain_prediction(model, dirs_full, prior_weight=1e-5)
eqv = pres.equivalent_strain() * 100      # (nL, nX) percent
vol = pres.volumetric_strain() * 100
Xg, Zg = np.meshgrid(xg, zg)
save_grid("field_eqv.dat", Xg, Zg, eqv)
save_grid("field_vol.dat", Xg, Zg, vol)

print("\nAll paper data written to", DATA)
