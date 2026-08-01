"""
dataset_interpolation.py — Run-once builder for a temporally up-sampled dataset.

Goal: densify the time axis of the original simulation by inserting
N_INTERPOLATED extra points between every pair of adjacent original samples
using one of three interpolation methods. The output is a new dataset folder
with the same structure as Dataset/, holding .npy files instead of .mat/.csv.

This is a smoothing/denser-discretization step on the existing samples — it
does not claim to recover the analogue signal between samples; it simply
reduces the spike-encoding discretization artifacts caused by the coarse
temporal grid.

Output layout (mirrors Dataset/):
    Dataset_interpolated_<method>_N<N>/
    ├── vorticity/
    │   ├── vort_a20/Vort_AoA20_base.npy   (Nt_new, Ny, Nx) float32
    │   └── ...
    └── lift/
        ├── lift_a20/Lift_AoA20_base.npy   (Nt_new,) float32
        └── ...

Usage:
    python main/dataset_interpolation.py
    python main/dataset_interpolation.py --force
    python main/dataset_interpolation.py --angles 20 30 --n 4 --method pchip

Memory: peak RAM stays ~2 GB (one case at a time, column-chunked spline).
Disk:   each interpolated case is ~Nt_new × Ny × Nx × 4 bytes. For N=9 with
        Nt_orig=1190, Ny=120, Nx=240 that is ~1.37 GB per case → ~213 GB
        total for 155 cases.
"""

import argparse
import gc
import glob
import os
import sys
import time

import numpy as np
import scipy.interpolate

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from pre_encoder import load_vorticity, load_lift


# ==========================================
# CONFIG
# ==========================================
INTERPOLATION_TYPE = "cubic_spline"   # "linear" | "pchip" | "cubic_spline"
N_INTERPOLATED     = 4                # extra points between adjacent original samples
ANGLES             = [20, 30, 40, 50, 60]


def _find_data_root():
    env = os.environ.get('THESIS_DATA_ROOT')
    if env:
        return os.path.abspath(env)
    base = os.path.join(_HERE, '..')
    # Datasets live under the repo root, e.g. <repo>/datasets/Dataset_original/
    for name in ('Datasets', 'DATASETS', 'datasets'):
        candidate = os.path.normpath(os.path.join(base, name, 'Dataset_original'))
        if os.path.isdir(candidate):
            return candidate
    # Legacy fallback: datasets live directly under the repo root (<repo>/Dataset/)
    for name in ('Dataset', 'DATASET', 'dataset'):
        candidate = os.path.normpath(os.path.join(base, name))
        if os.path.isdir(candidate):
            return candidate
    return os.path.normpath(os.path.join(base, 'datasets', 'Dataset_original'))


# ==========================================
# INTERPOLATOR DISPATCH
# ==========================================
def _build_interpolator(t_orig, y, axis, method):
    """Build an interpolator object for the requested method.

    All three methods are *interpolating* (pass through every original sample)
    rather than approximating, so f(t_orig) == y exactly (within float
    rounding).
    """
    if method == "linear":
        return scipy.interpolate.interp1d(
            t_orig, y, axis=axis, kind="linear", assume_sorted=True,
            copy=False,
        )
    if method == "pchip":
        return scipy.interpolate.PchipInterpolator(t_orig, y, axis=axis)
    if method == "cubic_spline":
        return scipy.interpolate.CubicSpline(
            t_orig, y, axis=axis, bc_type='not-a-knot',
        )
    raise ValueError(
        f"Unknown INTERPOLATION_TYPE: {method!r}. "
        f"Use 'linear', 'pchip', or 'cubic_spline'."
    )


# ==========================================
# CORE INTERPOLATION
# ==========================================
def _interpolate_vorticity_chunked(omega, t_new, method, col_chunk=4000):
    """Interpolate (Nt, Ny, Nx) vorticity along axis 0 onto t_new.

    Reshapes to (Nt, Ny*Nx) and evaluates the interpolator in column-chunks
    so the transient float64 buffer never reaches the full spatial extent.

    Args:
        omega:     (Nt, Ny, Nx) float32 vorticity
        t_new:     (Nt_new,) target time grid
        method:    "linear" | "pchip" | "cubic_spline"
        col_chunk: number of spatial columns per chunk

    Returns:
        out: (Nt_new, Ny, Nx) float32
    """
    Nt, Ny, Nx = omega.shape
    Nt_new = len(t_new)
    n_cols = Ny * Nx
    t_orig = np.arange(Nt, dtype=np.float64)

    omega_2d = omega.reshape(Nt, n_cols)
    out_2d = np.empty((Nt_new, n_cols), dtype=np.float32)

    for c0 in range(0, n_cols, col_chunk):
        c1 = min(c0 + col_chunk, n_cols)
        interp = _build_interpolator(
            t_orig, omega_2d[:, c0:c1], axis=0, method=method,
        )
        chunk = interp(t_new)
        out_2d[:, c0:c1] = chunk.astype(np.float32, copy=False)
        del interp, chunk

    return out_2d.reshape(Nt_new, Ny, Nx)


def _interpolate_cl(cl, t_new, method):
    """Interpolate (Nt,) Cl onto t_new. Returns (Nt_new,) float32."""
    Nt = cl.shape[0]
    t_orig = np.arange(Nt, dtype=np.float64)
    interp = _build_interpolator(t_orig, cl, axis=0, method=method)
    return interp(t_new).astype(np.float32, copy=False)


# ==========================================
# PER-CASE PROCESSOR
# ==========================================
def _process_case(angle, case_id, vort_in, cl_in, vort_out, cl_out,
                  n_interp, method, force):
    """Process one (vorticity, lift) case end-to-end.

    Returns (status, vort_bytes, cl_bytes) where status is "wrote" or "skip".
    """
    if (not force) and os.path.exists(vort_out) and os.path.exists(cl_out):
        return "skip", 0, 0

    omega = load_vorticity(vort_in)         # (Nt, Ny, Nx) float32
    cl    = load_lift(cl_in)                # (Nt,) float32

    if omega.shape[0] != cl.shape[0]:
        print(f"[WARN] AoA{angle}_{case_id}: time mismatch "
              f"(omega={omega.shape[0]}, cl={cl.shape[0]}). Skipping.")
        return "skip", 0, 0

    Nt = omega.shape[0]
    Nt_new = (Nt - 1) * (n_interp + 1) + 1
    t_new = np.linspace(0.0, Nt - 1, Nt_new, dtype=np.float64)

    vort_new = _interpolate_vorticity_chunked(omega, t_new, method)
    cl_new   = _interpolate_cl(cl, t_new, method)

    os.makedirs(os.path.dirname(vort_out), exist_ok=True)
    os.makedirs(os.path.dirname(cl_out),   exist_ok=True)
    np.save(vort_out, vort_new)
    np.save(cl_out, cl_new)

    vort_bytes = os.path.getsize(vort_out)
    cl_bytes   = os.path.getsize(cl_out)

    del omega, cl, vort_new, cl_new, t_new
    gc.collect()
    return "wrote", vort_bytes, cl_bytes


# ==========================================
# MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Build a temporally interpolated dataset from the original "
                    "vorticity .mat and lift .csv files.",
    )
    parser.add_argument(
        "--method", default=INTERPOLATION_TYPE,
        choices=["linear", "pchip", "cubic_spline"],
        help="Interpolation method (default: %(default)s).",
    )
    parser.add_argument(
        "--n", type=int, default=N_INTERPOLATED,
        help="Number of interpolated points between adjacent original samples "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--angles", type=int, nargs="+", default=ANGLES,
        help="Angles of attack to process (default: %(default)s).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output files instead of skipping.",
    )
    parser.add_argument(
        "--data-root", default=None,
        help="Override DATA_ROOT (default: auto-detected via THESIS_DATA_ROOT "
             "or repo-relative search).",
    )
    args = parser.parse_args()

    method = args.method
    n_interp = args.n
    angles = args.angles
    data_root = args.data_root or _find_data_root()
    # Output sits as a sibling of Dataset_original under Datasets/
    out_root = os.path.join(
        os.path.dirname(data_root), f"Dataset_interpolated_{method}_N{n_interp}",
    )

    vort_in_root = os.path.join(data_root, "vorticity")
    lift_in_root = os.path.join(data_root, "lift")
    if not os.path.isdir(vort_in_root) or not os.path.isdir(lift_in_root):
        raise FileNotFoundError(
            f"Source dataset not found under {data_root}. "
            f"Expected vorticity/ and lift/ subfolders."
        )

    print("=" * 60)
    print("  DATASET INTERPOLATION BUILDER")
    print("=" * 60)
    print(f"  Method       : {method}")
    print(f"  N_interp     : {n_interp} (density factor = {n_interp + 1}x)")
    print(f"  Angles       : {angles}")
    print(f"  Source root  : {data_root}")
    print(f"  Output root  : {out_root}")
    print(f"  Force        : {args.force}")
    print("=" * 60)

    n_total_cases = 0
    n_written = 0
    n_skipped = 0
    total_vort_bytes = 0
    total_cl_bytes   = 0
    t_global = time.time()

    for angle in angles:
        vort_folder = os.path.join(vort_in_root, f"vort_a{angle}")
        lift_folder = os.path.join(lift_in_root, f"lift_a{angle}")
        if not os.path.isdir(vort_folder):
            print(f"[WARN] Vorticity folder not found: {vort_folder}")
            continue

        mat_files = sorted(glob.glob(
            os.path.join(vort_folder, f"Vort_AoA{angle}_*.mat"),
        ))
        print(f"\n[ANGLE] AoA={angle}°  ({len(mat_files)} case files)")

        for mat_path in mat_files:
            fname = os.path.basename(mat_path)
            case_id = fname.replace(f"Vort_AoA{angle}_", "").replace(".mat", "")
            csv_path = os.path.join(
                lift_folder, f"Lift_AoA{angle}_{case_id}.csv",
            )
            if not os.path.exists(csv_path):
                print(f"  [WARN] Lift file missing for {fname}, skipping.")
                continue

            vort_out = os.path.join(
                out_root, "vorticity", f"vort_a{angle}",
                f"Vort_AoA{angle}_{case_id}.npy",
            )
            cl_out = os.path.join(
                out_root, "lift", f"lift_a{angle}",
                f"Lift_AoA{angle}_{case_id}.npy",
            )

            n_total_cases += 1
            t_case = time.time()
            status, vb, cb = _process_case(
                angle, case_id, mat_path, csv_path, vort_out, cl_out,
                n_interp, method, args.force,
            )
            dt = time.time() - t_case

            if status == "skip":
                n_skipped += 1
                print(f"  [SKIP] AoA{angle}_{case_id}  (output already exists)")
            else:
                n_written += 1
                total_vort_bytes += vb
                total_cl_bytes += cb
                vort_mb = vb / (1024 * 1024)
                print(f"  [OK]   AoA{angle}_{case_id}  "
                      f"vort={vort_mb:.0f} MB  ({dt:.1f}s)")

    total_dt = time.time() - t_global
    total_gb = (total_vort_bytes + total_cl_bytes) / (1024 ** 3)
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Cases scanned   : {n_total_cases}")
    print(f"  Cases written   : {n_written}")
    print(f"  Cases skipped   : {n_skipped}")
    print(f"  Total disk used : {total_gb:.2f} GB (this run only)")
    print(f"  Output root     : {out_root}")
    print(f"  Elapsed         : {total_dt/60:.1f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
