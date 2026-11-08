"""Compare two ov_pipeline.py outputs:  python compare.py REF.npz NEW.npz"""
import sys

import numpy as np


def pca_agreement(a, b):
    """Per-component |cosine| between two score matrices (sign-invariant)."""
    num = np.abs((a * b).sum(0))
    den = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    return num / np.where(den == 0, 1, den)


def main(ref_path, new_path):
    R, N = np.load(ref_path), np.load(new_path)
    ok = True
    for k in ("obs_names", "var_names"):
        same = R[k].shape == N[k].shape and np.array_equal(R[k], N[k])
        print(f"{k:38s} {'identical' if same else 'DIFFERENT'} ({len(R[k])})")
        ok &= same
    skip = {"obs_names", "var_names", "X_pca", "PCs", "variance_ratio"}
    for k in sorted(set(R.files) | set(N.files)):
        if k in skip:
            continue
        if k not in R.files or k not in N.files:
            print(f"{k:38s} only in {'ref' if k in R.files else 'new'}")
            ok = False
            continue
        a, b = R[k], N[k]
        if a.shape != b.shape:
            print(f"{k:38s} shape {a.shape} vs {b.shape}")
            ok = False
            continue
        if a.dtype.kind == "b" or b.dtype.kind == "b":
            nd = int((a != b).sum())
            print(f"{k:38s} {'identical' if nd == 0 else f'{nd} differ'}")
            ok &= nd == 0
            continue
        a64, b64 = a.astype(np.float64), b.astype(np.float64)
        both_nan = np.isnan(a64) & np.isnan(b64)
        d = np.where(both_nan, 0, np.abs(a64 - b64))
        scale = np.maximum(np.abs(a64), 1e-300)
        rel = np.where(both_nan, 0, d / scale)
        exact = bool(np.all(d == 0))
        print(f"{k:38s} {'bit-identical' if exact else f'max|d|={d.max():.3g} max rel={np.nanmax(rel):.3g}'}")
        ok &= exact or np.nanmax(rel) < 1e-5
    agree = pca_agreement(R["X_pca"], N["X_pca"])
    print(f"{'X_pca |cos| per PC':38s} min={agree.min():.8f} median={np.median(agree):.8f}")
    print(f"{'  PCs 1-10':38s} " + " ".join(f"{x:.6f}" for x in agree[:10]))
    vr = np.abs(R["variance_ratio"] - N["variance_ratio"]) / R["variance_ratio"]
    print(f"{'variance_ratio max rel diff':38s} {vr.max():.3g}")
    print("RESULT:", "PASS" if ok else "CHECK")


if __name__ == "__main__":
    main(*sys.argv[1:3])
