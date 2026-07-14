#!/usr/bin/env python3
# WAJ 2026-07-02
"""
corr2_finegrain_corrs.py
========================

Follow-up to corr1: does the ATAC-vs-sequence attribution coupling vary with POSITION inside
the 4096-bp window?  In particular, is it stronger in the central ~400 bp (where the bound
motif typically sits) than in the flanks?

Each window's `atac_attrs`, `pos_attrs`, `neg_attrs` are length-4096 per-position vectors.
We answer the question two complementary ways:

  (A) REGIONAL per-window correlation  [primary]
      Slice each window into equal-width position bins, recompute the SAME per-window
      correlation metric within each bin, and average 3across windows -> a "mean correlation
      vs position" profile.  Plus an explicit equal-width center-400 vs flank-400 contrast.

  (B) PER-POSITION cross-window correlation  [complementary]
      For each of the 4096 offsets, correlate the channel values ACROSS windows -> a length
      4096 correlation profile.

Both views are computed for the same two relationships as before:
    atac vs pos  in high-confidence POSITIVES (prob >= 0.99)
    atac vs neg  in high-confidence NEGATIVES (prob <= 0.01)

---
Usage:
    python3 corr2_finegrain_corrs.py
    python3 corr2_finegrain_corrs.py --tf FOXA1 --cell-line LNCAP
    python3 corr2_finegrain_corrs.py --bin-size 16 --center-bp 200

Cell line is spelled as in the parquet filename (e.g. 22Rv1, LNCAP, A-375).
"""
import os
import numpy as np
import polars as pl
import argparse
from pathlib import Path
from tqdm.auto import tqdm

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")


# --------------------------------------------------------------------- #
ROOT_DIR = str(Path(__file__).resolve().parents[1])   # .../src/analysis/interpretability
DEFAULT_TF = "AR"
DEFAULT_CELL_LINE = "VCAP"
# --------------------------------------------------------------------- #


def smooth(v, k):
    if k <= 1:
        return v
    kern = np.ones(k) / k
    return np.convolve(v, kern, mode="same")


def filter_expr(col: str, op: str, thresh: float) -> pl.Expr:
    c = pl.col(col)
    return {">=": c >= thresh, "<=": c <= thresh, ">": c > thresh, "<": c < thresh}[op]


def make_bins(width: int, bin_size: int):
    """Contiguous equal-width tiling of [0, width). Returns list of (lo, size, center_offset)."""
    bins = []
    for lo in range(0, width - bin_size + 1, bin_size):
        center = lo + bin_size / 2.0
        bins.append((lo, bin_size, center - width / 2.0))   # offset relative to window center
    return bins


def pearson_rowwise(x, y):
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    num = np.sum(xc * yc, axis=1)
    den = np.sqrt(np.sum(xc**2, axis=1) * np.sum(yc**2, axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def rank_rowwise(a):
    return np.argsort(np.argsort(a, axis=1, stable=True), axis=1, stable=True)


def spearman_rowwise(x, y):
    return pearson_rowwise(rank_rowwise(x), rank_rowwise(y))


def get_attrs(input_path, seq_attr_direction="pos"):
    print("=" * 90)
    print("Reading in Attributions...")
    print("=" * 90)
    if seq_attr_direction == "pos":
        df = (pl.scan_parquet(input_path)
                .filter(filter_expr(col="probabilities", op=">=", thresh=0.99))
                .select("atac_attrs", f"{seq_attr_direction}_attrs")
                .collect())
    elif seq_attr_direction == "neg":
        df = (pl.scan_parquet(input_path)
                .filter(filter_expr(col="probabilities", op="<=", thresh=0.01))
                .select("atac_attrs", f"{seq_attr_direction}_attrs")
                .collect())   

    X = np.asarray(df["atac_attrs"].to_list(), dtype=np.float32)
    Y = np.asarray(df[f"{seq_attr_direction}_attrs"].to_list(), dtype=np.float32)
    if X.shape == Y.shape:
        return X, Y
    else:
        raise ValueError(f"Matrices are not of same shape: {X.shape} != {Y.shape}")


def rowwise_correlations(X, Y, bins):
    print("=" * 90)
    print("Beginning Row-Wise Correlation Analysis...")
    print("=" * 90)
    bin_corrs = dict()
    bin_spearmans = dict()

    for bin in tqdm(bins, desc="Bins"):
        lo, size, offset = bin
        xs, ys = X[:, lo:lo + size], Y[:, lo:lo + size]

        bin_corrs[lo] = np.nanmean(pearson_rowwise(xs, ys))
        bin_spearmans[lo] = np.nanmean(spearman_rowwise(xs, ys))

    return bin_corrs, bin_spearmans


def positional_corrs(X, Y):
    print("=" * 90)
    print("Beginning Positional Correlation Analysis...")
    print("=" * 90)

    width = X.shape[1]

    pos_r = np.full(width, np.nan)
    for j in tqdm(range(width), desc="Position"):
        xs = X[:, j]
        ys = Y[:, j]
        xs = xs.astype(np.float64, copy=False)
        ys = ys.astype(np.float64, copy=False)
        xc = xs - xs.mean(keepdims=True)
        yc = ys - ys.mean(keepdims=True)

        num = np.sum(xc * yc)
        den = np.sqrt(np.sum(xc**2) * np.sum(yc**2))
        with np.errstate(invalid="ignore", divide="ignore"):
            pos_r[j] = num / den

    return pos_r


def plot_all(prof, pos_r, center_bp, out_dir, 
             seq_attr_direction="pos", 
             bin_size=20, 
             width=4096, 
             figsize=(16,9)):
    print("=" * 90)
    print("Plotting...")
    print("=" * 90)
    
    half_c = center_bp / 2.0
    fig, axes = plt.subplots(2, 1, figsize = figsize)

    # Rowwise Correlations
    ax = axes[0]
    off = prof['center_offset'].to_numpy()
    ax.plot(off, prof["pearson_mean"].to_numpy(), "-o", ms=3, color="#3b75af", label="Pearson")
    ax.plot(off, prof["spearman_mean"].to_numpy(), "-o", ms=3, color="#c1622f",
            label="Spearman")
    ax.axvspan(-half_c, half_c, color="gold", alpha=0.18, label=f"center {center_bp} bp")
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    ax.set_title("(A) regional per-window correlation")
    ax.set_xlabel("position offset from window center (bp)")
    ax.set_ylabel(f"mean corr(atac_attrs, {seq_attr_direction}_attrs")
    ax.legend(fontsize=8)

    # Positional Correlations
    ax = axes[1]
    pos_off = np.arange(width) - width / 2.0
    ax.plot(pos_off, pos_r, color="0.6", lw=0.4, label="raw")
    ax.plot(pos_off, smooth(pos_r, 32), color="#2f7d3b", lw=1.6, label="smoothed (25 bp)")
    ax.axvspan(-half_c, half_c, color="gold", alpha=0.18, label=f"center {center_bp} bp")
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    ax.set_title("(B) per-position cross-window correlation")
    ax.set_xlabel("position offset from window center (bp)")
    ax.set_ylabel("corr across windows")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"bin{bin_size}_{seq_attr_direction}_corrs.png"), dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--rootdir", default=ROOT_DIR)
    parser.add_argument("--tf", default=DEFAULT_TF)
    parser.add_argument("--cell-line", default=DEFAULT_CELL_LINE,
                        help="Cell line, parquet spelling (default VCAP)")
    parser.add_argument("--seq-direction", default="pos", choices = ["pos", "neg"])
    parser.add_argument("--bin-size", type=int, default=20)
    parser.add_argument("--center-bp", type=int, default=400)
    args = parser.parse_args()

    tag = f"{args.tf}_{args.cell_line}"
    input_path = os.path.join(args.rootdir, f"data/attribution_matrices/{tag}_attrs.parquet")
    if not os.path.exists(input_path):
        raise SystemExit(f"Input parquet not found: {input_path}")
    out_dir = os.path.join(args.rootdir, f"attr_analyses_output/{tag}/positional_corr")
    os.makedirs(out_dir, exist_ok=True)

    X, Y = get_attrs(input_path, seq_attr_direction=args.seq_direction)
    n, width = X.shape
    bins = make_bins(width, args.bin_size)
    print(f"Window: {width} bp | bins: {len(bins)} x {args.bin_size} bp | "
        f"center region: {args.center_bp} bp\n")

    bin_corrs, bin_spearmans = rowwise_correlations(X, Y, bins)
    prof = pl.DataFrame({
        "bin": list(range(len(bins))),
        "lo": [b[0] for b in bins],
        "center_offset": [b[2] for b in bins],
        "pearson_mean": list(bin_corrs.values()),
        "spearman_mean": list(bin_spearmans.values())
    })
    prof.write_csv(os.path.join(out_dir, f"{tag}_{args.seq_direction}rowwise_correlations.csv"))

    pos_r = positional_corrs(X, Y)
    posprof_path = os.path.join(out_dir, f"{tag}_{args.seq_direction}_positional_profile.csv")
    pl.DataFrame({"offset": np.arange(width) - width // 2, "pearson": pos_r}).write_csv(posprof_path)

    plot_all(prof, pos_r, 
             center_bp=args.center_bp, 
             out_dir=out_dir, 
             seq_attr_direction=args.seq_direction,
             bin_size=args.bin_size,
             width=width,
             figsize=(16,9))

if __name__ == "__main__":
    main()
