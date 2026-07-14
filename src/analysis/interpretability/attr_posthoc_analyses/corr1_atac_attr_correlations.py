#!/usr/bin/env python3
# WAJ 2026-06-29 
"""
corr1_atac_attr_correlations.py
===============================

Explore how the **ATAC (chromatin-accessibility) attribution channel** co-varies with the
**positive** and **negative sequence-attribution channels** inside high-confidence model
predictions.

Two relationships are explored (both configurable below):

  1. atac_attrs  vs  pos_attrs   within predictions where  probabilities >= 0.99
       -> "When the model is very confident a window IS a binding site, do the positions
           where accessibility matters line up with the positions where the (positive)
           sequence signal matters?"

  2. atac_attrs  vs  neg_attrs   within predictions where  probabilities <= 0.01
       -> "When the model is very confident a window is NOT a binding site, do acceibility
           and the (negative) sequence signal line up positionally?"

Each row of the parquet is one 4096-bp genomic window. `pos_attrs`, `neg_attrs` and
`atac_attrs` are each a length-4096 per-position vector.  A "correlation" here is therefore
the correlation *across the 4096 positions* of a single window — a per-window positional-
profile correlation.

We report, for each relationship:
  * the DISTRIBUTION of per-window Pearson and Spearman correlations (mean, median,
    quartiles, fraction positive, fraction strong), and
  * a single POOLED correlation over every (window, position) point in the subset.

The high-confidence subset is small relative to the full parquet, so we push the probability
filter down and read ONLY the matching rows fully into memory (`scan_parquet(...).filter(...)
.collect()`).  Each per-position channel then becomes a plain dense (n_windows, 4096) numpy
matrix, and correlations are computed with `scipy.stats` — one call per window.

Usage:
    python3 corr1_atac_attr_correlations.py                            # AR / VCAP defaults
    python3 corr1_atac_attr_correlations.py --tf FOXA1 --cell-line LNCAP
    python3 corr1_atac_attr_correlations.py --input /path/to/X_Y_attrs.parquet
    python3 corr1_atac_attr_correlations.py --no-spearman              # faster (Pearson only)

Cell line is spelled as in the parquet filename (e.g. 22Rv1, LNCAP, A-375).
"""

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

# --------------------------------------------------------------------------------------- #
# Defaults — override on the command line for a different TF / cell line.
# --------------------------------------------------------------------------------------- #
ROOT_DIR = str(Path(__file__).resolve().parents[1])   # .../src/analysis/interpretability

DEFAULT_TF = "AR"
DEFAULT_CELL_LINE = "VCAP"
DEFAULT_INPUT_TEMPLATE = "{root}/data/attribution_matrices/{tf}_{cell}_attrs.parquet"

# The relationships to explore. Each correlates the ATAC channel (`x`) against a
# sequence-attribution channel (`y`) within a probability-defined prediction subset.
ANALYSES = [
    dict(
        name="atac_vs_pos__highconf_positive",
        x="atac_attrs",
        y="pos_attrs",
        col="probabilities",
        op=">=",
        thresh=0.99,
        desc="ATAC vs POSITIVE seq-attr  |  high-confidence POSITIVE predictions (prob >= 0.99)",
    ),
    dict(
        name="atac_vs_neg__highconf_negative",
        x="atac_attrs",
        y="neg_attrs",
        col="probabilities",
        op="<=",
        thresh=0.01,
        desc="ATAC vs NEGATIVE seq-attr  |  high-confidence NEGATIVE predictions (prob <= 0.01)",
    ),
]

# Genomic-coordinate / metadata columns carried through to the per-window output (if present).
META_COLS = ["chr_name", "start", "end", "targets", "predicted", "probabilities"]


# --------------------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------------------- #
def filter_expr(col: str, op: str, thresh: float) -> pl.Expr:
    """Build the prediction-subset filter (e.g. probabilities >= 0.99)."""
    c = pl.col(col)
    return {">=": c >= thresh, "<=": c <= thresh, ">": c > thresh, "<": c < thresh,
            "==": c == thresh}[op]


def load_subset(path: str, analysis: dict, available: set):
    """Read the analysis subset fully into memory.

    Returns (meta_df, X, Y) where X and Y are dense (n_windows, width) float64 matrices
    for the two channels being correlated.
    """
    keep = [c for c in META_COLS if c in available]
    df = (pl.scan_parquet(path)
            .filter(filter_expr(analysis["col"], analysis["op"], analysis["thresh"]))
            .select(*keep, analysis["x"], analysis["y"])
            .collect())
    X = np.asarray(df[analysis["x"]].to_list(), dtype=np.float64)
    Y = np.asarray(df[analysis["y"]].to_list(), dtype=np.float64)
    return df.select(keep), X, Y


# --------------------------------------------------------------------------------------- #
# Per-window correlations (one scipy call per window)
# --------------------------------------------------------------------------------------- #
def per_window_correlations(X: np.ndarray, Y: np.ndarray, with_spearman: bool):
    """Per-window Pearson (and Spearman) correlation between the two channel profiles.

    Windows where either profile has zero variance (e.g. an all-zero attribution vector)
    get NaN, which the summary treats as undefined.
    """
    n = X.shape[0]
    pearson = np.full(n, np.nan)
    spearman = np.full(n, np.nan) if with_spearman else None
    xstd = X.std(axis=1)
    ystd = Y.std(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # constant-input warnings -> NaN
        for i in range(n):
            if xstd[i] > 0 and ystd[i] > 0:
                pearson[i] = stats.pearsonr(X[i], Y[i]).statistic
                if with_spearman:
                    spearman[i] = stats.spearmanr(X[i], Y[i]).statistic
    return pearson, spearman


def pooled_correlation(X: np.ndarray, Y: np.ndarray) -> dict:
    """Single Pearson correlation over EVERY (window, position) point in the subset."""
    xf, yf = X.ravel(), Y.ravel()
    if xf.size == 0 or xf.std() == 0 or yf.std() == 0:
        return {"pooled_pearson": None, "n_points": int(xf.size)}
    return {"pooled_pearson": float(stats.pearsonr(xf, yf).statistic),
            "n_points": int(xf.size)}


def summarize(pearson: np.ndarray, spearman, with_spearman: bool) -> dict:
    """Distribution summary of the per-window correlations (same fields as the original)."""
    out = {"n_windows": int(pearson.shape[0])}
    metrics = [("pearson", pearson)] + ([("spearman", spearman)] if with_spearman else [])
    for m, s in metrics:
        valid = s[np.isfinite(s)]
        out[f"{m}_n_valid"] = int(valid.size)
        out[f"{m}_n_undef"] = int(s.shape[0] - valid.size)
        if valid.size:
            out[f"{m}_mean"] = float(valid.mean())
            out[f"{m}_std"] = float(valid.std(ddof=1))            # sample sd
            out[f"{m}_min"] = float(valid.min())
            out[f"{m}_q25"] = float(np.quantile(valid, 0.25))
            out[f"{m}_median"] = float(np.median(valid))
            out[f"{m}_q75"] = float(np.quantile(valid, 0.75))
            out[f"{m}_max"] = float(valid.max())
            out[f"{m}_frac_pos"] = float(np.mean(valid > 0))
            out[f"{m}_frac_gt_0.3"] = float(np.mean(np.abs(valid) > 0.3))
            out[f"{m}_frac_gt_0.5"] = float(np.mean(np.abs(valid) > 0.5))
    return out


# --------------------------------------------------------------------------------------- #
# Optional histogram (skipped silently if matplotlib is unavailable)
# --------------------------------------------------------------------------------------- #
def plot_histograms(pearson, spearman, analysis, with_spearman, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    series = [("pearson", pearson)] + ([("spearman", spearman)] if with_spearman else [])
    fig, axes = plt.subplots(1, len(series), figsize=(5 * len(series), 4), squeeze=False)
    for ax, (m, s) in zip(axes[0], series):
        vals = s[np.isfinite(s)]
        ax.hist(vals, bins=60, color="#3b75af", edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="0.4", lw=1, ls="--")
        if len(vals):
            ax.axvline(vals.mean(), color="crimson", lw=1.5,
                       label=f"mean={vals.mean():.3f}")
            ax.legend(fontsize=8)
        ax.set_title(f"per-window {m}")
        ax.set_xlabel(f"corr({analysis['x']}, {analysis['y']})")
        ax.set_ylabel("windows")
    fig.suptitle(analysis["name"], fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------------------- #
def fmt(v, nd=4):
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tf", default=DEFAULT_TF, help="Transcription factor (default AR)")
    ap.add_argument("--cell-line", default=DEFAULT_CELL_LINE,
                    help="Cell line, parquet spelling (default VCAP)")
    ap.add_argument("--input", default=None,
                    help="Explicit FULL parquet path (overrides --tf/--cell-line template)")
    ap.add_argument("--no-spearman", action="store_true",
                    help="Skip the (slower) Spearman computation")
    ap.add_argument("--rootdir", default=ROOT_DIR, help="root project directory")
    args = ap.parse_args()

    tag = f"{args.tf}_{args.cell_line}"

    input_path = args.input or DEFAULT_INPUT_TEMPLATE.format(
        root=args.rootdir, tf=args.tf, cell=args.cell_line)
    if not os.path.exists(input_path):
        raise SystemExit(f"Input parquet not found: {input_path}")

    output_dir = os.path.join(args.rootdir, f"attr_analyses_output/{tag}/attr_correlations")
    os.makedirs(output_dir, exist_ok=True)

    with_spearman = not args.no_spearman

    print(f"Input : {input_path}")
    print(f"Spearman: {with_spearman}\n")

    available = set(pl.scan_parquet(input_path).collect_schema().names())
    for ch in ("atac_attrs", "pos_attrs", "neg_attrs"):
        if ch not in available:
            raise SystemExit(f"Required column '{ch}' missing from {input_path}")

    summary_rows = []
    for a in ANALYSES:
        print("=" * 90)
        print(a["desc"])
        print("=" * 90)
        t0 = time.time()

        meta, X, Y = load_subset(input_path, a, available)
        pearson, spearman = per_window_correlations(X, Y, with_spearman)
        pooled = pooled_correlation(X, Y)
        stats_ = summarize(pearson, spearman, with_spearman)
        dt = time.time() - t0

        # ---- report ----
        print(f"  windows in subset      : {stats_['n_windows']:,}")
        print(f"  pooled Pearson (all pos): {fmt(pooled['pooled_pearson'])}"
              f"   over {pooled['n_points']:,} (window,position) points")
        for m in (["pearson"] + (["spearman"] if with_spearman else [])):
            if stats_.get(f"{m}_n_valid"):
                print(f"  per-window {m:8s}: "
                      f"mean={fmt(stats_[f'{m}_mean'])}  "
                      f"median={fmt(stats_[f'{m}_median'])}  "
                      f"sd={fmt(stats_[f'{m}_std'])}  "
                      f"IQR=[{fmt(stats_[f'{m}_q25'])}, {fmt(stats_[f'{m}_q75'])}]")
                print(f"  {'':19s} frac r>0={fmt(stats_[f'{m}_frac_pos'],3)}  "
                      f"frac|r|>0.3={fmt(stats_[f'{m}_frac_gt_0.3'],3)}  "
                      f"frac|r|>0.5={fmt(stats_[f'{m}_frac_gt_0.5'],3)}  "
                      f"(undef={stats_[f'{m}_n_undef']})")
        print(f"  [{dt:.1f}s]")

        # ---- persist per-window correlations + histogram ----
        per_win = meta.with_columns(pl.Series("pearson", pearson))
        if with_spearman:
            per_win = per_win.with_columns(pl.Series("spearman", spearman))
        per_win_path = os.path.join(output_dir, f"{tag}_{a['name']}_perwindow.parquet")
        per_win.write_parquet(per_win_path)
        png = plot_histograms(pearson, spearman, a, with_spearman,
                              os.path.join(output_dir, f"{tag}_{a['name']}_hist.png"))
        print(f"  -> per-window : {per_win_path}")
        if png:
            print(f"  -> histogram  : {png}")
        print()

        row = {"analysis": a["name"], "x": a["x"], "y": a["y"],
               "subset": f"{a['col']}{a['op']}{a['thresh']}",
               "pooled_pearson": pooled["pooled_pearson"],
               "n_points": pooled["n_points"], **stats_}
        summary_rows.append(row)

    summary_path = os.path.join(output_dir, f"{tag}_correlation_summary.csv")
    pl.DataFrame(summary_rows).write_csv(summary_path)
    print(f"Combined summary -> {summary_path}")


if __name__ == "__main__":
    main()
