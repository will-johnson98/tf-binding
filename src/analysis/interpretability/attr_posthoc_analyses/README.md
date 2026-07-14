# attr_posthoc_analyses

Post-hoc analyses of DeepSeq attribution matrices — three independent branches: **ATAC↔sequence
attribution correlation**, **genome-aligned heatmaps**, and **inside-vs-outside binding-site
overlap**. Every script consumes the same per-position attribution parquet and writes figures +
CSVs. Nothing here computes attributions; that happens upstream during inference and is reshaped by
`../attr_matrices.py`.

This directory supersedes the old `hpc_scripts/` (pre-rename copy kept at
`../.20260714_hpcScripts_backup/`, whose README holds the numerical validation notes). The repo's
top-level `CLAUDE.md` still refers to the old name.

## Input

Every script reads one parquet:

```
../data/attribution_matrices/{TF}_{CELL}_attrs.parquet
```

produced by `../attr_matrices.py`. Each row is one 4096-bp model window centered on a region
midpoint. Columns used here: `chr_name`, `start`, `end`, `probabilities`, `targets`, `predicted`,
and the three length-4096 per-position vectors `pos_attrs`, `neg_attrs`, `atac_attrs`. The parquet
is **already filtered** to high-confidence rows (`probabilities <= 0.01 | >= 0.99`) and runs 15–21 GB
per pair.

`overlap1` additionally needs a ground-truth ChIP BED:

```
../../../../data/transcription_factors/{TF}/merged/{CELL_UPPER}_{TF}_merged.bed
```

**Cell line is spelled as in the parquet filename** — `22Rv1`, `LNCAP`, `A-375` (`overlap1`
uppercases it for the BED path). On disk today: `AR` × {`22Rv1`, `A-375`, `C4-2`, `LNCAP`, `VCAP`}.

## Quick start

The happy path is the batch runner — it fans out all five stages across TF/cell-line pairs with GNU
Parallel:

```bash
conda activate pterodactyl
bash run_attr_posthoc_analyses.sh --pairs pairs.tsv           # all stages, 2 pairs at a time
bash run_attr_posthoc_analyses.sh --pairs pairs.tsv --dry-run # print the plan, run nothing
bash run_attr_posthoc_analyses.sh --pairs pairs.tsv --stages overlap1,overlap2 --jobs 1
```

Parallelism is **across pairs, not stages**; `pairs.tsv` is `TF<TAB>CELL_LINE` (regenerate with
`../utils/find_samples.sh <TF>`). Logs land in `logs/hpc/<TF>_<CELL>.log` plus `logs/hpc/joblog`
(created on first run). Requires GNU Parallel, the `pterodactyl` env, and `Rscript`.

Or run any stage on its own — every script takes `--tf` / `--cell-line` (the top-of-file constants
are just defaults):

```bash
python  corr1_atac_attr_correlations.py    --tf AR --cell-line VCAP
python  corr2_finegrain_corrs.py           --tf AR --cell-line VCAP --seq-direction neg
Rscript heatmaps1_genome_heatmaps.R        --tf AR --cell-line 22Rv1 --alignment both
Rscript overlap1_binding_overlap_analysis.R --tf AR --cell-line 22Rv1
Rscript overlap2_binding_overlap_plots.R   --tf AR --cell-line 22Rv1
```

## The scripts

| stage | lang | what it answers | writes to `{TF}_{CELL}/…` |
|---|---|---|---|
| `corr1_atac_attr_correlations.py` | Python (polars, scipy) | Do ATAC and sequence attributions co-vary *within* a window? Per-window Pearson/Spearman across the 4096 positions, plus one pooled correlation. `atac×pos` on positives (≥0.99), `atac×neg` on negatives (≤0.01). | `attr_correlations/` — `*_correlation_summary.csv`, `*_perwindow.parquet`, `*_hist.png` |
| `corr2_finegrain_corrs.py` | Python (polars, matplotlib) | Does that coupling depend on *position* — stronger in the central ~400 bp than the flanks? Binned regional profile + per-offset cross-window profile. Extra flags: `--seq-direction pos\|neg`, `--bin-size`, `--center-bp`. | `positional_corr/` — `*rowwise_correlations.csv`, `*_positional_profile.csv`, `bin{N}_*_corrs.png` |
| `heatmaps1_genome_heatmaps.R` | R (arrow, ComplexHeatmap) | Genome-aligned heatmaps of `pos_attrs` for predicted positives (≥0.99), one PNG per chromosome. `--alignment center\|absolute\|both`. | `heatmaps/<mode>/` and `heatmaps/<mode>_complexheatmap/` |
| `heatmaps1a_app.R` | R / Shiny | Interactive browser over the same heatmaps (hover, brush, per-chromosome). No CLI — see below. | — |
| `overlap1_binding_overlap_analysis.R` | R (arrow, GenomicRanges) | Do attributions concentrate *inside* ground-truth ChIP peaks vs outside? Labels every within-window position inside/outside via `findOverlaps` + interval math. | `binding_overlap/` — `summary_stats.csv`, `perwindow_*.csv`, `sample_*.rds` |
| `overlap2_binding_overlap_plots.R` | R (ggplot2) | Plots overlap1's intermediates (fold-enrichment, boxplots, paired Wilcoxon). | `binding_overlap/` — `fig1…fig4*.png`, `perwindow_paired_stats.csv` |

`corr1`, `corr2`, `heatmaps1`, and the `overlap1 → overlap2` chain are mutually independent. Only
`overlap1 → overlap2` is a real producer→consumer edge (the runner skips `overlap2` when `overlap1`
fails); `corr1 → corr2` is a follow-up in intent only — `corr2` re-reads the parquet itself.

`heatmaps1a_app.R` has no flags because it sources the renderer in **library mode**
(`options(genome_heatmaps.lib = TRUE)`), which defines `make_ht()`, the index, and the in-memory data
without running the batch renderer; configuration comes from `heatmaps1_genome_heatmaps.R`'s
top-of-file constants.

## Outputs

Everything writes to the **parent** interpretability directory, one subtree per pair:

```
../attr_analyses_output/{TF}_{CELL}/
  ├── attr_correlations/   # corr1
  ├── positional_corr/     # corr2
  ├── heatmaps/            # heatmaps1  (center/, absolute/, *_complexheatmap/)
  └── binding_overlap/     # overlap1 + overlap2
```

## Known issues

- **A default sweep only produces the `pos` direction of `corr2`.** `--seq-direction` defaults to
  `pos`, and the runner passes only `--tf`/`--cell-line`, so the `neg` outputs are never generated by
  a plain sweep — run `corr2` again with `--seq-direction neg`. (Missing `neg` files are expected, not
  a failure.) Likewise the runner never passes `--alignment`, so `heatmaps1` always renders `both`.
- **`heatmaps1a_app.R` is currently broken.** Two lines need fixing before it launches: `:25` sources
  `heatmaps2_genome_heatmaps.R` (renamed — should be `heatmaps1_genome_heatmaps.R`), and `:44`
  references `TRANSCRIPTION_FACTOR` (the renderer defines `TF`). Launch with
  `Rscript -e 'shiny::runApp("heatmaps1a_app.R", launch.browser = TRUE)'` once patched.
- **`--rootdir` means different things per script.** For `corr1`/`corr2`/`heatmaps1`/`overlap2` it is
  the *interpretability* directory; for `overlap1` it is the *repo root* (it also needs
  `data/transcription_factors/`). The runner never passes it, so defaults apply — but it will trip a
  hand-run of `overlap1`.
- **Memory.** `overlap1` loads every window's attrs into RAM (~11 GB for ~340k windows), so peak
  usage ≈ `--jobs × 11 GB`. Hence the `--jobs 2` default; raising it is safe when `--stages` excludes
  `overlap1`.

## Also here

- `playground_corr2.ipynb` — stale exploratory scratch for `corr2`, not part of the pipeline (still
  imports the pre-rename module `hpc_scripts.corr2_positional_correlation` and points at a personal
  `ROOT_DIR`).
- `../.20260714_hpcScripts_backup/` — the pre-rename copy of these scripts, with the original
  validation notes.

## Requirements

- **`pterodactyl` conda env** for the Python stages: polars, scipy, numpy, matplotlib, tqdm.
- **R** for the R stages: arrow, dplyr, data.table, GenomicRanges, ComplexHeatmap, circlize,
  matrixStats, ggplot2, scales, optparse (+ InteractiveComplexHeatmap/shiny for the app).
- **GNU Parallel** for `run_attr_posthoc_analyses.sh`.
