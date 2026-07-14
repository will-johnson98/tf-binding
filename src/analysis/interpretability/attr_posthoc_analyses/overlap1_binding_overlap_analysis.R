#!/usr/bin/env Rscript
# overlap1_binding_overlap_analysis.R  --  WAJ / 2026-07-01  (HPC rewrite 2026-07-02)


# ---------------------------------------------------------------------------
# Do attributions concentrate INSIDE ground-truth TF binding sites vs OUTSIDE?
#
# For each row (a model region) the 4096-long attribution vectors are the model's
# fixed 4096-bp receptive field CENTERED on the region midpoint:
#     center      = (start + end) %/% 2
#     coord(i)    = center - 2048 + i          (i = 0..4095, 0-based genome coord)
# so the window spans [center-2048, center+2048).
#
# Ground-truth binding sites come from a BED (0-based half-open). GenomicRanges
# findOverlaps() flags which windows touch a binding site; exact interval->index
# math then labels every within-window position INSIDE / OUTSIDE.
#
# Three analyses:
#     pos_attrs   over high-confidence POSITIVES   (probabilities >= 0.99)
#     neg_attrs   over high-confidence NEGATIVES   (probabilities <= 0.01)
#     atac_attrs  over ALL windows                 (no confidence filter)
#
# ---------------------------------------------------------------------------


suppressMessages({
  library(arrow)
  library(dplyr)
  library(data.table)
  library(GenomicRanges)
  library(matrixStats)
  library(optparse)
})

## ------------------------------- config -----------------------------------
## Defaults; override with --tf / --cell-line / --rootdir.
## NB --rootdir is the REPO root here (this script also needs data/transcription_factors/),
## unlike the corr/heatmaps scripts whose rootdir is the interpretability dir.
TF        <- "AR"
CELL_LINE <- "VCAP"         # parquet spelling (22Rv1, LNCAP, A-375); BED path uppercases it
ROOT      <- "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/"

.opt <- parse_args(OptionParser(option_list = list(
  make_option("--tf",        type = "character", default = TF),
  make_option("--cell-line", type = "character", default = CELL_LINE, dest = "cell_line",
              help = "Cell line, parquet spelling [default %default]"),
  make_option("--rootdir",   type = "character", default = ROOT, dest = "rootdir",
              help = "Repo root [default %default]")
)))
TF        <- .opt$tf
CELL_LINE <- .opt$cell_line
ROOT      <- .opt$rootdir

INTERP    <- file.path(ROOT, "src/analysis/interpretability")
BED_FILE  <- file.path(ROOT, "data/transcription_factors", TF, "merged",
                       paste0(toupper(CELL_LINE), "_", TF, "_merged.bed"))
ATTR_FILE <- file.path(INTERP, "data/attribution_matrices",
                       paste0(TF, "_", CELL_LINE, "_attrs.parquet"))
OUT_DIR   <- file.path(INTERP, "attr_analyses_output",
                       paste0(TF, "_", CELL_LINE), "binding_overlap")
if (!file.exists(BED_FILE))  stop("Ground-truth BED not found: ", BED_FILE)
if (!file.exists(ATTR_FILE)) stop("Input parquet not found: ", ATTR_FILE)
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

NCOL       <- 4096L
HALF       <- NCOL %/% 2L
SAMPLE_CAP <- 150000L    # per-group position-level sample kept for the plotting script

# the three analyses: name, value column, confidence filter ("none" = all rows)
ANALYSES <- list(
  list(set = "positive", col = "pos_attrs", filter = "positive"),
  list(set = "negative", col = "neg_attrs", filter = "negative"),
  list(set = "atac",     col = "atac_attrs", filter = "none")
)

## ------------------------- ground-truth binding ---------------------------
cat("Loading ground-truth binding sites:", basename(BED_FILE), "\n")
bed <- fread(BED_FILE, header = FALSE, col.names = c("chr", "bs", "be", "score"))
bs0 <- bed$bs; be0 <- bed$be                              # 0-based half-open
gt  <- GRanges(bed$chr, IRanges(start = bs0 + 1L, end = be0))  # 1-based inclusive
cat(sprintf("  %d sites  |  %.2f Mb total  |  median width %d bp\n",
            length(gt), sum(width(gt)) / 1e6, median(width(gt))))

## ------------------------------ helpers ------------------------------------

# population moments (divide by n), matching the original streaming summary.
pop_stats <- function(x) {
  n  <- length(x)
  mu <- sum(x) / n
  sd <- sqrt(max(0, sum(x * x) / n - mu * mu))
  list(n = n, mean = mu, sd = sd, min = min(x), max = max(x))
}

# simple uniform subsample for the position-level plot (bounded file size).
subsample <- function(x, cap) if (length(x) > cap) x[sample.int(length(x), cap)] else x

## --------------------------- per-set analysis ------------------------------
analyze_set <- function(set, col, filter) {
  cat(sprintf("\n=== %s  (col=%s) ===\n", set, col))
  t0 <- Sys.time()

  # ---- read the subset fully into memory (column projection + filter push-down)
  q <- open_dataset(ATTR_FILE) |>
    select(all_of(c("chr_name", "start", "end", "probabilities", col)))
  if (filter == "positive") q <- q |> filter(probabilities >= 0.99)
  if (filter == "negative") q <- q |> filter(probabilities <= 0.01)
  DF <- as.data.table(collect(q))
  n_rows <- nrow(DF)
  cat(sprintf("  matched rows: %d\n", n_rows))

  chr    <- DF$chr_name
  center <- (DF$start + DF$end) %/% 2L
  w0     <- center - HALF                       # 0-based coord of within-window position i=0

  # dense value matrix: one row per window, one column per within-window position
  M <- do.call(rbind, DF[[col]])

  # ---- inside-site mask via GenomicRanges + exact interval->index math --------
  win <- GRanges(chr, IRanges(start = w0 + 1L, end = w0 + NCOL))
  ov  <- findOverlaps(win, gt)
  qh  <- queryHits(ov); sh <- subjectHits(ov)

  maskM <- matrix(FALSE, nrow = n_rows, ncol = NCOL)   # TRUE = position inside a binding site
  if (length(qh)) {
    w0v  <- w0[qh]
    lo   <- pmax(w0v, bs0[sh]) - w0v                   # inclusive rel start (0-based)
    hi   <- pmin(w0v + NCOL, be0[sh]) - w0v            # exclusive rel end
    keep <- hi > lo
    qh <- qh[keep]; lo <- lo[keep]; hi <- hi[keep]
    cnt  <- hi - lo
    rows <- rep(qh, cnt)
    cols <- sequence(cnt, from = lo + 1L)              # 1-based columns lo+1 .. hi
    maskM[cbind(rows, cols)] <- TRUE
  }

  # ---- exact inside / outside position-level moments --------------------------
  inside  <- M[maskM]
  outside <- M[!maskM]
  fi <- pop_stats(inside)
  fo <- pop_stats(outside)

  # ---- per-window summaries for windows overlapping >= 1 binding site ---------
  in_cnt  <- rowSums(maskM)
  out_cnt <- NCOL - in_cnt
  ow      <- which(in_cnt > 0L)                        # windows overlapping a site
  n_win_ov <- length(ow)

  sum_in   <- rowSums(M * maskM)
  mean_in  <- sum_in / in_cnt
  mean_out <- (rowSums(M) - sum_in) / out_cnt
  Min_in <- M; Min_in[!maskM] <- Inf                   # mask out non-inside for row min
  Max_in <- M; Max_in[!maskM] <- -Inf
  Min_out <- M; Min_out[maskM] <- Inf
  Max_out <- M; Max_out[maskM] <- -Inf
  perwin <- data.table(
    chr = chr, center = center,
    n_in = in_cnt, n_out = out_cnt,
    mean_in = mean_in, mean_out = mean_out,
    max_in = rowMaxs(Max_in), max_out = rowMaxs(Max_out),
    min_in = rowMins(Min_in), min_out = rowMins(Min_out)
  )[ow]
  # windows with no outside positions (never happens for 4096-bp windows vs small
  # sites, but keep the NA convention of the original just in case)
  perwin[n_out == 0L, c("mean_out", "max_out", "min_out") := NA_real_]

  dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat(sprintf("  DONE %d windows (%d overlap bed, %.1f%%) in %.0fs\n",
              n_rows, n_win_ov, 100 * n_win_ov / n_rows, dt))
  cat(sprintf("  INSIDE  n=%d mean=%.5g sd=%.5g\n", fi$n, fi$mean, fi$sd))
  cat(sprintf("  OUTSIDE n=%d mean=%.5g sd=%.5g  |  inside/outside mean ratio=%.2f\n",
              fo$n, fo$mean, fo$sd, fi$mean / fo$mean))

  # save intermediates for the plotting script
  saveRDS(list(inside = subsample(inside, SAMPLE_CAP), outside = subsample(outside, SAMPLE_CAP)),
          file.path(OUT_DIR, paste0("sample_", set, ".rds")))
  fwrite(perwin, file.path(OUT_DIR, paste0("perwindow_", set, ".csv")))

  list(set = set, col = col, n_windows = n_rows, n_windows_overlap = n_win_ov,
       inside = fi, outside = fo)
}

## ------------------------------- driver ------------------------------------
summ <- list()
for (a in ANALYSES) {
  r <- analyze_set(a$set, a$col, a$filter)
  summ[[a$set]] <- data.table(
    set = r$set, value_col = r$col,
    n_windows = r$n_windows, n_windows_overlap = r$n_windows_overlap,
    pct_windows_overlap = round(100 * r$n_windows_overlap / r$n_windows, 2),
    n_inside = r$inside$n,  mean_inside  = r$inside$mean,  sd_inside  = r$inside$sd,
    min_inside = r$inside$min, max_inside = r$inside$max,
    n_outside = r$outside$n, mean_outside = r$outside$mean, sd_outside = r$outside$sd,
    min_outside = r$outside$min, max_outside = r$outside$max,
    inside_over_outside_ratio = r$inside$mean / r$outside$mean)
}
summary_dt <- rbindlist(summ)
fwrite(summary_dt, file.path(OUT_DIR, "summary_stats.csv"))
cat("\n================ SUMMARY ================\n")
print(summary_dt)
cat("\nWrote outputs to", OUT_DIR, "\n")
