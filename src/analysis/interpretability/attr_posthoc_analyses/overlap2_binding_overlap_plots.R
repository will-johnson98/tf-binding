#!/usr/bin/env Rscript
# binding_overlap_plots.R  --  WAJ / 2026-07-01
# Visualize INSIDE vs OUTSIDE ground-truth-binding-site attributions.
# Reads the intermediates written by overlap1_binding_overlap_analysis.R (no re-streaming
# of the 36 GB parquet).
#
# Figures (attr_analyses_output/<TF>_<CELL_LINE>/binding_overlap/):
#   fig1_perwindow_mean_boxplot.png   per-window mean attr, inside vs outside  (PRIMARY)
#   fig2_position_level_boxplot.png   per-position distribution, asinh scale
#   fig3_fold_enrichment.png          inside/outside mean ratio  (HEADLINE)
#   fig4_perwindow_peak_boxplot.png   per-window PEAK attr, inside vs outside
# plus perwindow_paired_stats.csv
#
# Usage: Rscript overlap2_binding_overlap_plots.R --tf AR --cell-line 22Rv1

suppressMessages({
  library(data.table)
  library(ggplot2)
  library(scales)
  library(optparse)
})

## Defaults; override with --tf / --cell-line / --rootdir.
TF        <- "AR"
CELL_LINE <- "22Rv1"        # parquet spelling, as used by overlap1's output dir
ROOT      <- "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/src/analysis/interpretability/"

.opt <- parse_args(OptionParser(option_list = list(
  make_option("--tf",        type = "character", default = TF),
  make_option("--cell-line", type = "character", default = CELL_LINE, dest = "cell_line",
              help = "Cell line, parquet spelling [default %default]"),
  make_option("--rootdir",   type = "character", default = ROOT, dest = "rootdir")
)))
TF        <- .opt$tf
CELL_LINE <- .opt$cell_line
ROOT      <- .opt$rootdir

DIR   <- file.path(ROOT, "attr_analyses_output",
                   paste0(TF, "_", CELL_LINE), "binding_overlap")
# Inputs are overlap1's intermediates; a missing DIR means overlap1 did not run for this pair.
if (!dir.exists(DIR)) stop("No overlap1 output for ", TF, "/", CELL_LINE, ": ", DIR)

SETS  <- c("positive", "negative", "atac")
LABS  <- c(positive = "Positive attribution\n(prob ≥ 0.99)",
           negative = "Negative attribution\n(prob ≤ 0.01)",
           atac     = "ATAC attribution\n(all windows)")

# expected direction of the "stronger inside" signal
DIR_SIGN <- c(positive = 1, negative = -1, atac = 1)   # neg: more-negative = stronger
COL_IN  <- "#C0392B"; COL_OUT <- "#5D6D7E"
theme_set(theme_bw(base_size = 13) +
          theme(strip.background = element_rect(fill = "grey92"),
                strip.text = element_text(face = "bold"),
                panel.grid.minor = element_blank(),
                legend.position = "top"))

summary_dt <- fread(file.path(DIR, "summary_stats.csv"))

# ---- assemble per-window long tables -------------------------------------
pw_all <- rbindlist(lapply(SETS, function(s) {
  d <- fread(file.path(DIR, paste0("perwindow_", s, ".csv")))
  d[, setraw := s]; d
}), fill = TRUE)
pw_all[, set := factor(setraw, levels = SETS, labels = LABS[SETS])]

mklong <- function(dt, invar, outvar) {
  rbindlist(list(
    dt[, .(set, region = "inside",  value = get(invar))],
    dt[, .(set, region = "outside", value = get(outvar))]))
}
pw_mean <- mklong(pw_all, "mean_in", "mean_out")
# directional PEAK: neg_attrs are <=0 so the peak is the MOST-NEGATIVE (min);
# pos/atac peak is the MAX.
pw_all[, peak_in  := ifelse(setraw == "negative", min_in,  max_in)]
pw_all[, peak_out := ifelse(setraw == "negative", min_out, max_out)]
pw_peak <- mklong(pw_all, "peak_in", "peak_out")
pw_mean[, region := factor(region, c("inside", "outside"))]
pw_peak[, region := factor(region, c("inside", "outside"))]

# ---- per-window paired statistics ----------------------------------------
stat_rows <- lapply(SETS, function(s) {
  d <- pw_all[set == LABS[s]]
  d <- d[is.finite(mean_in) & is.finite(mean_out)]
  sgn <- DIR_SIGN[s]
  # "stronger inside" = signed inside beats signed outside
  stronger <- mean(sgn * d$mean_in > sgn * d$mean_out)
  wt <- suppressWarnings(wilcox.test(d$mean_in, d$mean_out, paired = TRUE))
  data.table(set = s, n_windows = nrow(d),
             median_mean_in = median(d$mean_in), median_mean_out = median(d$mean_out),
             pct_windows_stronger_inside = round(100 * stronger, 1),
             wilcoxon_p = wt$p.value)
})
paired_stats <- rbindlist(stat_rows)
fwrite(paired_stats, file.path(DIR, "perwindow_paired_stats.csv"))
cat("Per-window paired stats:\n"); print(paired_stats)

# annotation text per facet (ratio from exact moments + % windows stronger inside)
ann <- merge(summary_dt[, .(set, ratio = inside_over_outside_ratio,
                            n_in = n_inside, n_out = n_outside)],
             paired_stats[, .(set, pct_windows_stronger_inside, n_windows)], by = "set")
ann[, lab := sprintf("inside/outside mean = %.2f×\n%.0f%% of %s windows stronger inside",
                     ratio, pct_windows_stronger_inside, format(n_windows, big.mark = ","))]
ann[, set := factor(set, levels = SETS, labels = LABS[SETS])]

# ============================ FIG 1: per-window mean =======================
p1 <- ggplot(pw_mean, aes(region, value, fill = region)) +
  geom_violin(alpha = 0.35, colour = NA, scale = "width", trim = TRUE) +
  geom_boxplot(width = 0.22, outlier.shape = NA, alpha = 0.9) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 2.6,
               fill = "white", colour = "black") +
  facet_wrap(~ set, scales = "free_y") +
  scale_fill_manual(values = c(inside = COL_IN, outside = COL_OUT), guide = "none") +
  geom_text(data = ann, aes(x = 1.5, y = Inf, label = lab), inherit.aes = FALSE,
            vjust = 1.2, size = 3.1, lineheight = 0.95) +
  labs(title = sprintf("%s / %s  —  per-window mean attribution: inside vs outside binding sites", TF, CELL_LINE),
       subtitle = "Each point = one 4096-bp window overlapping ≥1 ground-truth site. White diamond = group mean.",
       x = NULL, y = "per-window mean attribution") +
  expand_limits(y = 0)
ggsave(file.path(DIR, "fig1_perwindow_mean_boxplot.png"), p1, width = 12, height = 5.2, dpi = 150)

# ============================ FIG 2: position level ========================
asinh_trans <- trans_new("asinh", asinh, sinh)
pos_long <- rbindlist(lapply(SETS, function(s) {
  smp <- readRDS(file.path(DIR, paste0("sample_", s, ".rds")))
  rbindlist(list(
    data.table(setraw = s, region = "inside",  value = smp$inside),
    data.table(setraw = s, region = "outside", value = smp$outside)))
}))
# robust per-set y-window (central 99%) so boxes aren't crushed by rare extreme
# outliers; whiskers/outliers beyond are omitted for display only.
clip <- pos_long[, .(lo = min(0, quantile(value, 0.005)),
                     hi = max(0, quantile(value, 0.995))), by = setraw]
pos_long <- merge(pos_long, clip, by = "setraw")
pos_clip <- pos_long[value >= lo & value <= hi]
pos_clip[, set := factor(setraw, levels = SETS, labels = LABS[SETS])]
pos_clip[, region := factor(region, c("inside", "outside"))]

# exact means (over ALL positions) for the diamonds -> unaffected by clipping
exact_mean <- summary_dt[, .(setraw = set, inside = mean_inside, outside = mean_outside)]
exact_mean <- melt(exact_mean, id.vars = "setraw", variable.name = "region", value.name = "value")
exact_mean[, set := factor(setraw, levels = SETS, labels = LABS[SETS])]
exact_mean[, region := factor(region, c("inside", "outside"))]

p2 <- ggplot(pos_clip, aes(region, value, fill = region)) +
  geom_violin(alpha = 0.35, colour = NA, scale = "width", trim = TRUE) +
  geom_boxplot(width = 0.20, outlier.shape = NA, alpha = 0.9) +
  geom_point(data = exact_mean, aes(region, value), inherit.aes = FALSE,
             shape = 23, size = 2.8, fill = "white", colour = "black") +
  facet_wrap(~ set, scales = "free_y") +
  scale_fill_manual(values = c(inside = COL_IN, outside = COL_OUT), guide = "none") +
  labs(title = sprintf("%s / %s  —  per-position attribution distribution", TF, CELL_LINE),
       subtitle = "All positions inside vs outside binding sites (y zoomed to central 99% per panel). White diamond = exact mean over ALL positions.",
       x = NULL, y = "attribution value")
ggsave(file.path(DIR, "fig2_position_level_boxplot.png"), p2, width = 12, height = 5.2, dpi = 150)

# ============================ FIG 3: fold enrichment =======================
fe <- summary_dt[, .(set, ratio = inside_over_outside_ratio)]
fe[, set := factor(set, levels = SETS, labels = c("Positive\n(prob≥0.99)",
                                                  "Negative\n(prob≤0.01)",
                                                  "ATAC\n(all windows)"))]
p3 <- ggplot(fe, aes(set, ratio, fill = set)) +
  geom_col(width = 0.62, colour = "black") +
  geom_hline(yintercept = 1, linetype = "dashed", colour = "grey40") +
  geom_text(aes(label = sprintf("%.2f×", ratio)), vjust = -0.4, size = 4.6, fontface = "bold") +
  scale_fill_manual(values = c("#2E86C1", "#8E44AD", "#E67E22"), guide = "none") +
  labs(title = sprintf("%s / %s  —  attribution enrichment INSIDE binding sites", TF, CELL_LINE),
       subtitle = "Ratio of mean attribution magnitude inside vs outside ground-truth sites (1× = no difference)",
       x = NULL, y = "inside / outside  mean ratio") +
  expand_limits(y = c(0, max(fe$ratio) * 1.12))
ggsave(file.path(DIR, "fig3_fold_enrichment.png"), p3, width = 8, height = 5, dpi = 150)

# ============================ FIG 4: per-window peak =======================
# strongest single position per window, in the channel's expected direction
# (pos/atac -> most positive; neg -> most negative).
p4 <- ggplot(pw_peak, aes(region, value, fill = region)) +
  geom_violin(alpha = 0.35, colour = NA, scale = "width", trim = TRUE) +
  geom_boxplot(width = 0.22, outlier.shape = NA, alpha = 0.9) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 2.6,
               fill = "white", colour = "black") +
  facet_wrap(~ set, scales = "free_y") +
  scale_y_continuous(trans = asinh_trans) +
  scale_fill_manual(values = c(inside = COL_IN, outside = COL_OUT), guide = "none") +
  labs(title = sprintf("%s / %s  —  per-window PEAK attribution: inside vs outside binding sites", TF, CELL_LINE),
       subtitle = "Strongest single-position attribution per window (pos/ATAC = max, negative = most-negative; asinh scale). White diamond = group mean.",
       x = NULL, y = "per-window peak attribution  (asinh-scaled axis)")
ggsave(file.path(DIR, "fig4_perwindow_peak_boxplot.png"), p4, width = 12, height = 5.2, dpi = 150)

cat("\nWrote figures to", DIR, "\n")
