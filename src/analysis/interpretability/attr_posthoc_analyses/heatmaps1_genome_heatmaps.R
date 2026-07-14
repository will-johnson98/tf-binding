#!/usr/bin/env Rscript
# genome_heatmaps.R  -- WAJ 2026-06-29  (HPC rewrite 2026-07-02)
# Genome-aligned heatmaps of predicted-positive `pos_attrs` profiles, one PNG per chromosome.
#
# Each region is a 4096-bp model attribution window; pos_attrs[i] is the positive
# attribution summed across the 4 nucleotide channels at within-window position i.
# Rows of every heatmap are regions ordered by genomic position (5'->3', top->bottom).
#
# Two layouts, chosen by the global ALIGNMENT:
#   "center"   : X = bp offset from window center (-2048..+2047). Stacked importance
#                profile heatmap; every region aligned at its window center.
#   "absolute" : X = true chromosome coordinate (Mb). Each window's profile is scattered
#                into genome bins at its real location -> genome-browser-style diagonal.
#   "both"     : render both of the above.
#
# Outputs per chromosome:
#   heatmaps/<mode>/<chr>_<mode>.png                    original base-R raster PNGs
#   heatmaps/<mode>_complexheatmap/<chr>_<mode>_ch.png  ComplexHeatmap static PNGs
#                                                       (center => Ward.D2 row clustering)
# Interactive (InteractiveComplexHeatmap / Shiny):  see heatmap_app.R
#   Rscript -e 'shiny::runApp("heatmap_app.R", launch.browser = TRUE)'
#
# make_ht(chr, mode) builds the ComplexHeatmap object shared by the static PNGs and the
# interactive app, so both views are identical. Sourcing this file with
# options(genome_heatmaps.lib = TRUE) defines the functions without running the renderer.
#
# HPC note: this reads the predicted-positive subset (probabilities >= 0.99) of the attrs
# parquet directly into memory with arrow + dplyr, then builds each chromosome's matrix by
# slicing that in-memory table. The separate streaming prep step (prep_heatmap_data.py) and
# its per-chromosome .f32 binary handoff are no longer needed.

suppressMessages({
  library(arrow)
  library(dplyr)
  library(data.table)
  library(viridisLite)
  library(ComplexHeatmap)
  library(circlize)
  library(grid)
})
ht_opt$message <- FALSE   # silence ComplexHeatmap's 'magick' rasterization suggestion

## ----------------------------- GLOBAL ARGUMENTS -----------------------------
## Defaults. Override on the command line -- see the CLI block below.
ROOT <- "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/src/analysis/interpretability/"
TF         <- "AR"
CELL_LINE  <- "A-375"       # parquet spelling (22Rv1, LNCAP, A-375)
ALIGNMENT  <- "both"        # "center" | "absolute" | "both"
PROB_MIN   <- 0.99          # predicted-positive threshold
NCOL       <- 4096L         # positions per window
CAP_KEY    <- "cap_p999"    # global color cap: cap_p99 | cap_p995 | cap_p999 | global_max | <numeric>
PALETTE    <- viridis(256)  # dark -> bright; sparse positive signal reads well
NA_COLOR   <- PALETTE[1]    # off-window cells (absolute mode) painted as "no signal"
ABS_NCOL   <- 2000          # number of genome bins (columns) in absolute mode
ABS_AGG    <- "max"         # how to aggregate pos_attrs within a genome bin: "max" | "mean"
DPI        <- 120

# --- ComplexHeatmap additions ---
RENDER_BASE_PNG <- TRUE         # original base-R raster PNGs  (heatmaps/<mode>/)
RENDER_CH_PNG   <- TRUE         # ComplexHeatmap static PNGs   (heatmaps/<mode>_complexheatmap/)
CLUSTER_METHOD  <- "ward.D2"    # row ("sample") clustering for center-aligned heatmaps
CLUSTER_DIST    <- "euclidean"  # distance for row clustering
## ---------------------------------------------------------------------------

# CLI: Rscript heatmaps1_genome_heatmaps.R --tf AR --cell-line 22Rv1 [--alignment both]
# Skipped in library mode (sourced by the Shiny app), where the defaults above stand.
if (!isTRUE(getOption("genome_heatmaps.lib", FALSE))) {
  suppressMessages(library(optparse))
  .opt <- parse_args(OptionParser(option_list = list(
    make_option("--tf",        type = "character", default = TF),
    make_option("--cell-line", type = "character", default = CELL_LINE, dest = "cell_line",
                help = "Cell line, parquet spelling [default %default]"),
    make_option("--rootdir",   type = "character", default = ROOT, dest = "rootdir"),
    make_option("--alignment", type = "character", default = ALIGNMENT,
                help = "center | absolute | both [default %default]")
  )))
  TF        <- .opt$tf
  CELL_LINE <- .opt$cell_line
  ROOT      <- .opt$rootdir
  ALIGNMENT <- .opt$alignment
}
stopifnot(ALIGNMENT %in% c("center", "absolute", "both"))

# Derived AFTER the CLI block: the parquet read below runs at source time.
ATTR_FILE <- file.path(ROOT, "data/attribution_matrices",
                       paste0(TF, "_", CELL_LINE, "_attrs.parquet"))
OUT_ROOT  <- file.path(ROOT, "attr_analyses_output",     # PNGs -> <OUT_ROOT>/<alignment>/
                       paste0(TF, "_", CELL_LINE), "heatmaps")
if (!file.exists(ATTR_FILE)) stop("Input parquet not found: ", ATTR_FILE)

# ---- read the predicted-positive subset directly into memory ---------------
# Only the small columns + pos_attrs are read (never the giant raw `attributions`),
# and the probability filter is pushed down so only positive windows are materialized.
DAT <- open_dataset(ATTR_FILE) |>
  filter(probabilities >= PROB_MIN) |>
  select(chr_name, start, end, probabilities, pos_attrs) |>
  collect() |>
  as.data.table()
DAT[, center := (start + end) %/% 2L]
DAT[, prob := probabilities]
setorder(DAT, chr_name, start)          # genomic order (5' -> 3') within each chromosome
DAT[, row := rowid(chr_name) - 1L]      # 0-based row index within chromosome

# global color caps over ALL positive per-position values (quantiles + max)
allvals <- unlist(DAT$pos_attrs, use.names = FALSE)
gv <- c(cap_p99  = quantile(allvals, 0.99,  names = FALSE),
        cap_p995 = quantile(allvals, 0.995, names = FALSE),
        cap_p999 = quantile(allvals, 0.999, names = FALSE),
        global_max = max(allvals))
rm(allvals)
CAP <- if (suppressWarnings(!is.na(as.numeric(CAP_KEY)))) as.numeric(CAP_KEY) else as.numeric(gv[[CAP_KEY]])

# per-chromosome index (row counts + genomic extent) and per-row metadata
idx  <- DAT[, .(n_rows = .N, min_center = min(center), max_center = max(center)), by = chr_name]
meta <- DAT[, .(chr_name, row, start, end, center, prob)]
setkey(meta, chr_name, row)

cat(sprintf("ALIGNMENT=%s  NCOL=%d  color cap=%.4g (%s)  chroms=%d  positives=%d\n",
            ALIGNMENT, NCOL, CAP, CAP_KEY, nrow(idx), nrow(DAT)))

# ---- helpers ---------------------------------------------------------------

# Build one chromosome's matrix (n_rows x NCOL) from the in-memory table,
# rows in genomic order (5' -> 3').
read_chrom_matrix <- function(chr, n_rows) {
  do.call(rbind, DAT[chr_name == chr][order(row)]$pos_attrs)
}

# Map a value matrix -> matrix of hex colors (NA -> NA_COLOR), capped & scaled 0..1.
colorize <- function(M, cap, pal, na_col) {
  n <- length(pal)
  scaled <- pmin(M, cap) / cap                 # 0..1
  ci <- as.integer(scaled * (n - 1)) + 1L      # 1..n
  ci[ci < 1L] <- 1L; ci[ci > n] <- n
  cols <- pal[ci]
  cols[is.na(M)] <- na_col
  matrix(cols, nrow = nrow(M), ncol = ncol(M))
}

# Draw a color matrix as a raster heatmap with axes + colorbar.
# colmat row 1 -> top of plot (region ordered 5'->3').
draw_heatmap <- function(colmat, file, xlim, xlab, ylab, main,
                         xticks = NULL, xticklab = NULL,
                         yticks = NULL, yticklab = NULL, cap, pal) {
  n_rows <- nrow(colmat)
  wpx <- 1500
  hpx <- max(700, min(4200, round(n_rows * 0.9) + 160))
  png(file, width = wpx, height = hpx, res = DPI)
  on.exit(dev.off())
  layout(matrix(c(1, 2), nrow = 1), widths = c(wpx - 150, 150))

  # main panel
  par(mar = c(4.5, 8, 3.5, 1))
  plot(NA, xlim = xlim, ylim = c(n_rows + 0.5, 0.5),    # row 1 at top
       xlab = xlab, ylab = "", main = main,
       xaxs = "i", yaxs = "i", axes = FALSE)
  rasterImage(as.raster(colmat), xlim[1], n_rows + 0.5, xlim[2], 0.5,
              interpolate = FALSE)
  if (is.null(xticks)) axis(1) else axis(1, at = xticks, labels = xticklab)
  if (is.null(yticks)) axis(2, las = 1) else axis(2, at = yticks, labels = yticklab, las = 1)
  mtext(ylab, side = 2, line = 6)                        # offset from wide Mb tick labels
  box()

  # colorbar
  par(mar = c(4.5, 0.5, 3.5, 4))
  zb <- seq(0, cap, length.out = length(pal) + 1)
  plot(NA, xlim = c(0, 1), ylim = c(0, cap), axes = FALSE, xlab = "", ylab = "",
       xaxs = "i", yaxs = "i")
  rasterImage(as.raster(matrix(rev(pal), ncol = 1)), 0, 0, 1, cap, interpolate = FALSE)
  axis(4, las = 1)
  mtext("pos_attrs", side = 4, line = 2.6, cex = 0.9)
  box()
}

# ---- per-mode renderers ----------------------------------------------------

render_center <- function(chr, n_rows, outdir) {
  M  <- read_chrom_matrix(chr, n_rows)
  off <- (seq_len(NCOL) - 1) - (NCOL / 2)      # -2048..2047
  cm <- colorize(M, CAP, PALETTE, NA_COLOR)
  # y ticks: genomic coordinate (Mb) of region centers at evenly spaced ranks
  m  <- meta[chr_name == chr][order(row)]
  yr <- unique(round(seq(1, n_rows, length.out = min(10, n_rows))))
  draw_heatmap(cm, file.path(outdir, paste0(chr, "_center.png")),
               xlim = c(min(off), max(off)),
               xlab = "position relative to window center (bp)",
               ylab = "region (genomic order, 5' -> 3')",
               main = sprintf("%s  -  center-aligned pos_attrs  (n=%d predicted-positive)", chr, n_rows),
               yticks = yr, yticklab = sprintf("%.1f Mb", m$center[yr] / 1e6),
               cap = CAP, pal = PALETTE)
}

# Scatter each region's 4096-bp profile into ABS_NCOL genome bins along the chromosome.
# Returns the binned matrix (NA off-window) plus the genomic frame. Shared by base-R
# and ComplexHeatmap absolute renderers.
build_abs_matrix <- function(M, center) {
  n_rows <- nrow(M)
  half   <- NCOL %/% 2
  gmin <- min(center) - half
  gmax <- max(center) + half
  binw <- (gmax - gmin) / ABS_NCOL

  vals <- as.vector(t(M))                                   # row-major: region then position
  rowi <- rep.int(seq_len(n_rows), rep.int(NCOL, n_rows))
  coord <- rep(center, each = NCOL) + (rep(seq_len(NCOL), n_rows) - 1 - half)
  binj <- as.integer((coord - gmin) %/% binw) + 1L
  binj[binj < 1L] <- 1L; binj[binj > ABS_NCOL] <- ABS_NCOL

  dt <- data.table(r = rowi, c = binj, v = vals)
  agg <- if (ABS_AGG == "mean") dt[, .(v = mean(v)), by = .(r, c)]
         else                   dt[, .(v = max(v)),  by = .(r, c)]

  Mabs <- matrix(NA_real_, nrow = n_rows, ncol = ABS_NCOL)  # NA = off-window
  Mabs[cbind(agg$r, agg$c)] <- agg$v
  list(mat = Mabs, gmin = gmin, gmax = gmax, binw = binw)
}

render_absolute <- function(chr, n_rows, outdir) {
  M <- read_chrom_matrix(chr, n_rows)
  m <- meta[chr_name == chr][order(row)]
  ab <- build_abs_matrix(M, m$center)
  Mabs <- ab$mat

  cm <- colorize(Mabs, CAP, PALETTE, NA_COLOR)
  xt <- pretty(c(ab$gmin, ab$gmax) / 1e6)
  xt <- xt[xt * 1e6 >= ab$gmin & xt * 1e6 <= ab$gmax]
  draw_heatmap(cm, file.path(outdir, paste0(chr, "_absolute.png")),
               xlim = c(1, ABS_NCOL),
               xlab = sprintf("chromosome coordinate (Mb)  [bin = %.0f kb]", ab$binw / 1e3),
               ylab = "region (genomic order, 5' -> 3')",
               main = sprintf("%s  -  absolute-coordinate pos_attrs  (n=%d predicted-positive)", chr, n_rows),
               xticks = (xt * 1e6 - ab$gmin) / ab$binw + 1,
               xticklab = sprintf("%g", xt),
               cap = CAP, pal = PALETTE)
}

# ---- ComplexHeatmap builders (shared by static PNGs + interactive Shiny app) ------

# Build a ComplexHeatmap `Heatmap` object for one chromosome.
#   mode == "center"   : rows ("samples") clustered with CLUSTER_METHOD (Ward.D2),
#                        columns = bp offset from window center (fixed order).
#   mode == "absolute" : rows in genomic order (no clustering), columns = genome bins.
# Row/column names carry genomic identity so interactive hover + sub-heatmaps are labelled.
make_ht <- function(chr, mode) {
  n_rows <- as.integer(idx[chr_name == chr, n_rows][1])
  M <- read_chrom_matrix(chr, n_rows)
  m <- meta[chr_name == chr][order(row)]
  col_fun <- colorRamp2(seq(0, CAP, length.out = length(PALETTE)), PALETTE)
  row_ids <- sprintf("%s:%d (p=%.2f)", chr, m$center, m$prob)

  if (mode == "center") {
    off <- (seq_len(NCOL) - 1) - (NCOL %/% 2)              # -2048..2047
    dimnames(M) <- list(row_ids, as.character(off))
    mark_off <- c(-2000, -1000, 0, 1000, 2000)
    mark_at  <- match(mark_off, off)
    bot <- HeatmapAnnotation(
      "bp" = anno_mark(at = mark_at, labels = sprintf("%+d", off[mark_at]),
                       which = "column", side = "bottom", labels_gp = gpar(fontsize = 9)),
      which = "column", annotation_label = "offset (bp)")
    Heatmap(M, name = "pos_attrs", col = col_fun,
            cluster_rows = TRUE, clustering_distance_rows = CLUSTER_DIST,
            clustering_method_rows = CLUSTER_METHOD,
            show_row_dend = TRUE, row_dend_width = unit(16, "mm"),
            cluster_columns = FALSE,
            show_row_names = FALSE, show_column_names = FALSE,
            bottom_annotation = bot,
            column_title = sprintf("%s — center-aligned pos_attrs  (%s row clustering, n=%d)",
                                   chr, CLUSTER_METHOD, n_rows),
            row_title = sprintf("regions / samples  (%s clusters)", CLUSTER_METHOD),
            use_raster = TRUE, raster_device = "png", raster_quality = 2,
            heatmap_legend_param = list(title = "pos_attrs"))
  } else {
    ab <- build_abs_matrix(M, m$center)
    Mabs <- ab$mat
    coord_mb <- (ab$gmin + (seq_len(ABS_NCOL) - 0.5) * ab$binw) / 1e6
    dimnames(Mabs) <- list(row_ids, sprintf("%.2f Mb", coord_mb))
    xt <- pretty(c(ab$gmin, ab$gmax) / 1e6)
    xt <- xt[xt * 1e6 >= ab$gmin & xt * 1e6 <= ab$gmax]
    mark_at <- as.integer((xt * 1e6 - ab$gmin) / ab$binw) + 1L
    bot <- HeatmapAnnotation(
      "Mb" = anno_mark(at = mark_at, labels = sprintf("%g", xt),
                       which = "column", side = "bottom", labels_gp = gpar(fontsize = 9)),
      which = "column", annotation_label = "coord (Mb)")
    Heatmap(Mabs, name = "pos_attrs", col = col_fun, na_col = NA_COLOR,
            cluster_rows = FALSE, cluster_columns = FALSE,
            show_row_names = FALSE, show_column_names = FALSE,
            bottom_annotation = bot,
            column_title = sprintf("%s — absolute-coordinate pos_attrs  (genomic order, bin=%.0f kb, n=%d)",
                                   chr, ab$binw / 1e3, n_rows),
            row_title = "regions / samples  (genomic order, 5' -> 3')",
            use_raster = TRUE, raster_device = "png", raster_quality = 2,
            heatmap_legend_param = list(title = "pos_attrs"))
  }
}

# Render a ComplexHeatmap object to a static PNG (counterpart of the interactive view).
render_ch_png <- function(chr, mode, outdir) {
  n_rows <- as.integer(idx[chr_name == chr, n_rows][1])
  ht <- make_ht(chr, mode)
  hpx <- max(800, min(5200, round(n_rows * 1.0) + 320))
  png(file.path(outdir, sprintf("%s_%s_ch.png", chr, mode)),
      width = 1700, height = hpx, res = DPI)
  on.exit(dev.off())
  draw(ht, merge_legend = TRUE)
}

# chromosomes in karyotype order (chr1..22, X, Y); used by the driver and the app
idx <- idx[order(as.integer(sub("chr", "", sub("X", "23", sub("Y", "24", idx$chr_name)))))]

# ---- drive over chromosomes ------------------------------------------------
main <- function() {
  modes <- if (ALIGNMENT == "both") c("center", "absolute") else ALIGNMENT
  for (mode in modes) {
    base_dir <- file.path(OUT_ROOT, mode)
    ch_dir   <- file.path(OUT_ROOT, paste0(mode, "_complexheatmap"))
    if (RENDER_BASE_PNG) dir.create(base_dir, recursive = TRUE, showWarnings = FALSE)
    if (RENDER_CH_PNG)   dir.create(ch_dir,   recursive = TRUE, showWarnings = FALSE)
    cat(sprintf("\n== %s mode ==\n", mode))
    for (i in seq_len(nrow(idx))) {
      chr <- idx$chr_name[i]; n_rows <- idx$n_rows[i]
      t0 <- Sys.time()
      if (RENDER_BASE_PNG) {
        if (mode == "center") render_center(chr, n_rows, base_dir)
        else                  render_absolute(chr, n_rows, base_dir)
      }
      if (RENDER_CH_PNG) render_ch_png(chr, mode, ch_dir)   # center => Ward.D2 clustered
      cat(sprintf("  %-6s n=%-5d  %.1fs\n", chr, n_rows,
                  as.numeric(difftime(Sys.time(), t0, units = "secs"))))
    }
  }
  cat("\nStatic PNGs done.")
  cat("\nInteractive heatmaps:  Rscript -e 'shiny::runApp(\"heatmap_app.R\", launch.browser=TRUE)'\n")
}

# Run only when executed directly; when sourced as a library (e.g. by heatmap_app.R,
# which sets options(genome_heatmaps.lib = TRUE)) just expose the functions above.
if (!isTRUE(getOption("genome_heatmaps.lib", FALSE))) main()
