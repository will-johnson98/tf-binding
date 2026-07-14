#!/usr/bin/env Rscript
# heatmap_app.R -- WAJ 2026-06-29  (HPC rewrite 2026-07-02)
# Interactive browser for the genome-aligned pos_attrs heatmaps, built on
# InteractiveComplexHeatmap. Reuses make_ht() from heatmaps2_genome_heatmaps.R so the
# interactive view is identical to the static ComplexHeatmap PNGs.
#
# Launch:  Rscript -e 'shiny::runApp("heatmaps1a_app.R", launch.browser = TRUE)'
#
# The renderer is sourced as a library (options(genome_heatmaps.lib = TRUE)); sourcing it
# reads the predicted-positive subset of the attrs parquet into memory (arrow) and defines
# config + make_ht without running the batch renderer. There is no separate prep step.
#
# Features: hover a cell for its pos_attrs value + region id + position; brush a
# rectangle to open a labelled sub-heatmap; search; export. Center-aligned mode
# clusters the regions ("samples") with Ward.D2; absolute mode keeps genomic order.

suppressMessages({
  library(shiny)
  library(ComplexHeatmap)
  library(InteractiveComplexHeatmap)
})

# Source the renderer as a library (defines config + make_ht + in-memory data, no main()).
options(genome_heatmaps.lib = TRUE)
source("heatmaps2_genome_heatmaps.R", chdir = TRUE)

CHROMS <- idx$chr_name          # karyotype order, set in heatmaps2_genome_heatmaps.R
MODES  <- c("center", "absolute")

# Cache drawn heatmaps so re-selecting a chromosome is instant (clustering is the
# expensive step and happens at draw() time). pdf(NULL) draws to a null device.
.ht_cache <- new.env(parent = emptyenv())
get_drawn_ht <- function(chr, mode) {
  key <- paste(chr, mode, sep = "/")
  if (is.null(.ht_cache[[key]])) {
    pdf(NULL); on.exit(dev.off())
    .ht_cache[[key]] <- draw(make_ht(chr, mode), merge_legend = TRUE)
  }
  .ht_cache[[key]]
}

ui <- fluidPage(
  tags$h3(sprintf("%s %s — interactive genome-aligned pos_attrs heatmaps",
                  TRANSCRIPTION_FACTOR, CELL_LINE)),
  sidebarLayout(
    sidebarPanel(
      width = 3,
      selectInput("chr",  "Chromosome", choices = CHROMS, selected = CHROMS[1]),
      radioButtons("mode", "Alignment",
                   choices = c("center-aligned (Ward.D2 clustered)" = "center",
                               "absolute coordinate (genomic order)" = "absolute"),
                   selected = "center"),
      actionButton("go", "Render", class = "btn-primary"),
      tags$hr(),
      tags$small(HTML(
        "Hover a cell for value + region id.<br>",
        "Drag a box to open a labelled sub-heatmap.<br><br>",
        "<b>center</b>: rows = regions clustered by Ward.D2 on their attribution profile.<br>",
        "<b>absolute</b>: rows in genomic order; cells binned to chromosome coordinates."))
    ),
    mainPanel(
      width = 9,
      InteractiveComplexHeatmapOutput(heatmap_id = "ht",
                                      width1 = 900, height1 = 650)
    )
  )
)

server <- function(input, output, session) {
  render_sel <- function() {
    withProgress(message = sprintf("Building %s / %s ...", input$chr, input$mode), {
      ht <- get_drawn_ht(input$chr, input$mode)
      makeInteractiveComplexHeatmap(input, output, session, ht, heatmap_id = "ht")
    })
  }
  # ignoreNULL = FALSE fires once at startup, then on every click.
  observeEvent(input$go, render_sel(), ignoreNULL = FALSE)
}

shinyApp(ui, server)
