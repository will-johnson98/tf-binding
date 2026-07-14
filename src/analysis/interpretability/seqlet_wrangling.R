#!/usr/bin/env Rscript --vanilla
# WAJ 2026-04-28

library(tidyverse)
library(Biostrings)

root_dir <- "/data1/home/wjohnson/interpretability"
setwd(root_dir)

model <- "ASCL1"
sample <- "LuCaP-49"

tb <- read_csv(paste0("output/", model, "_", sample, "/", "positive_seqlets.csv")) |>
  filter(attribution > 0) |>
  mutate(center = round((start + end) /  2))


max_attrs <- tb |>
  group_by(example_idx) |>
  summarize(max_attr = max(attribution)) |>
  filter(max_attr > 2)
tb <- filter(tb, example_idx %in% max_attrs$example_idx)
stopifnot(n_distinct(tb$example_idx) == n_distinct(max_attrs$example_idx))


mult_peaks <- tb |>
  group_by(example_idx) |>
  summarize(n = n()) |>
  filter(n > 1)
mult_peaks <- unique(mult_peaks$example_idx)
tb <- filter(tb, example_idx %in% mult_peaks)


new_tp <- lapply(mult_peaks, function(sample) {
  tmp <- filter(tb, example_idx == sample) |>
    arrange(desc(attribution))
  max_attr <- tmp[1, ]$attribution
  tmp <- filter(tmp, attribution >= (max_attr * 0.4))
  if (nrow(tmp) > 1) {
    max_center <- tmp[1, ]$center
    tmp <- mutate(tmp, distance = abs(center - max_center)) |>
      mutate(pct_max = attribution / max_attr) |>
      filter(distance <= 100) |>
      filter(distance > 10 | distance == 0)
    if (nrow(tmp) > 1) {
      tmp
    }
  }
}) |>
  bind_rows() |>
  mutate(is_max = ifelse(distance == 0, TRUE, FALSE)) |>
  add_count(example_idx, name = "n_peaks_in_sample")


ggplot(new_tp, aes(x = center, 
                   y = attribution,
                   color = is_max)) +
  geom_point(alpha = 1/2) + 
  theme_bw()


ggplot(new_tp, aes(x = nchar(sequence), 
                   y = attribution,
                   color = is_max)) +
  geom_point(alpha = 1/2) + 
  theme_bw()


ggplot(new_tp, aes(x = as.factor(n_peaks_in_sample), 
                   y = attribution,
                   fill = as.factor(n_peaks_in_sample))) + 
  geom_boxplot() + 
  theme_bw()

count(new_tp, n_peaks_in_sample) |> mutate(n_samples = n / n_peaks_in_sample)

top_samples <- new_tp |>
  filter(is_max == TRUE & n_peaks_in_sample < 5) |>
  group_by(n_peaks_in_sample) |>
  slice_max(attribution, n = 3)
top_samples <- unique(top_samples$example_idx)
top_samples <- append(top_samples, unique(filter(new_tp, n_peaks_in_sample == 5)$example_idx))

writeLines(as.character(top_samples), con = file("Downloads/top_samples.txt"), sep = "\n")

ready_tp <- filter(new_tp, example_idx %in% top_samples)

write_csv(ready_tp, file = "Downloads/filtered_true_positive_seqlets_with_distance.csv")


seqs <- unique(new_tp$sequence)
names(seqs) <- paste0("seq_", seq_along(seqs))
# names(seqs) <- lapply(seqs, function(i) {
#   paste("seq", unique(filter(new_tp, sequence == i)$example_idx), 
#         sep = "_", 
#         collapse = "_")
# })
dna <- DNAStringSet(seqs)
writeXStringSet(dna, filepath = "Downloads/filtered_tp_seqs.fasta")

