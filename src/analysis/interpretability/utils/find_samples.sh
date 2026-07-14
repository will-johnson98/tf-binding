#!/usr/bin/env bash
# WAJ 2026-05-06

MODEL=$1

PROJECT_DIR="/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding"
PARQUET_DIR="${PROJECT_DIR}/data/processed_results"
GROUND_TRUTH_DIR="${PROJECT_DIR}/data/transcription_factors/${MODEL}/merged"
JASPAR_DIR="${PROJECT_DIR}/src/analysis/interpretability/motifs"

parquet_ids=$(fd -u -C $PARQUET_DIR -e parquet "${MODEL}_" | xargs basename -a -s ".parquet" | sed "s/${MODEL}_//g" | sed "s/_processed//g" | sed 's/_/-/g' | sort -u) 
echo "Parquet IDs:"; echo "$parquet_ids"; echo

bed_ids=$(fd -u -C $GROUND_TRUTH_DIR -e bed $MODEL | xargs basename -a | sed "s/_${MODEL}_merged.bed//g" | sed 's/_/-/g' | sort -u)
echo "BED IDs:"; echo "$bed_ids"; echo

echo "In both:"
comm -12 <(echo "${parquet_ids^^}") <(echo "${bed_ids^^}")
echo