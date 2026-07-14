#!/usr/bin/env python3

# Unified interpretability pipeline for seqlet calling. 
# Synthesized from plot_with_atac.py (preferred for shared steps) and
# call_seqlets.py (PWM scoring / seqlet export / external tools).

import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl
import pysam
import torch
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('whitegrid')

from matplotlib.colors import TwoSlopeNorm
from tangermeme.plot import plot_logo
from tangermeme.seqlet import recursive_seqlets, tfmodisco_seqlets
from tqdm import tqdm
from pathlib import Path

from src.utils.generate_training_peaks import run_bedtools_command


# ============================================================================
# CONFIG
# ============================================================================

project_path = "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding"
interpretability_path = os.path.join(project_path, "src/analysis/interpretability")

# These are the cell lines I did the inference on using the AR model:
# - [X]VCAP
# - [X] LNCAP
# - [] MCF7 # TODO: find correct files
# - [] C42B # TODO: find correct files
# - [A] C4-2
# - [] A-375

# For AR, In both:
# - [X] 22RV1
# - [] A-375
# - [A] C4-2
# - [] DU145
# - [X] LNCAP
# - [] T-47D
# - [X] VCAP

# Model / sample
model = "AR"
sample = "A-375"

if model == "ASCL1":
    long_model_name = "ASCL1-LU49-PDXs"
else:
    long_model_name = model


# Pipeline stage toggles. Dependencies are auto-resolved (e.g. plotting
# implicitly turns on data + seqlet stages, and seqlets are loaded from cache
# if a CSV already exists).

RUN_PROCESS_DATA   = True   # parquet -> ground-truth join -> attributions + ATAC pileups
RUN_CALL_SEQLETS   = True   # seqlet extraction (cached to seqlets_path)
RUN_PLOT_SEQLETS   = True   # per-seqlet DNA logo + ATAC pileup/heatmap PDFs
RUN_SAVE_SEQLETS   = True   # split pos/neg seqlets to CSV + write positive FASTA
RUN_PWM_SCORING    = True   # IUPAC-Levenshtein score against JASPAR PWM
RUN_EXTERNAL_TOOLS = True   # levenshtein.py + posthoc.R

# Method options
seqlet_method = 'recursive'        # 'recursive' | 'tfmodisco'
ground_truth_subset = "TP"         # 'TP' | 'TN' | 'FP' | 'FN'
n_top_samples_to_plot = 20
plot_context_size = 200

# Constants used by threshold_peaks / data limits
HIGH_COUNT_QUANTILE = 0.75
MAX_COUNT_THRESHOLD = 30
MID_COUNT_THRESHOLD = 10

# External-tool config
PWM_MIN_SEQLET = 5  # passed to posthoc.R

# ----------------------------------------------------------------------------
# Derived paths (shouldn't usually need edits)
# ----------------------------------------------------------------------------
today = datetime.now().strftime("%Y%m%d")

seqlet_export_dir = os.path.join(
    interpretability_path,
    f"output/{today}_{model}_{sample}",
)

plots_dir = f"{seqlet_export_dir}/{today}_{model}_{sample}_{ground_truth_subset}_{plot_context_size}bp"

seqlets_path = os.path.join(
    seqlet_export_dir, f"{model}_{sample.upper()}_{ground_truth_subset}_seqlets.csv",
)

parquet_file = f"{project_path}/data/processed_results/{long_model_name}_{sample}_processed.parquet"


gt_combined_path = os.path.join(
    seqlet_export_dir, f"{model}_{sample.upper()}_{ground_truth_subset}_gt.parquet",
)

# ====================================================================================
# ------------------------------------------------------------------------------------
# chip_gt_file = f"/data1/datasets_1/human_prostate_PDX/processed/external_data/ChIP_atlas/{model}/SRX8406455.05.bed"
# atac_gt_file = f"/data1/datasets_1/human_prostate_PDX/processed/ATAC/{sample}/SRR12455441/peaks/SRR12455441.filtered.broadPeak"
# ------------------------------------------------------------------------------------
# ====================================================================================

if sample.startswith("LuCaP"):
    LUCAP = True
    smp_chip = f"{sample}_{model}"
    lcm_df = pd.read_csv("LuCaP_ChIP_matching.txt", sep="\t").rename(columns={"PDX ": "PDX"})
    lcm_df = lcm_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    lcm_dict = dict(zip(lcm_df['PDX_ChIP'], lcm_df['Experiment']))
    pdx_dir = f"/data1/datasets_1/human_prostate_PDX/processed/external_data/ChIP_atlas/{model}"
    ground_truth_file = str(list(Path(pdx_dir).glob(f"{lcm_dict[smp_chip]}*.bed"))[0])
else:
    LUCAP = False
    ground_truth_file = f"{project_path}/data/transcription_factors/{model}/merged/{sample.upper()}_{model}_merged.bed"
print(ground_truth_file)

jaspar_file = f"{interpretability_path}/motifs/{model}.jaspar"
levenshtein_script = f"{interpretability_path}/levenshtein.py"
posthoc_script = f"{interpretability_path}/posthoc.R"

# ============================================================================


# ----------------------------------------------------------------------------
# Data assembly (ground truth join, thresholding, attribution + ATAC loading)
# ----------------------------------------------------------------------------

def intersect_bed_files(main_df: pl.LazyFrame,
                        intersect_df: pl.DataFrame,
                        region_type: str = None) -> pl.LazyFrame:
    """Intersect two BED frames via bedtools and tag overlap on the main frame."""
    _schema = main_df.collect_schema()
    main_cols = _schema.keys()

    with tempfile.NamedTemporaryFile(delete=False, mode='w') as main_file, \
         tempfile.NamedTemporaryFile(delete=False, mode='w') as intersect_file, \
         tempfile.NamedTemporaryFile(delete=False, mode='w') as result_file:
        main_path = main_file.name
        intersect_path = intersect_file.name
        result_path = result_file.name

        main_df.collect().write_csv(main_path, separator="\t", include_header=False)
        intersect_df.write_csv(intersect_path, separator="\t", include_header=False)

        run_bedtools_command(
            f"bedtools intersect -a {main_path} -b {intersect_path} -c > {result_path}"
        )

        result_df = pl.read_csv(
            result_path,
            separator="\t",
            has_header=False,
            new_columns=[*main_cols, "overlap_count"],
        ).lazy()

        os.remove(main_path)
        os.remove(intersect_path)
        os.remove(result_path)

        return result_df.with_columns(
            pl.col("overlap_count").gt(0).alias("overlaps_ground_truth")
        ).drop("overlap_count")


def threshold_peaks(df):
    """Adaptive count filter: stricter when the max count is high."""
    def get_scalar(expr):
        if isinstance(df, pl.LazyFrame):
            return expr.collect().item()
        return expr.item()

    max_count = get_scalar(df.select(pl.col("count").max()))

    if max_count <= 2:
        return df
    elif max_count > MAX_COUNT_THRESHOLD:
        threshold = get_scalar(df.select(pl.col("count").quantile(HIGH_COUNT_QUANTILE)))
        return df.filter(pl.col("count") > threshold)
    elif max_count > MID_COUNT_THRESHOLD:
        threshold = get_scalar(df.select(pl.col("count").median()))
        return df.filter(pl.col("count") > threshold)

    return df


def reshape_attributions_fast(df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized attribution reshape -> (DNA attrs n,4,4096), (ATAC attrs n,4096)."""
    print("Reshaping attribution data...")
    attributions = np.array(df['attributions'].to_list())
    reshaped = attributions.reshape(-1, 4096, 5)
    attrs_list = reshaped[..., :4].transpose(0, 2, 1)
    atac_attribution_list = reshaped[..., 4]
    return attrs_list, atac_attribution_list


def process_pileups_batch(pileup_dir: Path,
                          regions_df: pl.DataFrame) -> Dict[int, np.ndarray]:
    """Read pileups for many regions of one cell line, opening each tabix file once."""
    context_length = 4_096
    chromosomes = regions_df['chr'].unique().to_list()
    chr_tabix_files = {}

    print(f"Opening tabix files for {len(chromosomes)} chromosomes...")
    for chr_name in tqdm(chromosomes, desc="Loading chromosome files", leave=False):
        pileup_file = pileup_dir / f"{chr_name}.pileup.gz"
        if pileup_file.exists():
            chr_tabix_files[chr_name] = pysam.TabixFile(str(pileup_file))

    atac_arrays = {}
    total_regions = len(regions_df)
    for row in tqdm(regions_df.iter_rows(named=True), total=total_regions,
                    desc="Processing regions", leave=False):
        idx, chr_name, start, end = row['idx'], row['chr'], row['start'], row['end']

        interval_length = end - start
        extra_seq = context_length - interval_length
        extra_left_seq = extra_seq // 2
        extra_right_seq = extra_seq - extra_left_seq
        adj_start = start - extra_left_seq
        adj_end = end + extra_right_seq

        atac_array = np.zeros(context_length)

        if chr_name in chr_tabix_files:
            tabixfile = chr_tabix_files[chr_name]
            positions, counts = [], []
            try:
                for rec in tabixfile.fetch(chr_name, adj_start, adj_end):
                    fields = rec.split("\t")
                    positions.append(int(fields[1]))
                    counts.append(float(fields[3]))

                if positions:
                    positions = np.array(positions)
                    counts = np.array(counts)
                    array_indices = positions - adj_start
                    valid_mask = (array_indices >= 0) & (array_indices < context_length)
                    atac_array[array_indices[valid_mask]] = counts[valid_mask]
            except Exception as e:
                print(f"Warning: Could not fetch data for {chr_name}:{adj_start}-{adj_end}: {e}")

        atac_arrays[idx] = atac_array

    print("Closing tabix files...")
    for tabixfile in chr_tabix_files.values():
        tabixfile.close()

    return atac_arrays


def process_region_data_fast(df: pl.DataFrame,
                             base_pileup_dir: Path = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reshape attributions and pull raw ATAC pileups in one pass over cell-line groups."""
    if base_pileup_dir is None:
        base_pileup_dir = Path("/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines/")

    print(f"Processing {len(df)} regions across cell lines...")
    attrs_list, atac_attribution_list = reshape_attributions_fast(df)

    df_with_idx = df.with_row_index("idx")
    atac_pileup_arrays = [None] * len(df)
    cell_line_groups = list(df_with_idx.group_by("cell_line"))

    print(f"Processing {len(cell_line_groups)} cell lines...")
    for cell_line, group_df in tqdm(cell_line_groups, desc="Processing cell lines"):
        cell_line_name = cell_line[0]
        pileup_dir = base_pileup_dir / cell_line_name / "pileup_mod"

        if not pileup_dir.exists():
            print(f"Warning: Pileup directory does not exist: {pileup_dir}")
            for row in group_df.iter_rows(named=True):
                atac_pileup_arrays[row['idx']] = np.zeros(4096)
            continue

        print(f"Processing {len(group_df)} regions for cell line: {cell_line_name}")
        atac_arrays_dict = process_pileups_batch(pileup_dir, group_df)
        for idx, atac_array in atac_arrays_dict.items():
            atac_pileup_arrays[idx] = atac_array

    print("Converting to final numpy arrays...")
    atac_pileup_list = np.array(atac_pileup_arrays)
    print("Processing complete!")
    return attrs_list, atac_attribution_list, atac_pileup_list


# ----------------------------------------------------------------------------
# Seqlet calling + plotting
# ----------------------------------------------------------------------------

def get_seqlets(attrs_list,
                use_absolute_values=False,
                method='recursive',
                direction='positive',
                **kwargs):
    """
    Extract seqlets via tangermeme. method='recursive' | 'tfmodisco';
    direction='positive' | 'negative' (flips sign before calling).
    """
    attrs_array = np.stack(attrs_list, axis=0)
    summed_attrs = attrs_array.sum(axis=1)

    if direction == 'negative':
        summed_attrs = -summed_attrs
    elif direction != 'positive':
        raise ValueError("direction must be 'positive' or 'negative'")

    if use_absolute_values:
        summed_attrs = np.abs(summed_attrs)

    if method == 'tfmodisco':
        summed_attrs_tensor = torch.from_numpy(summed_attrs).float()
        seqlets = tfmodisco_seqlets(summed_attrs_tensor)
    elif method == 'recursive':
        r_kwargs = {'threshold': 0.01, 'min_seqlet_len': 5,
                    'max_seqlet_len': 25, 'additional_flanks': 2}
        r_kwargs.update(kwargs)
        seqlets = recursive_seqlets(summed_attrs, **r_kwargs)
    else:
        raise ValueError(f"Unknown seqlet calling method: '{method}'.")

    nt_idx = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    sequences = []
    for i in range(len(seqlets)):
        s = seqlets.iloc[i]
        start, end, sample_idx = int(s['start']), int(s['end']), int(s['example_idx'])
        sample_attrs = attrs_array[sample_idx, :, start:end].T.squeeze()
        hits = np.argmax(sample_attrs, axis=1)
        sequences.append(''.join(nt_idx[i] for i in hits))
    seqlets['sequence'] = sequences
    return seqlets


def plot_seqlet_with_atac(seqlets,
                          attrs_list,
                          atac_attribution_list,
                          atac_pileup_list,
                          sample_rank=0,
                          context_size=20,
                          colormap='RdBu_r',
                          equal_color_scale=False,
                          fname=None):
    """Two-panel plot: DNA-base logo above, ATAC pileup with attribution heatmap below.

    The ATAC heatmap uses TwoSlopeNorm centered at 0 with the actual data
    bounds, so neutrality stays white even when min/max are asymmetric.
    """
    sample = seqlets.iloc[[sample_rank]]
    slice_idx = int(sample['example_idx'].tolist()[0])
    sequence = sample['sequence'].tolist()[0]
    start = int(sample['start'].tolist()[0])
    end = int(sample['end'].tolist()[0])

    seqlet_center = (start + end) // 2
    seqlet_length = end - start
    total_window_size = seqlet_length + (2 * context_size)
    window_start = max(0, seqlet_center - (total_window_size // 2))
    window_end = min(4096, seqlet_center + (total_window_size // 2))
    if window_end - window_start < total_window_size:
        if window_start == 0:
            window_end = min(4096, window_start + total_window_size)
        elif window_end == 4096:
            window_start = max(0, window_end - total_window_size)

    print(f"Seqlet: {start}-{end} (center: {seqlet_center})")
    print(f"Window: {window_start}-{window_end} (size: {window_end - window_start})")
    if 'p-value' in sample.columns:
        print(f"P-Value: {sample['p-value'].tolist()[0]}")

    plot_coords = np.arange(window_start, window_end)
    X_attr = attrs_list[slice_idx].astype(np.float64)
    atac_attr = atac_attribution_list[slice_idx].astype(np.float64)
    atac_pileup = atac_pileup_list[slice_idx].astype(np.float64)
    X_attr_windowed = X_attr[:, window_start:window_end]
    atac_attr_windowed = atac_attr[window_start:window_end]
    atac_pileup_windowed = atac_pileup[window_start:window_end]

    fig = plt.figure(figsize=(18, 10), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[20, 1], height_ratios=[1, 1],
                          hspace=0.3, wspace=0.02)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])

    plot_logo(X_attr_windowed, ax=ax1)
    n_ticks = 8
    tick_positions = np.linspace(0, len(plot_coords) - 1, n_ticks)
    tick_labels = np.linspace(plot_coords[0], plot_coords[-1], n_ticks).astype(int)
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)
    ax1.set_xlabel("Genomic Coordinate")
    ax1.set_ylabel("DNA Attributions")
    ax1.set_title(f"DNA Base Attributions | Sample: {slice_idx} | {sequence}")

    heatmap_height = 25
    attr_heatmap = np.tile(atac_attr_windowed, (heatmap_height, 1))
    max_pileup = np.max(atac_pileup_windowed) if len(atac_pileup_windowed) > 0 else 1
    y_max = max_pileup * 1.1

    if atac_attr_windowed.size > 0:
        vmin_val = float(np.min(atac_attr_windowed))
        vmax_val = float(np.max(atac_attr_windowed))
    else:
        vmin_val, vmax_val = -1.0, 1.0

    if equal_color_scale:
        vabs_max = float(np.max(np.abs(atac_attr_windowed))) if atac_attr_windowed.size > 0 else 1.0
        vmin_val, vmax_val = -vabs_max, vabs_max

    norm = TwoSlopeNorm(vcenter=0, vmin=vmin_val, vmax=vmax_val)

    im = ax2.imshow(attr_heatmap, cmap=colormap, aspect='auto',
                    extent=[plot_coords[0], plot_coords[-1], 0, y_max],
                    alpha=0.7, interpolation='bilinear', norm=norm)

    ax2.plot(plot_coords, atac_pileup_windowed, color='black', linewidth=2.5,
             label='ATAC-seq Pileup', alpha=0.9)
    ax2.set_xlim(plot_coords[0], plot_coords[-1])

    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('ATAC Attribution', rotation=270, labelpad=15, fontsize=11)

    ax2.set_xlabel("Genomic Coordinate")
    ax2.set_ylabel("ATAC-seq Signal")
    ax2.set_title(f"ATAC Pileup with Attribution Heatmap | Sample: {slice_idx}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', dpi=300)
        print(f"Plot saved to: {fname}")
        plt.close()
    else:
        plt.show()


# ----------------------------------------------------------------------------
# PWM scoring + seqlet export (unique to interpretability.py)
# ----------------------------------------------------------------------------

@dataclass
class PWM:
    """Container for a position weight matrix."""
    name: str
    matrix: np.ndarray
    bases: List[str] = field(default_factory=lambda: ['A', 'C', 'G', 'T'])

    def get_consensus(self, prob_threshold: float = 0.25) -> str:
        iupac_map = {
            'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T',
            'AC': 'M', 'AG': 'R', 'AT': 'W',
            'CG': 'S', 'CT': 'Y', 'GT': 'K',
            'ACG': 'V', 'ACT': 'H', 'AGT': 'D', 'CGT': 'B',
            'ACGT': 'N',
        }
        consensus = []
        for pos_probs in self.matrix.T:
            sig = ''.join(b for b, p in zip(self.bases, pos_probs) if p >= prob_threshold)
            sig = ''.join(sorted(sig))
            consensus.append(iupac_map.get(sig, 'N'))
        return ''.join(consensus)


def parse_jaspar(path: str) -> PWM:
    with open(path) as f:
        lines = f.readlines()
    if not lines or len(lines) != 5:
        raise ValueError("Invalid JASPAR format")
    name = lines[0].split()[0]
    matrix = []
    for line in lines[1:]:
        nums = line.split('[')[1].split(']')[0].strip().split()
        matrix.append([float(x) for x in nums])
    matrix = np.array(matrix)
    matrix = matrix / matrix.sum(axis=0)
    return PWM(name=name, matrix=matrix)


_IUPAC = {
    'A': {'A'}, 'C': {'C'}, 'G': {'G'}, 'T': {'T'},
    'R': {'A', 'G'}, 'Y': {'C', 'T'}, 'S': {'G', 'C'}, 'W': {'A', 'T'},
    'K': {'G', 'T'}, 'M': {'A', 'C'},
    'B': {'C', 'G', 'T'}, 'D': {'A', 'G', 'T'}, 'H': {'A', 'C', 'T'}, 'V': {'A', 'C', 'G'},
    'N': {'A', 'C', 'G', 'T'},
}


def iupac_match(a: str, b: str) -> bool:
    a, b = a.upper(), b.upper()
    if a not in _IUPAC or b not in _IUPAC:
        raise ValueError(f"Invalid IUPAC code: {a if a not in _IUPAC else b}")
    return bool(_IUPAC[a] & _IUPAC[b])


def levenshtein_iupac(seq1: str, seq2: str) -> int:
    if not seq1: return len(seq2)
    if not seq2: return len(seq1)
    previous_row = list(range(len(seq2) + 1))
    current_row = [0] * (len(seq2) + 1)
    for i, c1 in enumerate(seq1):
        current_row[0] = i + 1
        for j, c2 in enumerate(seq2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (not iupac_match(c1, c2))
            current_row[j + 1] = min(insertions, deletions, substitutions)
        previous_row, current_row = current_row, [0] * (len(seq2) + 1)
    return previous_row[-1]


def score_seqlet(pwm: PWM, seq: str) -> Tuple[float, int]:
    """Slide consensus over seq (or vice versa) and return best (norm_score, offset)."""
    seq_len = len(seq)
    pwm_width = pwm.matrix.shape[1]
    consensus = pwm.get_consensus()

    if seq_len < pwm_width:
        max_score, best_pos = float('-inf'), 0
        for i in range(pwm_width - seq_len + 1):
            cons_slice = consensus[i:i + seq_len]
            raw_dist = levenshtein_iupac(seq, cons_slice)
            norm_score = 1 - (raw_dist / max(len(seq), len(cons_slice)))
            if norm_score > max_score:
                max_score, best_pos = norm_score, i
        return max_score, best_pos
    elif seq_len == pwm_width:
        raw_dist = levenshtein_iupac(seq, consensus)
        return 1 - (raw_dist / len(consensus)), 0
    else:
        max_score, best_pos = float('-inf'), 0
        for i in range(seq_len - pwm_width + 1):
            raw_dist = levenshtein_iupac(seq[i:i + pwm_width], consensus)
            norm_score = 1 - (raw_dist / len(consensus))
            if norm_score > max_score:
                max_score, best_pos = norm_score, i
        return max_score, best_pos


def write_fasta(sequences, outfile):
    with open(outfile, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f'>seq_{i + 1}\n{seq}\n')


def save_seqlets(seqlets: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    pos = seqlets[seqlets['attribution'] > 0].reset_index(drop=True)
    neg = seqlets[seqlets['attribution'] < 0].reset_index(drop=True)
    pos.to_csv(os.path.join(output_dir, "positive_seqlets.csv"), index=False)
    neg.to_csv(os.path.join(output_dir, "negative_seqlets.csv"), index=False)
    write_fasta(pos['sequence'].tolist(), os.path.join(output_dir, "positive_seqlets.fa"))


def add_genomic_coords(seqlets: pd.DataFrame,
                       ground_truth_df: pl.DataFrame,
                       context_length: int = 4096) -> pd.DataFrame:
    """Add absolute genomic coords (chr, g_start, g_end) to seqlets.

    The seqlet 'start'/'end' are relative to the 4096bp model window, and
    'example_idx' is the row position in `ground_truth_df` -- which MUST be the
    same in-memory frame that built the attribution arrays this run, since the
    streaming collect order in _build_ground_truth_df is not reproducible across
    runs. Mirrors the symmetric padding used to build the window (see
    process_pileups_batch): adj_start = p_start - (context_length - L) // 2.
    """
    chrom = ground_truth_df["chr"].to_numpy()
    g_s = ground_truth_df["start"].to_numpy()
    g_e = ground_truth_df["end"].to_numpy()
    ex = seqlets["example_idx"].to_numpy().astype(int)
    adj_start = g_s[ex] - (context_length - (g_e[ex] - g_s[ex])) // 2
    seqlets = seqlets.copy()
    seqlets["chr"] = chrom[ex]
    seqlets["g_start"] = adj_start + seqlets["start"].to_numpy().astype(int)
    seqlets["g_end"] = adj_start + seqlets["end"].to_numpy().astype(int)
    return seqlets


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def _build_ground_truth_df(lucap=False) -> pl.DataFrame:
    df = pl.read_parquet(
        parquet_file,
        columns=["chr_name", "start", "end", "cell_line",
                 "targets", "predicted", "weights",
                 "probabilities", "attributions"],
        parallel="columns",
        use_statistics=True,
        memory_map=True,
    ).lazy()
    df = df.rename({"chr_name": "chr"})

    if lucap:
         df_ground_truth = pl.read_csv(
            ground_truth_file,
            separator="\t",
            has_header=False,
            new_columns=["chr", "start", "end", "id", "count"],
            columns=[0, 1, 2, 3, 4],
            )
    else:
        df_ground_truth = pl.read_csv(
            ground_truth_file,
            separator="\t",
            has_header=False,
            new_columns=["chr", "start", "end", "count"],
            columns=[0, 1, 2, 3],
            # new_columns=["chr", "start", "end", "id", "count"],
            # columns=[0, 1, 2, 3, 4],
        )
    df_ground_truth_filtered = threshold_peaks(df_ground_truth)

    intersected_df = intersect_bed_files(df.select(["chr", "start", "end"]),
                                         df_ground_truth_filtered)
    ground_truth_df = df.join(intersected_df, on=["chr", "start", "end"], how="left")
    ground_truth_df = ground_truth_df.with_columns(
        pl.when(pl.col("overlaps_ground_truth")).then(1).otherwise(0).alias("targets")
    )

    ground_truth_df = (
        ground_truth_df.with_columns(
            correct=pl.when((pl.col("targets") == 1) & (pl.col("predicted") == 1)).then(pl.lit("TP"))
                     .when((pl.col("targets") == 0) & (pl.col("predicted") == 0)).then(pl.lit("TN"))
                     .when((pl.col("targets") == 1) & (pl.col("predicted") == 0)).then(pl.lit("FN"))
                     .when((pl.col("targets") == 0) & (pl.col("predicted") == 1)).then(pl.lit("FP"))
        ).collect(streaming=True)
    )
    ground_truth_df = ground_truth_df.filter(pl.col("correct") == ground_truth_subset)
    # ground_truth_df.write_csv(gt_combined_path)
    ground_truth_df.write_parquet(gt_combined_path)

    return ground_truth_df


def main():
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(os.path.dirname(seqlets_path), exist_ok=True)

    seqlets_cached = os.path.isfile(seqlets_path)

    needs_attrs_arrays = RUN_PLOT_SEQLETS or (
        (RUN_CALL_SEQLETS or RUN_PWM_SCORING or RUN_SAVE_SEQLETS) and not seqlets_cached
    )
    needs_data_stage = RUN_PROCESS_DATA or needs_attrs_arrays

    attrs_list = atac_attribution_list = atac_pileup_list = None
    ready_df = None
    if needs_data_stage:
        if LUCAP:
            ready_df = _build_ground_truth_df(lucap=True)
        else:
            ready_df = _build_ground_truth_df()
        attrs_list, atac_attribution_list, atac_pileup_list = process_region_data_fast(ready_df)

    needs_seqlets = RUN_CALL_SEQLETS or RUN_PLOT_SEQLETS or RUN_SAVE_SEQLETS or RUN_PWM_SCORING
    seqlets = None
    seqlets_from_cache = False
    if needs_seqlets:
        if seqlets_cached:
            print(f"Loading cached seqlets: {seqlets_path}")
            seqlets = pd.read_csv(seqlets_path)
            seqlets_from_cache = True
        elif RUN_CALL_SEQLETS:
            seqlets = get_seqlets(attrs_list, method=seqlet_method)
            seqlets.to_csv(seqlets_path, index=False)
            print(f"Wrote seqlets: {seqlets_path}")
        else:
            print(f"No cached seqlets at {seqlets_path}; enable RUN_CALL_SEQLETS to generate them.")
            return

    if RUN_PLOT_SEQLETS:
        for i in range(min(n_top_samples_to_plot, len(seqlets))):
            plot_seqlet_with_atac(
                seqlets,
                attrs_list,
                atac_attribution_list=atac_attribution_list,
                atac_pileup_list=atac_pileup_list,
                sample_rank=i,
                context_size=plot_context_size,
                fname=os.path.join(
                    plots_dir,
                    f'rank{i}_{model}_{sample.upper()}_attributions_with_ATAC.pdf',
                ),
            )

    if RUN_SAVE_SEQLETS:
        save_seqlets(seqlets, seqlet_export_dir)
        print(f"Saved split seqlets + FASTA to: {seqlet_export_dir}")

    if RUN_PWM_SCORING:
        pwm = parse_jaspar(jaspar_file)
        print(f"Loaded PWM: {pwm.name} | consensus: {pwm.get_consensus()}")
        scores, positions = [], []
        for _, row in tqdm(seqlets.iterrows(), total=len(seqlets), desc="PWM scoring"):
            s, p = score_seqlet(pwm, row['sequence'])
            scores.append(s)
            positions.append(p)
        seqlets['pwm_score'] = scores
        seqlets['pwm_position'] = positions
        if not seqlets_from_cache and ready_df is not None:
            seqlets = add_genomic_coords(seqlets, ready_df)
        else:
            print(f"WARNING: skipping genomic coords (chr/g_start/g_end) -- seqlets "
                  f"came from cache, so example_idx ordering cannot be re-derived "
                  f"(streaming collect is non-deterministic). Delete {seqlets_path} "
                  f"to regenerate with coords.")
        scored_path = seqlets_path.replace('.csv', '_pwm_scored.csv')
        seqlets.to_csv(scored_path, index=False)
        print("\nTop 10 PWM matches:")
        print(seqlets.sort_values('pwm_score', ascending=False)
                     .head(10)[['sequence', 'pwm_score', 'pwm_position']])
        print(f"Wrote scored seqlets: {scored_path}")

    if RUN_EXTERNAL_TOOLS:
        # Why guarded: the legacy levenshtein.py path under src/inference/ no longer
        # exists in this repo; only posthoc.R is shipped under src/analysis/.
        if not os.path.isfile(seqlet_export_dir + "/positive_seqlets.csv"):
            print("External tools require RUN_SAVE_SEQLETS first.")
        else:
            if os.path.isfile(levenshtein_script):
                os.system(
                    f"python {levenshtein_script} "
                    f"--jaspar {jaspar_file} "
                    f"--seqlets {seqlet_export_dir}/positive_seqlets.csv "
                    f"--output {seqlet_export_dir}/lev_pwm.csv"
                )
            else:
                print(f"Skipping levenshtein.py (missing at {levenshtein_script}).")

            if os.path.isfile(posthoc_script):
                receptor_name = model.split("_")[0]
                os.system(
                    f"Rscript {posthoc_script} "
                    f"{PWM_MIN_SEQLET} {receptor_name} {seqlet_export_dir}"
                )
            else:
                print(f"Skipping posthoc.R (missing at {posthoc_script}).")


if __name__ == "__main__":
    main()

