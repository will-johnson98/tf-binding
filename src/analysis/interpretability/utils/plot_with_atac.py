#!/usr/bin/env python3
# WAJ 2026-04-28

import os
import pickle
import gzip
import numpy as np
import polars as pl
import pandas as pd
import torch
import tempfile
import pysam
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('whitegrid')

from tangermeme.plot import plot_logo
from tangermeme.seqlet import recursive_seqlets, tfmodisco_seqlets
from matplotlib.colors import TwoSlopeNorm
from src.utils.generate_training_peaks import run_bedtools_command
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Dict
from datetime import datetime


######################################################################################################################################
project_path = "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding"
interpretability_path = os.path.join(project_path, 
                                     "src/analysis/interpretability")

model = "ASCL1"
long_model_name = "ASCL1-LU49-PDXs"
sample = "LuCaP-49"
# sample = "22RV1"

# method = 'tfmodisco'
method = 'recursive'

make_plots = True
n_top_samples_to_plot = 10
plot_context_size = 100
today = datetime.now().strftime("%Y%m%d")
ground_truth_subset = "TP"


# Shouldn't need to change these, but just in case:

plots_dir = f"plots/{today}_{model}_{sample}_{ground_truth_subset}_{plot_context_size}bp"
seqlets_path = os.path.join(interpretability_path, 
                            f"output/{today}_{model}_{sample.upper()}_{ground_truth_subset}_seqlets.csv")


MAX_SAMPLES_FOR_TFMODISCO = 5000
HIGH_COUNT_QUANTILE = 0.75
MAX_COUNT_THRESHOLD = 30
MID_COUNT_THRESHOLD = 10
######################################################################################################################################



def intersect_bed_files(main_df: pl.LazyFrame, 
                        intersect_df: pl.DataFrame, 
                        region_type: str = None) -> pl.LazyFrame:
    """
    Intersect two BED files using bedtools and return the original DataFrame with overlap flags.
    Args:
    main_df: Primary Polars DataFrame with BED data
    intersect_df: Secondary Polars DataFrame to intersect with
    region_type: Optional region type label to add to results
    Returns:
    Original DataFrame with additional column indicating overlaps
    """

    # Get column names from schema
    _schema = main_df.collect_schema()
    # main_cols = main_df.schema.keys()
    main_cols = _schema.keys()
    
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as main_file, \
         tempfile.NamedTemporaryFile(delete=False, mode='w') as intersect_file, \
         tempfile.NamedTemporaryFile(delete=False, mode='w') as result_file:
        main_path = main_file.name
        intersect_path = intersect_file.name
        result_path = result_file.name
        
        # Write DataFrames to temporary files - collect LazyFrame first
        main_df.collect().write_csv(main_path, separator="\t", include_header=False)
        intersect_df.write_csv(intersect_path, separator="\t", include_header=False)
        
        # Run bedtools intersect with -c flag to count overlaps
        command = f"bedtools intersect -a {main_path} -b {intersect_path} -c > {result_path}"
        run_bedtools_command(command)
        
        # Read results back into Polars DataFrame
        result_df = pl.read_csv(
            result_path,
            separator="\t",
            has_header=False,
            new_columns=[*main_cols, "overlap_count"]
        ).lazy()
        
        # Clean up temporary files
        os.remove(main_path)
        os.remove(intersect_path)
        os.remove(result_path)
        
        # Add boolean overlap column
        return result_df.with_columns(
            pl.col("overlap_count").gt(0).alias("overlaps_ground_truth")
        ).drop("overlap_count")



def threshold_peaks(df):
    """
    Filter peaks based on count thresholds.
    Works with both DataFrame and LazyFrame.
    """

    # Handle scalar operations safely
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



def process_pileups(pileup_dir: Path, 
                    chr_name: str, 
                    start: int, 
                    end: int) -> pl.DataFrame:
    """Process pileup files for a given genomic region with 4096bp context."""
    context_length = 4_096
    interval_length = end - start
    extra_seq = context_length - interval_length
    extra_left_seq = extra_seq // 2
    extra_right_seq = extra_seq - extra_left_seq
    start -= extra_left_seq
    end += extra_right_seq
    
    # Get the pileup file for the given chromosome
    pileup_file = pileup_dir / f"{chr_name}.pileup.gz"
    assert pileup_file.exists(), f"pileup file for {pileup_file} does not exist"
    
    tabixfile = pysam.TabixFile(str(pileup_file))
    records = []
    for rec in tabixfile.fetch(chr_name, start, end):
        records.append(rec.split("\t"))
    
    # Convert records to a DataFrame using Polars
    df = pl.DataFrame({
        "chr_name": [rec[0] for rec in records],
        "position": [int(rec[1]) for rec in records],
        "nucleotide": [rec[2] for rec in records],
        "count": [float(rec[3]) for rec in records],
    })
    
    return df



def create_position_to_count_mapping(pileup_df: pl.DataFrame) -> Dict[int, float]:
    """Create a mapping from genomic position to ATAC count."""
    return dict(zip(pileup_df['position'].to_list(), pileup_df['count'].to_list()))



def create_atac_pileup_array(position_count_map: Dict[int, float], 
                            start_pos: int, 
                            length: int = 4096) -> np.ndarray:
    """Create ATAC pileup array for a genomic region."""
    atac_array = np.zeros(length)
    
    for i in range(length):
        pos = start_pos + i
        if pos in position_count_map:
            atac_array[i] = position_count_map[pos]
    
    return atac_array



def reshape_attributions_fast(df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast reshape attribution data using vectorized operations.
    
    Returns:
        attrs_list: Attribution scores for ACGT (shape: n_samples, 4, 4096)
        atac_attribution_list: ATAC attribution scores (shape: n_samples, 4096)
    """
    print("Reshaping attribution data...")
    # Convert to numpy array more efficiently
    attributions = np.array(df['attributions'].to_list())
    
    # Vectorized reshape - much faster than loops
    reshaped = attributions.reshape(-1, 4096, 5)
    
    # Split into ACGT and ATAC components
    attrs_list = reshaped[..., :4].transpose(0, 2, 1)  # Shape: (n_samples, 4, 4096)
    atac_attribution_list = reshaped[..., 4]  # Shape: (n_samples, 4096)
            
    return attrs_list, atac_attribution_list



def process_pileups_batch(pileup_dir: Path, 
                          regions_df: pl.DataFrame) -> Dict[int, np.ndarray]:
    """Process multiple pileup regions for a single cell line efficiently."""
    context_length = 4_096
    atac_arrays = {}
    
    # Get unique chromosomes to minimize file operations
    chromosomes = regions_df['chr'].unique().to_list()
    chr_tabix_files = {}
    
    # Open all needed tabix files once
    print(f"Opening tabix files for {len(chromosomes)} chromosomes...")
    for chr_name in tqdm(chromosomes, desc="Loading chromosome files", leave=False):
        pileup_file = pileup_dir / f"{chr_name}.pileup.gz"
        if pileup_file.exists():
            chr_tabix_files[chr_name] = pysam.TabixFile(str(pileup_file))
    
    # Process each region
    region_iterator = regions_df.iter_rows(named=True)
    total_regions = len(regions_df)
    
    for row in tqdm(region_iterator, total=total_regions, desc="Processing regions", leave=False):
        idx, chr_name, start, end = row['idx'], row['chr'], row['start'], row['end']
        
        # Calculate adjusted coordinates
        interval_length = end - start
        extra_seq = context_length - interval_length
        extra_left_seq = extra_seq // 2
        extra_right_seq = extra_seq - extra_left_seq
        adj_start = start - extra_left_seq
        adj_end = end + extra_right_seq
        
        # Initialize array
        atac_array = np.zeros(context_length)
        
        # Get data if tabix file exists
        if chr_name in chr_tabix_files:
            tabixfile = chr_tabix_files[chr_name]
            
            # Collect all positions and counts at once
            positions = []
            counts = []
            
            try:
                for rec in tabixfile.fetch(chr_name, adj_start, adj_end):
                    fields = rec.split("\t")
                    positions.append(int(fields[1]))
                    counts.append(float(fields[3]))
                
                # Vectorized assignment
                if positions:
                    positions = np.array(positions)
                    counts = np.array(counts)
                    
                    # Calculate array indices
                    array_indices = positions - adj_start
                    
                    # Filter valid indices
                    valid_mask = (array_indices >= 0) & (array_indices < context_length)
                    valid_indices = array_indices[valid_mask]
                    valid_counts = counts[valid_mask]
                    
                    # Assign values
                    atac_array[valid_indices] = valid_counts
                    
            except Exception as e:
                print(f"Warning: Could not fetch data for {chr_name}:{adj_start}-{adj_end}: {e}")
        
        atac_arrays[idx] = atac_array
    
    # Close tabix files
    print("Closing tabix files...")
    for tabixfile in chr_tabix_files.values():
        tabixfile.close()
    
    return atac_arrays



def process_region_data_fast(df: pl.DataFrame, 
                             base_pileup_dir: Path = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast process both attribution and pileup data using Polars optimizations.
    
    Args:
        df: DataFrame containing attribution data and region info (chr, start, end, cell_line columns)
        base_pileup_dir: Base directory path for pileup files (optional, uses default if None)
    
    Returns:
        attrs_list: Attribution scores for ACGT
        atac_attribution_list: ATAC attribution scores
        atac_pileup_list: Raw ATAC pileup counts
    """
    if base_pileup_dir is None:
        base_pileup_dir = Path("/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines/")
    
    print(f"Processing {len(df)} regions across cell lines...")
    
    # Get attribution data (fast vectorized version)
    attrs_list, atac_attribution_list = reshape_attributions_fast(df)
    
    # Add row index for tracking
    df_with_idx = df.with_row_index("idx")
    
    # Group by cell line for batch processing
    atac_pileup_arrays = [None] * len(df)
    cell_line_groups = list(df_with_idx.group_by("cell_line"))
    
    print(f"Processing {len(cell_line_groups)} cell lines...")
    
    for cell_line, group_df in tqdm(cell_line_groups, desc="Processing cell lines"):
        cell_line_name = cell_line[0]
        
        # Construct cell-line specific pileup directory
        pileup_dir = base_pileup_dir / cell_line_name / "pileup_mod"
        
        if not pileup_dir.exists():
            print(f"Warning: Pileup directory does not exist: {pileup_dir}")
            # Fill with zeros for this cell line
            for row in group_df.iter_rows(named=True):
                atac_pileup_arrays[row['idx']] = np.zeros(4096)
            continue
        
        print(f"Processing {len(group_df)} regions for cell line: {cell_line_name}")
        
        # Process all regions for this cell line at once
        atac_arrays_dict = process_pileups_batch(pileup_dir, group_df)
        
        # Assign to the correct positions in the final array
        for idx, atac_array in atac_arrays_dict.items():
            atac_pileup_arrays[idx] = atac_array
    
    print("Converting to final numpy arrays...")
    # Convert to numpy array
    atac_pileup_list = np.array(atac_pileup_arrays)
    
    print("Processing complete!")
    return attrs_list, atac_attribution_list, atac_pileup_list



def get_seqlets(attrs_list, 
                use_absolute_values=False, 
                method='recursive', 
                direction='positive', 
                **kwargs):
    """
    Extract seqlets from attribution data using one of two methods.

    Args:
        attrs_list (list): List of attribution arrays for each sample.
        use_absolute_values (bool): Whether to use absolute attribution values for peak finding.
        method (str): The seqlet calling method to use. Either 'tfmodisco' (default)
                      or 'recursive'.
        direction (str): 'positive' to find seqlets in high-attribution regions,
                         'negative' to find them in low-attribution (negative) regions.
        **kwargs: Method-specific arguments.
            For 'tfmodisco':
                - See tangermeme.seqlet.tfmodisco_seqlets documentation. Common
                  parameters include `window_size` and `flank`.
            For 'recursive':
                - threshold (float): p-value threshold for a span to be considered
                                     a seqlet. Default: 0.01.
                - min_seqlet_len (int): Minimum length of a seqlet. Default: 4.
                - max_seqlet_len (int): Maximum length of a seqlet. Default: 25.
                - additional_flanks (int): Number of base pairs to add to each side
                                           of a discovered seqlet. Default: 0.
    """
    attrs_array = np.stack(attrs_list, axis=0)

    # Sum attributions across one-hot encoded dimension to get a per-position score
    summed_attrs = attrs_array.sum(axis=1)

    # If looking for negative contributions, flip the scores
    if direction == 'negative':
        summed_attrs = -summed_attrs
    elif direction != 'positive':
        raise ValueError("direction must be 'positive' or 'negative'")

    # Optionally use absolute values. Note: this will make 'negative' direction meaningless.
    if use_absolute_values:
        summed_attrs = np.abs(summed_attrs)

    if method == 'tfmodisco':
        summed_attrs_tensor = torch.from_numpy(summed_attrs).float()
        seqlets = tfmodisco_seqlets(summed_attrs_tensor)#, **kwargs)
    elif method == 'recursive':
        # # recursive_seqlets uses a p-value based threshold.
        # # We set some reasonable defaults based on its documentation.
        r_kwargs = {
            'threshold': 0.01,
            'min_seqlet_len': 5,
            'max_seqlet_len': 25,
            'additional_flanks': 2
        }
        r_kwargs.update(kwargs)
        seqlets = recursive_seqlets(summed_attrs, **r_kwargs)
    else:
        raise ValueError(f"Unknown seqlet calling method: '{method}'. Choose 'tfmodisco' or 'recursive'.")


    nt_idx = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

    # Add sequences to seqlets df
    sequences = []
    for i in range(len(seqlets)):
        sample = seqlets.iloc[i]
        start = int(sample['start'])
        end = int(sample['end'])
        sample_idx = int(sample['example_idx'])

        sample_attrs = attrs_array[sample_idx, :, start:end].T.squeeze()
        hits = np.argmax(sample_attrs, axis=1)
        seq = ''.join([nt_idx[i] for i in hits])
        sequences.append(seq)
    
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
    """
    Create a two-panel plot with a NON-SYMMETRIC color-normalized heatmap.
    - Top: DNA base attributions (logo plot)  
    - Bottom: ATAC pileup with attribution heatmap background (0 is always white)
    
    The color scale for the heatmap now stretches to the true min and max of the data in the window.
    """
    # --- This part of the function is unchanged ---
    sample = seqlets.iloc[[sample_rank]]
    slice_idx = int(sample['example_idx'].tolist()[0])
    sequence = sample['sequence'].tolist()[0]
    start = int(sample['start'].tolist()[0])
    end = int(sample['end'].tolist()[0])

    seqlet_center = (start + end) // 2
    seqlet_length = end - start
    total_window_size = seqlet_length + (2 * context_size)
    window_start = seqlet_center - (total_window_size // 2)
    window_end = seqlet_center + (total_window_size // 2)
    window_start = max(0, window_start)
    window_end = min(4096, window_end)
    if window_end - window_start < total_window_size:
        if window_start == 0:
            window_end = min(4096, window_start + total_window_size)
        elif window_end == 4096:
            window_start = max(0, window_end - total_window_size)
    
    print(f"Seqlet: {start}-{end} (center: {seqlet_center})")
    print(f"Window: {window_start}-{window_end} (size: {window_end - window_start})")
    if 'p-value' in sample.columns:
        p_value = sample['p-value'].tolist()[0]
        print(f"P-Value: {p_value}")
    
    plot_coords = np.arange(window_start, window_end)
    X_attr = attrs_list[slice_idx].astype(np.float64)
    atac_attr = atac_attribution_list[slice_idx].astype(np.float64)
    atac_pileup = atac_pileup_list[slice_idx].astype(np.float64)
    X_attr_windowed = X_attr[:, window_start:window_end]
    atac_attr_windowed = atac_attr[window_start:window_end]
    atac_pileup_windowed = atac_pileup[window_start:window_end]
    
    print(f"Windowed shapes: DNA={X_attr_windowed.shape}, ATAC_attr={atac_attr_windowed.shape}, ATAC_pileup={atac_pileup_windowed.shape}")

    fig = plt.figure(figsize=(18, 10), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[20, 1], height_ratios=[1, 1], hspace=0.3, wspace=0.02)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])

    # Top panel logic remains the same...
    plot_logo(X_attr_windowed, ax=ax1)
    n_ticks = 8
    tick_positions = np.linspace(0, len(plot_coords)-1, n_ticks)
    tick_labels = np.linspace(plot_coords[0], plot_coords[-1], n_ticks).astype(int)
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)
    ax1.set_xlabel("Genomic Coordinate")
    ax1.set_ylabel("DNA Attributions")
    ax1.set_title(f"DNA Base Attributions | Sample: {slice_idx} | {sequence}")

    # --- This part of the function is also unchanged ---
    heatmap_height = 25
    attr_heatmap = np.tile(atac_attr_windowed, (heatmap_height, 1))
    max_pileup = np.max(atac_pileup_windowed) if len(atac_pileup_windowed) > 0 else 1
    y_max = max_pileup * 1.1

    # <<< MODIFIED SECTION >>>
    # CREATE THE ASYMMETRIC NORMALIZER CENTERED AT 0
    if atac_attr_windowed.size > 0:
        # Get the true min and max of the data in the window
        vmin_val = np.min(atac_attr_windowed)
        vmax_val = np.max(atac_attr_windowed)
    else:
        # Handle empty window case
        vmin_val, vmax_val = -1, 1

    if equal_color_scale:
        # Handle empty window case for np.abs
        if atac_attr_windowed.size > 0:
            vabs_max = np.max(np.abs(atac_attr_windowed))
        else:
            vabs_max = 1
        vmin_val = -vabs_max
        vmax_val = vabs_max 
        
    # Create the normalizer with the actual data bounds, keeping 0 as the center
    norm = TwoSlopeNorm(vcenter=0, vmin=vmin_val, vmax=vmax_val)
    # <<< END OF MODIFIED SECTION >>>

    # Create the heatmap background using the SAME coordinate system and the NEW norm
    im = ax2.imshow(attr_heatmap, 
                    cmap=colormap,
                    aspect='auto',
                    extent=[plot_coords[0], plot_coords[-1], 0, y_max],
                    alpha=0.7,
                    interpolation='bilinear',
                    norm=norm) # Apply the new asymmetric normalizer
    
    # --- The rest of the function is unchanged ---
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


    # Handle file saving or display
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', dpi=300)
        print(f"Plot saved to: {fname}")
        plt.close()  
    else:
        plt.show()



def main():
    os.makedirs(plots_dir, exist_ok=True)

    # jaspar_file = f"{project_path}/src/analysis/interpretability/motifs/{model}.jaspar"  # Update this path
    ground_truth_file = f"{project_path}/data/transcription_factors/{model}/merged/{sample}_{model}_merged.bed"
    parquet_file = f"{project_path}/data/processed_results/{long_model_name}_{sample}_processed.parquet"


    df = pl.read_parquet(parquet_file, 
                            columns=["chr_name", "start", "end", "cell_line", 
                                    "targets", "predicted", "weights",
                                    "probabilities", "attributions"],
                            parallel="columns",                     # Enable parallel reading
                            use_statistics=True,                    # Use parquet statistics
                            memory_map=True).lazy()                 # Use memory mapping
    df = df.rename({"chr_name": "chr"})
    
    df_ground_truth = pl.read_csv(ground_truth_file,
                                    separator="\t",
                                    has_header=False,
                                    new_columns=["chr", "start", "end", "count"],
                                    columns=[0,1,2,3])

    df_ground_truth_filtered = threshold_peaks(df_ground_truth)

    intersected_df = intersect_bed_files(df.select(["chr", "start", "end"]), df_ground_truth_filtered)

    ground_truth_df = df.join(intersected_df, on=["chr", "start", "end"], how="left")

    ground_truth_df = ground_truth_df.with_columns(
        pl.when(pl.col("overlaps_ground_truth")).then(1).otherwise(0).alias("targets")
    )
    
    ground_truth_df = (
        ground_truth_df
        .with_columns(
            correct=pl.when((pl.col("targets") == 1) & (pl.col("predicted") == 1)).then(pl.lit("TP"))
                        .when((pl.col("targets") == 0) & (pl.col("predicted") == 0)).then(pl.lit("TN"))
                        .when((pl.col("targets") == 1) & (pl.col("predicted") == 0)).then(pl.lit("FN"))
                        .when((pl.col("targets") == 0) & (pl.col("predicted") == 1)).then(pl.lit("FP"))
                        )
                        .collect(streaming=True)
                        )

    
    ready_df = ground_truth_df.filter(pl.col("correct") == ground_truth_subset)


    attrs_list, atac_attribution_list, atac_pileup_list = process_region_data_fast(ready_df)



    if os.path.isfile(seqlets_path): 
        seqlets = pd.read_csv(seqlets_path)
    else:
        seqlets = get_seqlets(attrs_list, method=method)
        seqlets.to_csv(seqlets_path, index=False)

    if make_plots:
        for i in range(n_top_samples_to_plot):
            plot_seqlet_with_atac(
                seqlets, 
                attrs_list,
                atac_attribution_list=atac_attribution_list, 
                atac_pileup_list=atac_pileup_list, 
                sample_rank=i, 
                context_size=plot_context_size,
                fname=os.path.join(plots_dir, f'rank{i}_{model}_{sample.upper()}_attributions_with_ATAC.pdf')
            )


if __name__ == "__main__":
    main()