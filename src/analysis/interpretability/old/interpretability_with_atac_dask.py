# %%
######## LOAD DATA ########
import os
import sys
import numpy as np
import polars as pl

notebook_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(notebook_dir)



project_path = "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding"
model = "FOXA1"
sample = "22Rv1"
ground_truth_file = "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data/transcription_factors/FOXA1/merged/22RV1_FOXA1_merged.bed"

df = pl.read_parquet("/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data/processed_results/FOXA1-NoFlip_22Rv1_processed.parquet", 
                    columns=["chr_name", "start", "end", "cell_line", "targets", "predicted", "weights", "probabilities", "attributions"],
                    parallel="columns",                     # Enable parallel reading
                    use_statistics=True,                    # Use parquet statistics
                    memory_map=False).lazy()                         # Use memory mapping
df = df.rename({"chr_name": "chr"})

# %%
######## INTERSECT BED FILES ########
import tempfile
import subprocess

def run_bedtools_command(command: str) -> None:
    subprocess.run(command, shell=True, check=True)

def intersect_bed_files(main_df: pl.LazyFrame, intersect_df: pl.DataFrame, region_type: str = None) -> pl.LazyFrame:
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
    main_cols = main_df.schema.keys()
    
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

HIGH_COUNT_QUANTILE = 0.75
MAX_COUNT_THRESHOLD = 30
MID_COUNT_THRESHOLD = 10

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

df_ground_truth = pl.read_csv(ground_truth_file,
                             separator="\t",
                             has_header=False,
                             new_columns=["chr", "start", "end", "count"],
                             columns=[0,1,2,3])

df_ground_truth_filtered = threshold_peaks(df_ground_truth)

# Use select() instead of subscripting
intersected_df = intersect_bed_files(df.select(["chr", "start", "end"]), df_ground_truth_filtered)

# add overlaps ground truth to df from intersected_df
ground_truth_df = df.join(intersected_df, on=["chr", "start", "end"], how="left")

# add overlaps_ground_truth to df under targets, 1 if overlaps_ground_truth is true, 0 otherwise
ground_truth_df = ground_truth_df.with_columns(
    pl.when(pl.col("overlaps_ground_truth")).then(1).otherwise(0).alias("targets")
)

# %%
######## BALANCE DATASET ########

# Step 1: Keep the filtering lazy until collection
df_positive = ground_truth_df.filter(pl.col("targets") == 1).collect()
df_negative_all = ground_truth_df.filter(pl.col("targets") == 0).collect()

# Step 2: Get the count of positive samples
pos_count = len(df_positive)

# Step 3: Sample from the materialized negative DataFrame
df_negative = df_negative_all.sample(n=min(pos_count, len(df_negative_all)), seed=42)

# Step 4: Concatenate the two DataFrames
df_balanced = pl.concat([df_positive, df_negative])

df_balanced.head()
# %%
######## DEFINE FUNCTIONS FOR DASK PARALLELIZATION ########

import pysam
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
import gc
from tqdm.notebook import tqdm



def process_pileups(pileup_dir: Path, chr_name: str, start: int, end: int) -> pl.DataFrame:
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
    
    try:
        for rec in tabixfile.fetch(chr_name, start, end):
            records.append(rec.split("\t"))
    finally:
        tabixfile.close()
    
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
    # Handle both lazy and eager frames efficiently
    if hasattr(pileup_df, 'collect'):
        df_collected = pileup_df.collect()
    else:
        df_collected = pileup_df
        
    positions = df_collected.get_column("position").to_list()
    counts = df_collected.get_column("count").to_list()
    return dict(zip(positions, counts))


def create_atac_pileup_array(position_count_map: Dict[int, float], 
                            start_pos: int, 
                            length: int = 4096) -> np.ndarray:
    """Create ATAC pileup array for a genomic region."""
    atac_array = np.zeros(length, dtype=np.float32)  # Use float32 to save memory
    
    for i in range(length):
        pos = start_pos + i
        if pos in position_count_map:
            atac_array[i] = position_count_map[pos]
    
    return atac_array


def safe_collect_series(df: pl.DataFrame, column: str) -> pl.Series:
    """Safely collect a series from either LazyFrame or DataFrame."""
    try:
        if hasattr(df, 'collect'):
            return df.select(column).collect().get_column(column)
        else:
            return df.get_column(column)
    except Exception as e:
        print(f"Error collecting column '{column}': {e}")
        raise


def reshape_attributions_fast(attributions_list_input: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Memory-efficient reshape attribution data using vectorized operations.
    Accepts attributions directly as a list.
    
    Returns:
        attrs_list: Attribution scores for ACGT (shape: n_samples, 4, 4096)
        atac_attribution_list: ATAC attribution scores (shape: n_samples, 4096)
    """
    print("Reshaping attribution data from list...")
    
    try:
        # Attributions are now directly the input list
        attributions_list = attributions_list_input
        
        # Check if we have data
        if not attributions_list:
            raise ValueError("No attribution data found")
            
        # Convert to numpy array with memory-efficient dtype
        print(f"Converting {len(attributions_list)} attribution records to numpy array...")
        attributions = np.array(attributions_list, dtype=np.float32)
        
        # Clear the input list if it's safe to do so and helps memory,
        # but dask.delayed might need the original reference until execution.
        # For now, let's rely on standard GC.
        # del attributions_list_input # Potentially risky if Dask holds reference
        gc.collect()
        
        print(f"Attribution array shape: {attributions.shape}")
        
        # Vectorized reshape - much faster than loops
        reshaped = attributions.reshape(-1, 4096, 5)
        
        # Split into ACGT and ATAC components
        attrs_list = reshaped[..., :4].transpose(0, 2, 1)  # Shape: (n_samples, 4, 4096)
        atac_attribution_list = reshaped[..., 4]  # Shape: (n_samples, 4096)
        
        # Clean up intermediate arrays
        del attributions, reshaped
        gc.collect()
            
        return attrs_list, atac_attribution_list
        
    except Exception as e:
        print(f"Error in reshape_attributions_fast: {e}")
        raise


def get_unique_chromosomes(df: pl.DataFrame) -> List[str]:
    """Extract unique chromosomes from the dataframe."""
    try:
        chr_series = safe_collect_series(df, "chr")
        return chr_series.unique().to_list()
    except Exception as e:
        print(f"Error getting unique chromosomes: {e}")
        return []


def open_chromosome_tabix_files(pileup_dir: Path, chromosomes: List[str]) -> Dict[str, pysam.TabixFile]:
    """Open tabix files for all chromosomes and return mapping."""
    chr_tabix_files = {}
    
    print(f"Opening tabix files for {len(chromosomes)} chromosomes...")
    for chr_name in tqdm(chromosomes, desc="Loading chromosome files", leave=False):
        pileup_file = pileup_dir / f"{chr_name}.pileup.gz"
        if pileup_file.exists():
            try:
                chr_tabix_files[chr_name] = pysam.TabixFile(str(pileup_file))
            except Exception as e:
                print(f"Warning: Could not open {pileup_file}: {e}")
    
    return chr_tabix_files


def process_single_region(row: Dict, chr_tabix_files: Dict[str, pysam.TabixFile], context_length: int = 4096) -> np.ndarray:
    """Process a single genomic region and return ATAC array."""
    idx, chr_name, start, end = row['idx'], row['chr'], row['start'], row['end']
    
    # Calculate adjusted coordinates
    interval_length = end - start
    extra_seq = context_length - interval_length
    extra_left_seq = extra_seq // 2
    extra_right_seq = extra_seq - extra_left_seq
    adj_start = start - extra_left_seq
    adj_end = end + extra_right_seq
    
    # Initialize array with float32 to save memory
    atac_array = np.zeros(context_length, dtype=np.float32)
    
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
                counts = np.array(counts, dtype=np.float32)
                
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
    
    return atac_array


def close_tabix_files(chr_tabix_files: Dict[str, pysam.TabixFile]) -> None:
    """Safely close all tabix files."""
    print("Closing tabix files...")
    for chr_name, tabixfile in chr_tabix_files.items():
        try:
            tabixfile.close()
        except Exception as e:
            print(f"Warning: Error closing tabix file for {chr_name}: {e}")


def process_pileups_batch(pileup_dir: Path, regions_df: pl.DataFrame) -> Dict[int, np.ndarray]:
    """Process multiple pileup regions for a single cell line efficiently."""
    context_length = 4_096
    
    # Get unique chromosomes to minimize file operations
    chromosomes = get_unique_chromosomes(regions_df)
    if not chromosomes:
        print("Warning: No chromosomes found in regions data")
        return {}
    
    chr_tabix_files = open_chromosome_tabix_files(pileup_dir, chromosomes)
    
    if not chr_tabix_files:
        print("Warning: No tabix files could be opened")
        return {}
    
    # Process each region
    atac_arrays = {}
    
    try:
        # Convert to regular DataFrame if it's a LazyFrame - but do it efficiently
        if hasattr(regions_df, 'collect'):
            regions_data = regions_df.collect()
        else:
            regions_data = regions_df
        
        region_iterator = regions_data.iter_rows(named=True)
        total_regions = regions_data.height
        
        for row in tqdm(region_iterator, total=total_regions, desc="Processing regions", leave=False):
            atac_array = process_single_region(row, chr_tabix_files, context_length)
            atac_arrays[row['idx']] = atac_array
    
    finally:
        # Always close tabix files
        close_tabix_files(chr_tabix_files)
    
    return atac_arrays


def get_dataframe_length(df: pl.DataFrame) -> int:
    """Get the length of a DataFrame, handling both lazy and eager frames."""
    try:
        if hasattr(df, 'collect'):
            return df.select(pl.len()).collect().item()
        else:
            return len(df)
    except Exception as e:
        print(f"Error getting dataframe length: {e}")
        return 0


def group_by_cell_line(df_with_idx: pl.DataFrame) -> List[Tuple[Any, pl.DataFrame]]:
    """
    Groups a Polars DataFrame (eager or lazy) by the 'cell_line' column.
    Returns a list of tuples, where each tuple contains the cell line key (as a tuple)
    and its corresponding eager DataFrame.
    """
    try:
        is_lazy = hasattr(df_with_idx, 'collect') and callable(df_with_idx.collect) and hasattr(df_with_idx, 'lazy')

        if is_lazy:
            # print("Processing LazyFrame in group_by_cell_line: collecting unique cell lines and then groups.")
            unique_cell_lines_df = df_with_idx.select(pl.col("cell_line")).unique().collect(streaming=True)
            
            if unique_cell_lines_df.height == 0:
                return []
            unique_cell_lines = unique_cell_lines_df.get_column("cell_line").to_list()

            grouped_data = []
            for cell_line_value in unique_cell_lines:
                group_df = df_with_idx.filter(pl.col("cell_line") == cell_line_value).collect(streaming=True)
                grouped_data.append(((cell_line_value,), group_df))
            return grouped_data
        else: # Eager DataFrame
            # print("Processing EagerFrame in group_by_cell_line.")
            return list(df_with_idx.group_by("cell_line", maintain_order=True))

    except Exception as e:
        print(f"Error in group_by_cell_line: {e}")
        return []



# %%
######## PROCESS DATA WITH DASK TO GET LISTS OF BASE ATTRIBUTIONS, ATAC ATTRIBUTIONS, AND PILEUP VALUES ########
import dask
from dask.distributed import Client, progress
from dask_jobqueue import SGECluster


def reshape_attributions_on_worker(attributions_df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function to run on a Dask worker.
    Converts the 'attributions' column of the given DataFrame to a list,
    then calls reshape_attributions_fast.
    """
    print("Reshaping attributions on worker: converting column to list...")
    attributions_list = attributions_df.get_column("attributions").to_list()
    print(f"Converted to list with {len(attributions_list)} entries on worker.")
    return reshape_attributions_fast(attributions_list)


def process_region_data_dask_fully_utilized(
    df: pl.DataFrame,  # Expects an eager Polars DataFrame or a LazyFrame that's manageable to collect parts of on client/pass to workers
    base_pileup_dir: Optional[Path] = None,
    worker_cores: int = 4,
    worker_memory: str = "16GB",
    num_workers: int = 10,
    sge_queue: Optional[str] = 'main', # Example: Add SGE queue
    sge_project: Optional[str] = None, # Example: Add SGE project
    network_interface: str = 'ens7f0' # on worker: ['lo', 'ens7f0', 'ens7f1', 'docker0', 'enp0s20f0u7u2c2'] on head node:['lo', 'eno1', 'eno2', 'enp23s0f0', 'enp23s0f1', 'docker0', 'vethde8cbac']
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Processes attribution and pileup data, utilizing Dask for the entire computational workflow
    once the initial Polars DataFrame is provided.
    """
    if base_pileup_dir is None:
        base_pileup_dir = Path("/data1/projects/human_cistrome/aligned_chip_data/merged_cell_lines/")

    # This length is determined on the client. If df is Lazy, it will be collected here.
    # This is needed to pre-allocate the final numpy array.
    # Ensure `df` is eager or collect it if it's lazy and its length is needed upfront.
    # If df is truly massive and lazy, get_dataframe_length might be too slow/memory intensive on client.
    # For this refactor, we assume df is either eager or get_dataframe_length is acceptable.
    actual_df_length = get_dataframe_length(df)
    if actual_df_length == 0:
        raise ValueError("DataFrame is empty or its length could not be determined.")

    print(f"Processing {actual_df_length} regions across cell lines using Dask...")

    print("Step 1: Setting up Dask SGECluster...")

    os.makedirs("/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/logs/dask_logs", exist_ok=True)
    
    if os.path.exists("/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/logs/dask_logs/dask_logs.txt"):
        os.remove("/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/logs/dask_logs/dask_logs.txt")

    cluster_kwargs = {
        "queue": sge_queue,
        "cores": worker_cores,
        "memory": worker_memory,
        "job_extra_directives": [
            "-j y", 
            "-o /data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/logs/dask_logs/dask_logs.txt",
            "-l hostname=node5"
        ],
        "interface": network_interface, # Interface for Dask workers (e.g., 'ens7f0')
        "scheduler_options": {'interface': 'eno1'}, # Interface for Dask scheduler (on head node)
        "log_directory": '/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/logs/dask_logs', # Recommended for debugging
    }
    if sge_project:
        cluster_kwargs["project"] = sge_project
    
    cluster = SGECluster(**cluster_kwargs)
    cluster.scale(n=num_workers)
    client = Client(cluster)
    print(f"Dask client created. Dashboard link: {client.dashboard_link}")

    # --- Task 1: Reshape Attributions (to be run on a Dask worker) ---
    print("Step 2: Submitting attribution reshaping task to Dask...")
    # Prepare the "attributions" column as a DataFrame for the Dask task
    # Cloning helps ensure it's a clean, eager DataFrame.
    print("Preparing attributions column DataFrame for Dask task...")
    attributions_df_for_task = df.select("attributions").clone()
    print(f"Prepared DataFrame with 'attributions' column for worker.")

    # Scatter the large DataFrame to workers and get a future
    print("Scattering attributions DataFrame to Dask workers...")
    future_attributions_df = client.scatter(attributions_df_for_task, broadcast=True)
    # Clear the local copy to free memory on the client
    del attributions_df_for_task 
    gc.collect()

    future_attrs_tuple = dask.delayed(reshape_attributions_on_worker, name="reshape-attributions-on-worker")(future_attributions_df)

    # --- Prepare for Pileup Processing Tasks ---
    # `df.with_row_index` is relatively cheap on the client if `df` is eager.
    # If `df` is lazy, it remains lazy. Sub-dataframes passed to workers will be Polars DFs.
    df_with_idx = df.with_row_index("idx")

    # Select only necessary columns for grouping and pileup processing
    df_for_grouping = df_with_idx.select(["idx", "chr", "start", "end", "cell_line"]) 

    # `group_by_cell_line` will collect if `df_for_grouping` is lazy.
    # This happens on the client before submitting tasks.
    # The resulting `group_df_for_cell_line` (Polars DFs) are sent to workers.
    cell_line_groups = group_by_cell_line(df_for_grouping)

    pileup_delayed_tasks = []
    if not cell_line_groups:
        print("Warning: No cell line groups found for pileup processing.")
    else:
        print(f"Step 3: Submitting {len(cell_line_groups)} cell line pileup processing tasks to Dask...")
        for i, (cell_line_key, group_df_for_cell_line) in enumerate(tqdm(cell_line_groups, desc="Preparing Dask pileup tasks")):
            cell_line_name = cell_line_key[0] if isinstance(cell_line_key, tuple) else cell_line_key
            pileup_dir_for_task = base_pileup_dir / str(cell_line_name) / "pileup_mod" # Ensure cell_line_name is string
            
            # Ensure group_df_for_cell_line is an eager DataFrame before sending
            # (group_by typically yields eager DataFrames for groups)
            if hasattr(group_df_for_cell_line, 'collect'): # Should not be needed if group_by collected
                 group_df_for_cell_line = group_df_for_cell_line.collect()

            # Scatter the group DataFrame to workers
            future_group_df = client.scatter(group_df_for_cell_line, broadcast=False)
            # Clear the local copy to free memory on the client.
            del group_df_for_cell_line 
            gc.collect()

            task = dask.delayed(process_pileups_batch, name=f"pileup-batch-{i}-{cell_line_name}")(pileup_dir_for_task, future_group_df)
            pileup_delayed_tasks.append(task)

    # --- Compute All Tasks ---
    all_tasks_to_compute = [future_attrs_tuple] + pileup_delayed_tasks
    
    print("Step 4: Computing all tasks on Dask workers...")
    # Using client.compute and client.gather for explicit control
    # Alternatively, use dask.compute(*all_tasks_to_compute) which blocks and returns results.
    # For long computations, you might prefer futures to monitor with progress() or dashboard.
    # Here, for simplicity matching the original blocking style:
    # computed_results = dask.compute(*all_tasks_to_compute) # This blocks until all tasks are done

    # Switch to client.compute to get futures, then use progress
    futures = client.compute(all_tasks_to_compute)
    progress(futures) # Display Dask progress bar
    computed_results = client.gather(futures) # This blocks until all tasks are done and gathers results

    # --- Process Results ---
    # Results from reshape_attributions_fast
    attrs_list, atac_attribution_list = computed_results[0]
    print("Attribution reshaping complete (results received from Dask worker).")
    
    # Initialize the final pileup result array now that we have actual_df_length (or from attrs_list.shape[0])
    # If attrs_list.shape[0] is more reliable or if actual_df_length was deferred:
    num_samples_from_attrs = attrs_list.shape[0]
    if num_samples_from_attrs != actual_df_length:
        print(f"Warning: DataFrame length mismatch. Client: {actual_df_length}, Attributions: {num_samples_from_attrs}.")
        # Potentially use num_samples_from_attrs if it's considered more accurate post-processing
    
    final_atac_pileup_list = np.zeros((num_samples_from_attrs, 4096), dtype=np.float32)

    # Results from pileup_delayed_tasks
    pileup_results_list_of_dicts = computed_results[1:]
    print("Step 5: Consolidating pileup results from Dask workers...")
    for atac_arrays_dict in tqdm(pileup_results_list_of_dicts, desc="Consolidating pileup results"):
        if atac_arrays_dict: # It could be {} if pileup_dir didn't exist or no data
            for idx, atac_array in atac_arrays_dict.items():
                if idx < num_samples_from_attrs: # Check bounds against the actual size
                    final_atac_pileup_list[idx] = atac_array
                else:
                    print(f"Warning: Index {idx} out of bounds for final_atac_pileup_list (length {num_samples_from_attrs})")
    
    # --- Clean Up Dask Cluster ---
    print("Step 6: Closing Dask client and cluster...")
    client.close()
    cluster.close()
    print("Processing complete!")
    
    return attrs_list, atac_attribution_list, final_atac_pileup_list


attrs_list, atac_attribution_list, atac_pileup_list = process_region_data_dask_fully_utilized(df_balanced)

# %%
# Import additional required libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('whitegrid')
from tangermeme.plot import plot_logo
from tangermeme.seqlet import recursive_seqlets

# Get seqlets
def get_seqlets(attrs_list):
    attrs_array = np.stack(attrs_list, axis=0)
    seqlets = recursive_seqlets(attrs_array.sum(axis=1))
    
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

# Plot function (simplified version)
def plot_seqlet(seqlets, attrs_list, sample_rank=0, context_size=20):
    sample = seqlets.iloc[[sample_rank]]
    slice = int(sample['example_idx'].tolist()[0])
    sequence = sample['sequence'].tolist()[0]
    start = int(sample['start'].tolist()[0])
    end = int(sample['end'].tolist()[0])
    
    seqlen = end - start
    window_size = seqlen + (context_size * 2)
    
    X_attr = attrs_list[slice]
    X_attr = X_attr.astype(np.float64)
    
    TSS_pos = int(np.mean([start, end]))
    window = (TSS_pos - (window_size // 2), TSS_pos + (window_size // 2))
    
    plt.figure(figsize=(16, 9), dpi=300)
    ax = plt.subplot(111)
    plot_logo(
        X_attr,
        ax=ax,
        start=window[0],
        end=window[1]
    )
    
    plt.xlabel("Genomic Coordinate")
    plt.ylabel("Attributions")
    plt.title(f"DeepLIFT Attributions for sample: {slice} | {sequence}")
    plt.show()

# %%
seqlets = get_seqlets(attrs_list)
filtered_seqlets = seqlets[seqlets["sequence"] == "AAAAA"]
filtered_seqlets

# %%
plot_seqlet(seqlets, attrs_list, sample_rank=1341, context_size=200)


