import polars as pl
import os
import sys
import logging
from typing import Optional, List
from pathlib import Path
from polars import DataType
from polars.type_aliases import SchemaDict

logger = logging.getLogger(__name__)

def read_single_jsonl(file_path: Path) -> Optional[pl.LazyFrame]:
    """
    Read a single JSONL file with explicit schema inference
    """
    df = pl.read_json(file_path)

    # change df to lazyframe
    lf = df.lazy()
    return lf

def process_jsonl_files(directory: str) -> Optional[pl.LazyFrame]:
    """
    Combine JSONL files into a LazyFrame using streaming
    """
    logger.info(f"Processing JSONL files in {directory}")
    
    try:
        json_files = list(Path(directory).glob('*.jsonl.gz.out'))
        
        if not json_files:
            logger.warning(f"No JSONL files found in {directory}")
            return None

        # Read and collect valid LazyFrames
        lazy_frames: List[pl.LazyFrame] = []
        for file in json_files:
            lf = read_single_jsonl(file)
            if lf is not None:
                lazy_frames.append(lf)
                logger.info(f"Successfully processed {file}")

        if not lazy_frames:
            logger.warning("No valid LazyFrames created from JSONL files")
            return None

        # Combine all lazy frames
        result_lf = pl.concat(lazy_frames, how="vertical")
        
        logger.info(f"Created LazyFrame from {len(lazy_frames)} files")
        return result_lf

    except Exception as e:
        logger.error(f"Error processing JSONL files: {e}")
        return None

def save_to_parquet(
    lazy_frame: pl.LazyFrame,
    output_path: str,
    compression: str = "snappy",
    row_group_size: int = 100_000
) -> None:
    """
    Save LazyFrame to parquet
    """
    try:
        # Collect schema and print it for debugging
        schema = lazy_frame.schema
        logger.info(f"Schema before saving: {schema}")
        
        # First materialize to handle any schema issues
        df = lazy_frame.collect()

        print(f"Schema after collecting: {df.schema}")

        print(f"LazyFrame head: {df.head()}")
        
        # Save to parquet
        df.write_parquet(
            output_path,
            compression=compression,
            row_group_size=row_group_size
        )
        logger.info(f"Successfully saved parquet file to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving parquet file: {e}")
        raise

def main():
    # Configure logging to see more details
    logging.basicConfig(level=logging.INFO)
    
    # Configuration 
    jsonl_dir = "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/data/jsonl_output/AR-log10-22Rv1"
    project_path = "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding"
    model = "AR-log10"
    sample = "22Rv1"
    
    # Process files
    lazy_frame = process_jsonl_files(jsonl_dir)
    
    if lazy_frame is None:
        logger.error("Failed to create LazyFrame")
        sys.exit(1)
    
    # Save to parquet
    output_file = f"{project_path}/data/processed_results/{model}_{sample}_processed.parquet"
    save_to_parquet(lazy_frame, output_file)

if __name__ == "__main__":
    main()

