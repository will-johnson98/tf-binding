import os
from typing import Optional, List, Iterator
from pathlib import Path
import polars as pl
from functools import partial


def find_jsonl_files(directory: str) -> List[Path]:
    """Find all JSONL files in directory"""
    return list(Path(directory).glob('*.jsonl.gz.out'))


def create_lazy_frame_from_file(file_path: Path) -> Optional[pl.LazyFrame]:
    """Create a LazyFrame from a single file with error handling"""
    try:
        lf = pl.scan_ndjson(file_path)  # Use scan_ndjson for lazy loading
        print(f"Successfully processed {file_path}")
        return lf
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def filter_valid_frames(lazy_frames: List[Optional[pl.LazyFrame]]) -> List[pl.LazyFrame]:
    """Filter out None values from lazy frames"""
    return [lf for lf in lazy_frames if lf is not None]


def process_jsonl_files_streaming(directory: str) -> Optional[pl.LazyFrame]:
    """Combine JSONL files into a LazyFrame using streaming without materialization"""
    print(f"Processing JSONL files in {directory}")
    
    json_files = find_jsonl_files(directory)
    if not json_files:
        print(f"No JSONL files found in {directory}")
        return None

    # Process files lazily
    lazy_frames = [create_lazy_frame_from_file(file) for file in json_files]
    valid_frames = filter_valid_frames(lazy_frames)
    
    if not valid_frames:
        print("No valid LazyFrames created from JSONL files")
        return None

    # Combine all lazy frames without materializing
    result_lf = pl.concat(valid_frames, how="vertical")
    print(f"Created LazyFrame from {len(valid_frames)} files")
    return result_lf


def ensure_directory_exists(file_path: str) -> None:
    """Ensure the directory for the file path exists"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


def write_lazy_frame_streaming(lazy_frame: pl.LazyFrame, output_file: str) -> bool:
    """Write LazyFrame to parquet using streaming without full materialization"""
    try:
        ensure_directory_exists(output_file)
        
        # Use sink_parquet for streaming write (if available) or collect in chunks
        if hasattr(lazy_frame, 'sink_parquet'):
            lazy_frame.sink_parquet(
                output_file,
                compression="snappy",
                row_group_size=50_000  # Smaller row groups for memory efficiency
            )
        else:
            # Fallback: collect and write (still problematic for large data)
            df = lazy_frame.collect(streaming=True)  # Use streaming collection if available
            df.write_parquet(
                output_file,
                compression="snappy",
                row_group_size=50_000
            )
        
        print(f"Results saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving results: {e}")
        return False


def create_output_path(project_path: str, model: str, sample: str) -> tuple[str, str]:
    """Create input and output paths"""
    input_dir = f"{project_path}/data/jsonl_output/{model}-{sample}"
    output_file = f"{project_path}/data/processed_results/{model}_{sample}_processed.parquet"
    return input_dir, output_file


def process_and_save_results(project_path: str, model: str, sample: str) -> bool:
    """Main processing pipeline function"""
    input_dir, output_file = create_output_path(project_path, model, sample)
    
    lazy_frame = process_jsonl_files_streaming(input_dir)
    if lazy_frame is None:
        return False
    
    return write_lazy_frame_streaming(lazy_frame, output_file)


def main():
    """Main entry point"""
    config = {
        "project_path": "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding",
        "model": "AR", 
        "sample": "DTB-036-A"
    }
    
    success = process_and_save_results(**config)
    exit_code = 0 if success else 1
    exit(exit_code)


if __name__ == "__main__":
    main()