#!/usr/bin/env python3
"""Convert a large processed-results parquet to csv.gz, expanding the nested
`attributions` column into per-channel, JSON-encoded list columns.

The parquet stores `attributions` as List(List(Float64)) shaped (4096, 5) per
row: 4096 positions x 5 channels (A, C, G, T, ATAC). This script flattens that
into five columns -- attrs_1..attrs_4 (DNA bases) and attrs_atac -- where each
cell holds the 4096-length channel as a JSON array string (csv-safe).

Memory note: the file's attributions decompress to ~14 GB for a typical sample
and there is usually a single parquet row group, so we stream over it in
row-batches via pyarrow.iter_batches and pipe each batch's CSV through pigz
(falling back to gzip / python gzip). Nothing larger than one batch is ever held
in memory at once.

Round-trip in pandas/polars:
    df["attrs_1"].map(json.loads)        # -> list[float]
    pl.col("attrs_1").str.json_decode()  # -> List(Float64)

Usage:
    /data1/home/wjohnson/opt/miniforge3/bin/python parquet_to_csv_gz.py \
        INPUT.parquet [-o OUTPUT.csv.gz] [--batch-size N] [--threads N] \
        [--level 6] [--limit N]
"""

import argparse
import contextlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Columns to read (mirrors call_seqlets.py; linear_512_output is intentionally
# dropped). chr_name is renamed to chr on output.
READ_COLUMNS = [
    "chr_name", "start", "end", "cell_line",
    "targets", "predicted", "weights",
    "probabilities", "attributions",
]

CONTEXT_LENGTH = 4096   # positions per row
N_CHANNELS = 5          # A, C, G, T, ATAC
ATTR_OUT_NAMES = ["attrs_1", "attrs_2", "attrs_3", "attrs_4", "attrs_atac"]


@contextlib.contextmanager
def compressed_writer(out_path: Path, threads: int, level: int):
    """Yield a writable binary stream whose bytes land gzip-compressed at
    out_path. Prefers pigz (parallel), then gzip CLI, then python gzip."""
    if shutil.which("pigz"):
        cmd = ["pigz", "-c", f"-{level}", "-p", str(threads)]
    elif shutil.which("gzip"):
        cmd = ["gzip", "-c", f"-{level}"]
    else:
        cmd = None

    if cmd is not None:
        print(f"Compressing with: {' '.join(cmd)}", flush=True)
        out_fh = open(out_path, "wb")
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=out_fh)
        try:
            yield proc.stdin
        finally:
            proc.stdin.close()
            ret = proc.wait()
            out_fh.close()
            if ret != 0:
                raise RuntimeError(f"Compressor exited with code {ret}")
    else:
        import gzip
        print("pigz/gzip not found; using python gzip (slower).", flush=True)
        gz = gzip.open(out_path, "wb", compresslevel=level)
        try:
            yield gz
        finally:
            gz.close()


def attributions_to_array(batch) -> np.ndarray:
    """(n, 4096, 5) float64 from the nested-list `attributions` column.

    Flattens both list levels in arrow (respecting offsets, no Python objects)
    then reshapes -- so the channel order matches reshape_attributions_fast in
    call_seqlets.py.
    """
    attr = batch.column("attributions")
    flat = pc.list_flatten(pc.list_flatten(attr)).to_numpy(zero_copy_only=False)
    expected = batch.num_rows * CONTEXT_LENGTH * N_CHANNELS
    if flat.size != expected:
        raise ValueError(
            f"attributions has {flat.size} values; expected {expected} "
            f"({batch.num_rows} rows x {CONTEXT_LENGTH} x {N_CHANNELS}). "
            "A row may not be 4096x5 -- check the source parquet."
        )
    return flat.reshape(batch.num_rows, CONTEXT_LENGTH, N_CHANNELS)


def json_encode_channel(arr2d: np.ndarray, name: str) -> pl.Series:
    """(n, 4096) float64 -> Utf8 Series of JSON arrays, one per row.

    Float -> Utf8 cast uses polars' shortest round-trippable representation, so
    json.loads / str.json_decode recover the values exactly.
    """
    s = pl.Series(name, np.ascontiguousarray(arr2d))  # Array(Float64, 4096)
    encoded = pl.lit("[") + s.cast(pl.List(pl.Utf8)).list.join(",") + pl.lit("]")
    return pl.select(encoded.alias(name)).to_series()


def process_batch(batch) -> pl.DataFrame:
    """One arrow RecordBatch -> output DataFrame with expanded attr columns."""
    reshaped = attributions_to_array(batch)
    scalar = batch.select([c for c in READ_COLUMNS if c != "attributions"])
    out = pl.from_arrow(scalar).rename({"chr_name": "chr"})
    out = out.with_columns([
        json_encode_channel(reshaped[:, :, k], ATTR_OUT_NAMES[k])
        for k in range(N_CHANNELS)
    ])
    return out


def convert(in_path: Path, out_path: Path, batch_size: int,
            threads: int, level: int, limit: int | None) -> None:
    pf = pq.ParquetFile(in_path)
    total_rows = pf.metadata.num_rows
    if limit is not None:
        total_rows = min(total_rows, limit)
    n_batches = (total_rows + batch_size - 1) // batch_size
    print(f"Input : {in_path}  ({total_rows:,} rows)", flush=True)
    print(f"Output: {out_path}", flush=True)
    print(f"Batches: {n_batches} x up to {batch_size} rows", flush=True)

    written = 0
    t0 = time.time()
    with compressed_writer(out_path, threads, level) as sink:
        first = True
        for batch in pf.iter_batches(batch_size=batch_size, columns=READ_COLUMNS):
            if limit is not None and written >= limit:
                break
            if limit is not None and written + batch.num_rows > limit:
                batch = batch.slice(0, limit - written)

            out = process_batch(batch)
            out.write_csv(sink, include_header=first)
            first = False

            written += out.height
            elapsed = time.time() - t0
            rate = written / elapsed if elapsed else 0
            print(f"  {written:,}/{total_rows:,} rows "
                  f"({100 * written / total_rows:.1f}%) | "
                  f"{rate:,.0f} rows/s | {elapsed:,.0f}s", flush=True)

    print(f"Done. Wrote {written:,} rows in {time.time() - t0:,.0f}s -> "
          f"{out_path} ({out_path.stat().st_size / 1e9:.2f} GB)", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="Input .parquet file")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output .csv.gz (default: input with .csv.gz suffix)")
    p.add_argument("--batch-size", type=int, default=2000,
                   help="Rows per batch (default 2000; lower if memory-tight)")
    p.add_argument("--threads", type=int, default=min(8, len(__import__('os').sched_getaffinity(0))),
                   help="pigz compression threads (default min(8, cpus))")
    p.add_argument("--level", type=int, default=6,
                   help="gzip compression level 1-9 (default 6)")
    p.add_argument("--limit", type=int, default=None,
                   help="Only convert the first N rows (for testing)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.input.is_file():
        sys.exit(f"Input not found: {args.input}")
    out_path = args.output
    if out_path is None:
        out_path = args.input.with_suffix("").with_suffix(".csv.gz") \
            if args.input.suffix == ".parquet" \
            else args.input.with_name(args.input.name + ".csv.gz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    convert(args.input, out_path, args.batch_size, args.threads, args.level, args.limit)


if __name__ == "__main__":
    main()
