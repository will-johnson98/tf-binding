#!/usr/bin/env python3
# WAJ 2026-06-29

import os
import polars as pl
import numpy as np

#########################
MODEL = "AR"
CELL_LINE = "VCAP"
#########################

ROOT_DIR = "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding"
PROJ_DIR = os.path.join(ROOT_DIR, "src/analysis/interpretability")

def wrangle_attributions(attributions):
    attrs = np.array(attributions.to_list())
    attrs = attrs.reshape(-1, 4096, 5)
    atac_attrs = attrs[..., 4]
    attrs = attrs[..., :4]
    neg_attrs = np.clip(attrs, None, 0)
    pos_attrs = np.clip(attrs, 0, None)
    pos_sums = np.array([mat.sum(axis=1) for mat in pos_attrs])
    neg_sums = np.array([mat.sum(axis=1) for mat in neg_attrs])
    return pos_sums, neg_sums, atac_attrs


def mutate_df_attrs(df: pl.DataFrame):
    pos_sums, neg_sums, atac_attrs = wrangle_attributions(df['attributions'])

    df = df.with_columns(pl.Series("pos_attrs", pos_sums.tolist(), dtype=pl.List(pl.Float64)))
    df = df.with_columns(pl.Series("neg_attrs", neg_sums.tolist(), dtype=pl.List(pl.Float64)))
    df = df.with_columns(pl.Series("atac_attrs", atac_attrs.tolist(), dtype=pl.List(pl.Float64)))
    df = df.filter(
        (pl.col("probabilities") <= 0.01) | (pl.col("probabilities") >= 0.99)
    )

    return df



def main():
    parquet_file = f"{ROOT_DIR}/data/processed_results/{MODEL}_{CELL_LINE}_processed.parquet"
    output_path = f"{PROJ_DIR}/data/attribution_matrices/{MODEL}_{CELL_LINE}_attrs.parquet"
    compression = "zstd"

    print("Loading in parquet file...")
    df = pl.read_parquet(
        parquet_file,
        columns=['chr_name', 'start', 'end', 'targets', 'predicted', 'probabilities', 'attributions'],
        parallel="columns",
        use_statistics=True,
        memory_map=True,
    )

    print("Modifying dataframe...")
    df = mutate_df_attrs(df)

    print(f"Saving {MODEL}_{CELL_LINE} dataframe to file (parquet.zst)...")
    df.write_parquet(output_path,
                     compression=compression,
                     row_group_size=100_000)
    

if __name__ == "__main__":
    main()
