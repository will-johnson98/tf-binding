# TF Binding Prediction

## Setup

### Environment Installation

From the base directory, create the conda environment:

```bash
conda env create -f environment.yml
```

Activate the environment

```bash
conda activate pterodactyl
```

Install the Pterodactyl package:

```bash
pip install --no-deps -e .
```

Configure AWS credentials:

```bash
aws configure
```

## Finetuning

To finetune a model for a specific transcription factor:

1. Navigate to the training directory:
   ```bash
   cd src/training
   ```

2. Ensure your conda environment is activated:
   ```bash
   conda activate pterodactyl
   ```

3. Run the finetuning script with the desired TF and cell line:
   ```bash
   python tf_finetuning.py --tf_name AR --cell_line 22Rv1
   ```

This process takes anywhere from a couple hours to days depending on the amount of training cell lines. You can monitor progress in SageMaker training at [AWS SageMaker Console](https://016114370410-4y4js2yi.us-west-2.console.aws.amazon.com/sagemaker/home?region=us-west-2#/jobs).

## Inference

To run inference with your finetuned models:

1. Navigate to the inference directory:
   ```bash
   cd src/inference
   ```

2. Ensure your desired model is listed in `models.json`. Example:
   ```json
   {
     "FOXA1": "s3://tf-binding-sites/finetuning/results/output/FOXA1-22Rv1-2025-02-15-18-14-22-064/output/model.tar.gz",
     "HOXB13": "s3://tf-binding-sites/finetuning/results/output/HOXB13-22Rv1-2025-02-18-23-29-14-654/output/model.tar.gz"
   }
   ```
   
   Note: The paths contain the SageMaker training job names (e.g., `FOXA1-22Rv1-2025-02-15-18-14-22-064`).

3. Run inference (recommend using `screen` or `nohup` as this will take several hours):
   ```bash
   bash tf_inference.sh --atac_dir /data1/datasets_1/human_prostate_PDX/processed/ATAC_merge/LuCaP_145_1 --models FOXA1,HOXB13 --parallel
   ```


   This will output a parquet file in `data/processed_results` which we can use for downstream analysis (some example scripts can be found in `src/inference/analysis`)




# to add new cell lines to training