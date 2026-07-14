import os
import subprocess

base_dir = "/data1/datasets_1/human_cistrome/chip-atlas/peak_calls/tfbinding_scripts/tf-binding/src/preprocessing/chip"

# List of transcription factors
tf_list = [
    "ASCL1"
    #"FOXA1",  "NEUROD1", "ASCL1", "RB1", "HOXB13", "E2F1", "E2F2", "CTCF"
    #"NR3C1", "FOXA2", "SOX2", "ESR1", "JUN", "JUNB", "JUND", "FOSL1", "FOSL2"
    #"E2F1", "E2F2","HNF4G", "FOS" 
    #"FOSB", "MYC","TEAD1", "TEAD4", "TAZ", "ERG", "GATA2", "GATA4", "POU2F1", "REST"
    # "ASCL1"
    #  "YAP1", "POU5F1"
    #  "NFKB", "YY1", "ONECUT2", "ARNTL"

]
# tf_list = [
#     'ASCL1', 'FOXA1', 'NR3C1', 'HDAC3', 'HOXB13', 'HDAC1', 'TP53', 'TRIM24', 'LEO1', 'SMARCA5', 'INO80', 'SMARCA4', 
#     'SMARCA2', 'CHD1', 'CHD4', 'SOX9', 'CREB1', 'CBX3', 'XRCC6', 'CTCF', 'H3', 'ARNTL', 'H2A.Z', 'H2A.ZK4K7K11AC', 
#     'MED1', 'POU2F1', 'H3AC', 'EZH2', 'SUZ12', 'BRD4', 'GRHL2', 'LMNB1', 'PIAS1', 'RELA', 'VDR', 'E2F1', 'ARID1A', 
#     'TLE3', 'TRIM28', 'ETV1', 'H4AC', 'ONECUT2', 'TCF7L2', 'KDM1A', 'MYC', 'O-GLCNAC', 'HIF1A', 'REST', 'EP300', 
#     'MYBL2', 'CTBP1', 'CTBP2', 'MEN1', 'RUNX1', 'STAT2', 'MRE11A', 'POU5F1', 'FOXA2', 'CCNT1', 'TET2', 'YY1', 
#     'TEAD1', 'YAP1', 'EED', 'H2AK119UB', 'RNF2', 'BMI1', 'ETS1', 'ELF1', 'ELK4', 'GABPA', 'JUND', 'H4K20ME3', 'NRF1', 
#     'NFE2L2', 'MAX', 'WDR5', 'MGA', 'MNT', 'ZFX', 'ZNF711', 'RUNX2', 'RB1', 'WT1', 'EWSR1', 'TFAP2C', 'ESR1', 'STAT3', 
#     'RAD21', 'FOS', 'JUN', 'ERG', 'GATA3', 'SPDEF', 'CTR9', 'ZNF331', 'CREBBP', 'AFF4', 'TAF1', 'H3R17ME2', 'JMJD6', 
#     'ZEB1', 'ZNF143', 'KMT2D', 'KDM6A', 'STAT1', 'FOXM1', 'FOSL2', 'HDAC2', 'EGR1', 'TEAD4', 'TCF12', 'SIN3A', 'CEBPB', 
#     'KLF4', 'RXRA', 'BRDU', 'MBD3', 'H4K8AC', 'KMT2A', 'KMT2B', 'ARNT', 'TFAP2A', 'SREBF1', 'HDGF', 'ZBTB33', 'MAZ', 
#     'ELK1', 'RCOR1', 'FOXK2', 'MLLT1', 'SP1', 'ESRRA', 'PKNOX1', 'ZBTB11', 'H3.3', 'H4K20ME1', 'NR5A2', 'KLF9', 
#     'NFRKB', 'ZBTB1', 'ZNF24', 'TARDBP', 'RFX5', 'SUMO2', 'BRD3', 'SMC1A', 'CTNNB1', 'H2AK120UB', 'AGO1', 'CBX5', 
#     'H4R3ME2', 'MAFK', 'CLOCK', 'ZMYND8', 'JUNB', 'CDK9', 'BRD2', 'BCOR', 'ZBTB7A', 'DOT1L', 'CBX8', 'AFF1', 'CBX2', 
#     'BAP1', 'CDK12', 'H4K16AC', 'H4K5AC', 'KAT8', 'KANSL3', 'NIPBL', 'SMC3', 'PRDM1', 'ZSCAN20', 'ZNF302', 'ZNF624', 
#     'ASXL1', 'EHMT2', 'EPAS1', 'PAF1'
# ]


# Directory to store the qsub scripts
output_dir = "qsub_scripts"
os.makedirs(output_dir, exist_ok=True)

# Directory to store the output logs
log_dir = "qsub_logs"
os.makedirs(log_dir, exist_ok=True)

# Generate a qsub script for each transcription factor
for tf in tf_list:
    script_content = f"""#!/bin/bash
#$ -N {tf}
#$ -pe smp 6
#$ -l mem_free=16G
#$ -cwd 
#$ -o {base_dir}/{log_dir}/{tf}.out
#$ -j y
#$ -V


source ~/.bashrc

echo "activating conda environment"
conda activate pterodactyl

cd {base_dir}

# running basic_tf.sh
./basic_tf.sh {tf}

"""

    script_path = os.path.join(output_dir, f"{tf}.sh")
    with open(script_path, "w") as script_file:
        script_file.write(script_content)

# Create a master script to submit all qsub scripts
master_script_content = "#!/bin/bash\n\n"
for tf in tf_list:
    master_script_content += f"qsub {os.path.abspath(output_dir)}/{tf}.sh\n"

master_script_path = os.path.join(output_dir, "submit_all.sh")
with open(master_script_path, "w") as master_script_file:
    master_script_file.write(master_script_content)

# Make the master script executable
# os.chmod(master_script_path, 0o755)

print(f"Generated {len(tf_list)} qsub scripts in {output_dir}")
print(f"Master script to submit all qsub scripts created at {master_script_path}")

# Submit the master script
subprocess.run(["bash", master_script_path])
