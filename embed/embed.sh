#!/bin/bash
#SBATCH --job-name=embed
#SBATCH --account=h2020deciderficarra
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/main_%j.out    
#SBATCH --error=logs/main_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=5:00:00
source /work/tesi_vmelotti/miniconda3/etc/profile.d/conda.sh
conda activate model0_env

cd /homes/vmelotti/project/src/embed

python -u embed.py \
    --infile /homes/vmelotti/project/data/raw/data_SRR.pkl \
    --outfile /homes/vmelotti/project/data/embeddings/sequence_SRR.pkl


#python -u embed.py \
#    --infile /homes/vmelotti/project/data/raw/data_decider_positive_WITH_SEQUENCE.pkl \
#    --outfile /homes/vmelotti/project/data/embeddings/sequence_decider_positive.pkl \
#    --sequence_column sequence