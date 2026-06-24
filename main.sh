#!/bin/bash
#SBATCH --job-name=main
#SBATCH --account=h2020deciderficarra
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/main_%j.out    
#SBATCH --error=logs/main_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=1:00:00

source /work/tesi_vmelotti/miniconda3/etc/profile.d/conda.sh
conda activate model0_env

#echo sequence
#python -u main_train.py --mode sequence  --task is_onco --checkpoint checkpoints/COMPLETE_sequence_model.pt

python -u main_test.py \
--mode sequence \
--task is_onco \
--checkpoint checkpoints/COMPLETE_sequence_model.pt \
--output /homes/vmelotti/project/src/out/seq_decider_pos.tsv \
--df /homes/vmelotti/project/data/raw/data_decider_positive_WITH_SEQUENCE.pkl \
--seq_embs /homes/vmelotti/project/data/embeddings/seq_emb_DECIDER.npz

#python -u main_test.py \
#--mode sequence \
#--task is_onco \
#--checkpoint checkpoints/COMPLETE_sequence_model.pt \
#--output /homes/vmelotti/project/src/out/seq_SRR.tsv \
#--df /homes/vmelotti/project/data/raw/data_SRR.pkl \
#--seq_embs /homes/vmelotti/project/data/embeddings/sequence_SRR.pkl.npz


#python -u main_test.py --mode sequence --reduced_db --task is_onco --checkpoint checkpoints/full_sequence_model.pt --output /homes/vmelotti/project/src/out/sequence.tsv
#echo structure
#python -u main_train.py --mode structure  --checkpoint checkpoints/full_structure_model.pt 
#python -u main_test.py --mode structure  --checkpoint checkpoints/full_structure_model.pt --output /homes/vmelotti/project/src/out/structure.tsv
#echo ensemble concat
#python -u main_train.py --mode ensemble  --checkpoint checkpoints/full_ens_conc_model.pt 
#python -u main_test.py --mode ensemble  --checkpoint checkpoints/full_ens_conc_model.pt --output /homes/vmelotti/project/src/out/ens_conc.tsv
#echo ensemble voting
#python -u main_train.py --mode ensemble_soft --checkpoint checkpoints/full_ens_vot_model.pt 
#python -u main_test.py --mode ensemble_soft --checkpoint checkpoints/full_ens_vot_model.pt --output /homes/vmelotti/project/src/out/ens_vote.tsv
