#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=96:00:00
#SBATCH --job-name=inference
#SBATCH --mem=128G
#SBATCH --partition=177huntington
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-8%2
#SBATCH --dependency=singleton

CACHE_DIR="/scratch/shaib.c/"

export HF_DATASETS_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR

# use 177huntington 
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate mds_env
module load cuda/11.8

   COMMANDS=(

    "python3 generate_train.py --dataset cochrane --model_id mistralai/Mistral-7B-Instruct-v0.3"
    "python3 generate_train.py --dataset cochrane --model_id google/gemma-2-9b-it"
    "python3 generate_train.py --dataset cochrane --model_id meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    "python3 generate_train.py --dataset rotten_tomatoes --model_id mistralai/Mistral-7B-Instruct-v0.3"
    "python3 generate_train.py --dataset rotten_tomatoes --model_id google/gemma-2-9b-it"
    "python3 generate_train.py --dataset rotten_tomatoes --model_id meta-llama/Meta-Llama-3.1-8B-Instruct"
   )

   # Get the command for the current SLURM task ID
   COMMAND=${COMMANDS[$SLURM_ARRAY_TASK_ID]}
   echo "Running command: $COMMAND"
   eval $COMMAND
