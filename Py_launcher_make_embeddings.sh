#!/bin/bash -l

#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH -t 1:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24

singularity run --no-home -B <project_path> pytorch_gnn.sif python ${1} ${2}

## Usage:
# sbatch Py_launcher_make_embeddings.sh make_embeddings.py

