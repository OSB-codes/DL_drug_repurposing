#!/bin/bash -l

#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH -t 1:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3

Rscript compute_drug_pvalues.R  ${1} 

