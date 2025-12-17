#!/bin/bash -l

#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH -t 1:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24

singularity run --no-home -B  <project_path> -B /usr/bin/sbatch:/usr/bin/sbatch -B /usr/bin/srun:/usr/bin/srun -B /usr/lib64/slurm:/usr/lib64/slurm -B /var/spool/slurm:/var/spool/slurm -B /etc/passwd:/etc/passwd -B /etc/group:/etc/group -B /usr/bin/munge:/usr/bin/munge -B /usr/lib64/libmunge.so.2:/usr/lib64/libmunge.so.2 -B /usr/lib64/libmunge.so.2.0.0:/usr/lib64/libmunge.so.2.0.0 -B /var/run/munge:/var/run/munge pytorch_gnn.sif python "$@"

### Usage:
# sbatch Py_launcher_run_FFNs.sh run_FFNs.py ./ .005 0.0 1500 5 0.7 0 2.0 0.1 ./  3










