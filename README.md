# DL-based Drug Repurposing Framework

This repository contains code and configuration to run a deep learning (DL)–based framework for cell type–specific drug repurposing, originally applied to Leigh syndrome using single-cell and bulk transcriptomic data.

---

## 1. System requirements

### 1.1 Operating system and environment

This framework is designed to run on a high-performance computing (HPC) cluster with:

- **Operating system:** Linux
  - Tested on typical HPC distributions (e.g. Red Hat Enterprise Linux )
- **Job scheduler:** [Slurm](https://slurm.schedmd.com/) with `sbatch` / `srun` available
- **Container runtime:** Singularity or Apptainer
  - Tested with Singularity/Apptainer ≥ 3.8 (adapt to your cluster)

A “normal desktop” without Slurm/Singularity is not the primary supported mode. Local execution is possible in principle with `singularity exec`, but all scripts and launchers are written for an HPC + Slurm environment.

### 1.2 Software dependencies

Most software dependencies (Python, PyTorch, DL libraries, etc.) are packaged inside the Singularity image:

- `pytorch_gnn.sif` – container image with PyTorch + graph and neural network dependencies
- `my_torch_SIGN_VAE.model` – trained SIGN-VAE model weights

External tools expected on the host system:

- `bash`
- `Slurm` (`sbatch`, `srun`)
- `Singularity` or `Apptainer`
- Standard core utilities (`cp`, `mkdir`, `tar`, etc.)

The provided Slurm launchers internally call commands of the form:

```bash
singularity run --no-home \
  -B <project_path> \
  -B /usr/bin/sbatch:/usr/bin/sbatch \
  -B /usr/bin/srun:/usr/bin/srun \
  -B /usr/lib64/slurm:/usr/lib64/slurm \
  -B /var/spool/slurm:/var/spool/slurm \
  -B /etc/passwd:/etc/passwd \
  -B /etc/group:/etc/group \
  -B /usr/bin/munge:/usr/bin/munge \
  -B /usr/lib64/libmunge.so.2:/usr/lib64/libmunge.so.2 \
  -B /usr/lib64/libmunge.so.2.0.0:/usr/lib64/libmunge.so.2.0.0 \
  -B /var/run/munge:/var/run/munge \
  pytorch_gnn.sif python "$@"
```

## 2. Installation guide

### 2.1 Clone the repository

git clone https://github.com/temporary-code-2025/manuscript_code.git
cd manuscript_code

### 2.2 Download required artifacts

Singularity container:

```  wget -O pytorch_gnn.sif https://zenodo.org/records/17625383/files/pytorch_gnn.sif```

Trained SIGN-VAE model:

```  wget -O my_torch_SIGN_VAE.model https://zenodo.org/records/17635308/files/my_torch_SIGN_VAE.model```

### 2.3 Typical install time on a normal desktop

On a typical desktop or login node with a reasonable network connection:

    Cloning this repository: < 1 minute

    Downloading pytorch_gnn.sif (~8 GB): a few minutes to tens of minutes (bandwidth-dependent)

    Downloading my_torch_SIGN_VAE.model: seconds to ~1 minute

No additional Python packages need to be installed on the host if you use the Singularity image.

## 3. Demo

### 3.1 Demo input data

For the Leigh use case, the main prediction outputs that are later ranked are:

    drug_cl6cl7_ranked.csv
    drug_cl4cl7_ranked.csv

If a small demo dataset is included, its location can be documented here (e.g. data/demo/).

### 3.2 Running the demo pipeline

# 1. Make embeddings
sbatch Py_launcher_make_embeddings.sh make_embeddings.py

# 2. Run VAE, FNN, and drug scoring.
#    Random perturbation is deactivated for reproducibility.
#    Original random perturbation outputs are stored under randoms/
#    For a new application, you can activate it and results will be saved under randoms_local/
sbatch Py_launcher_run_FFNs.sh run_FFNs.py ./ .005 0.0 1500 5 0.7 0 2.0 0.1 ./ 3

After the Slurm jobs complete, you should have predOut_<tag>_drug_pvalues.csv files. Then rank drugs by BES:

# 3. Manual ranking step for Leigh comparisons
singularity run --no-home -B <project_path> pytorch_gnn.sif \
  python rank_drugs_by_BES.py predOut_cl6cl7_drug_pvalues.csv

singularity run --no-home -B <project_path> pytorch_gnn.sif \
  python rank_drugs_by_BES.py predOut_cl4cl7_drug_pvalues.csv

This will generate:

    drug_cl6cl7_ranked.csv
    drug_cl4cl7_ranked.csv

### 3.3 Expected runtime on a normal-sized HPC job

On a modest CPU-only node (e.g. 1–4 cores, 16–32 GB RAM, no GPU):

    Embedding generation (make_embeddings.py): 15 minutes to ~1 hour (dataset-dependent)

    VAE/FNN training + scoring (run_FFNs.py): 1-5 minutes without randomization, 30min to an hour with randomization

    Ranking (rank_drugs_by_BES.py): a few minutes

These times are approximate and will vary with input size and cluster load.

## 4. Instructions for use (your own data)

### 4.1 Prepare your input

   1. An initial gene expression state

   2. Differentially expressed transcription factors (DETFs) between these states.
   
### 4.2 Follow the steps in 3.2

