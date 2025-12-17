import os, sys, re, glob, math, fcntl, pickle, argparse
import itertools
from itertools import product
import numpy as np
import joblib
import pandas as pd
from model_library import *
Dtype = 'float32'

# Usage:
# Input must match: predOut_<TAG>_drug_pvalues.csv
# Output will be:  drug_<TAG>_ranked.csv

if len(sys.argv) < 2:
    raise SystemExit("Usage: python script.py predOut_<TAG>_drug_pvalues.csv")

infile = sys.argv[1]
base = os.path.basename(infile)

prefix = "predOut_"
suffix = "_drug_pvalues.csv"
if not (base.startswith(prefix) and base.endswith(suffix)):
    raise ValueError(f"Input filename must look like 'predOut_<TAG>_drug_pvalues.csv', got: {base}")

tag = base[len(prefix):-len(suffix)]  # the part between predOut_ and _drug_pvalues.csv
outfile = f"drug_{tag}_ranked.csv"
df = pd.read_csv(infile, sep='\t')

# Compute BES
df['BES'], df['rank_BES'] = compute_BES_score(
    pemp=df['pemp_JaRs'].values,
    hypergeo_pval=df['hyergeoPvalue'].values,
    rank_hy=df['rank_hy_z_scores'].values
)

# take top 2.5%
b = df[df['rank_BES'] <= np.round(df.shape[0] * 0.025)]
b = b.sort_values(['rank_BES'], ascending=[True])

# change column order
b.rename({'Functional.Annotation.Functional.Annotation': 'Functional.Annotation'}, axis=1, inplace=True)
b = b[['Compound','BES','rank_BES','Functional.Annotation','Functional.Annotation.Category',
       'Targeted','Untargeted','pemp_JaRs','hyergeoPvalue','hy_z_scores','rank_hy_z_scores']]

b.to_csv(outfile, sep='\t', index=False, quoting=2)
print(f"Wrote: {outfile}")

