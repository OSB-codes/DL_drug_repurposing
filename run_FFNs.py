import os, sys, re, time, glob, subprocess, fcntl, pickle, argparse, itertools
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset, WeightedRandomSampler
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
import umap.umap_ as umap
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from model_library import *
Dtype = 'float32'

script = sys.argv[0]

initial_hidden_size=1500
num_layers=5
decay_factor=0.7
Nodes = open('nodeList_adjacency_PKN_sorted_GNN_nodeLabel.csv', 'r').read().splitlines()
num_nodes = len(Nodes)

# check if GPUs are used
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
sys.stdout.flush()
if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

print('Device')
print(device)

# Input data
input_batch1 = torch.load('query_data_cl6cl7.pt')
input_batch2 = torch.load('query_data_cl4cl7.pt')
input_batch = torch.cat([input_batch1, input_batch2],axis=0)
nSam = 2
input_batch = input_batch.to(device)
outnames = ['cl6cl7','cl4cl7']

# VAE
weightfile_VAE = 'my_torch_SIGN_VAE.model'
initial_hidden_size_VAE = 1500
num_layers_VAE = 5
dropout_VAE = 0.0
latent_dimension_VAE = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE_decrease(input_batch.shape[1], initial_hidden_size_VAE, num_layers_VAE, dropout_VAE, latent_dimension_VAE).to(device)
model.load_state_dict(torch.load(weightfile_VAE, map_location=device))
with torch.no_grad():
  mu, logvar = model.encode(input_batch)
  Xdec = model.decode(mu)

# FNN 
def deactivate_ema(model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.track_running_stats = False
            m._saved_running_mean, m.running_mean = m.running_mean, None
            m._saved_running_var, m.running_var = m.running_var, None

weightfiles = ['ffn_model_weights_1.pth','ffn_model_weights_2.pth','ffn_model_weights_3.pth']
MODELS = []
for i in range(len(weightfiles)):
  model = FFN_decay(Xdec.shape[1], initial_hidden_size, 3, num_layers, decay_factor).to(device)
  deactivate_ema(model)
  model.eval()
  checkpoint = torch.load(weightfiles[i])
  model.load_state_dict(checkpoint['model_state_dict'])
  MODELS.append(model)

# loss function
loss_fn = nn.CrossEntropyLoss()

def runDrugPrediction(preds, Nodes, outdir, infix, nSam, outnames, runDrugRanking):
    preds = preds.view(-1, num_nodes)
    split_tensors = torch.chunk(preds, nSam, dim=0)
    for k in range(len(split_tensors)):
      part = split_tensors[k]
      P = torch.mode(part,0)[0]
      sigPs = [Nodes[i]+"__act" for i in torch.where(P==1)[0]] + [Nodes[i]+"__inh" for i in torch.where(P==2)[0]] + [Nodes[i]+"__act" for i in torch.where(P==3)[0]] + [Nodes[i]+"__inh" for i in torch.where(P==3)[0]]; len(sigPs)
      outname ="predOut_"+outnames[k]+infix+'.txt'
      exportPredictedProteins(outdir+"/"+outname,sigPs)
      # run drug ranking
      if runDrugRanking:
        wait_for_file(outdir,outname)
        print(f"Running drug prediction",flush=True)
        sbatch_command1 = f"sbatch R_launcher_compute_drug_pvalues.sh {outname}"
        result1 = subprocess.run(sbatch_command1, shell=True, capture_output=True, text=True)

# random permutation (deactivated)
'''
dir_path = Path('randoms_local')
os.makedirs(dir_path, exist_ok=True)
repNum = 100
for k in range(repNum):
    test_correct = 0; test_total = 0; all_test_preds = []
    PREDS = torch.empty([0, Xdec.shape[0]])
    shuffled_x = Xdec[torch.randperm(Xdec.size(0))]
    for i in range(len(MODELS)):
        with torch.no_grad():
          logits = MODELS[i](shuffled_x)
          pred = logits.argmax(dim=1)
        PREDS = torch.cat([PREDS, pred.view(1,-1)],dim=0)
    P = torch.mode(PREDS,0)[0]
    P = P.to(torch.int)
    infix = "__"+str(k)
    runDrugPrediction(P, Nodes, str(dir_path), infix, nSam, outnames, False)
'''

# target prediction
PREDS = torch.empty([0, Xdec.shape[0]])
for i in range(len(MODELS)):
  with torch.no_grad():
    logits = MODELS[i](Xdec)
    preds = logits.argmax(dim=1)
  PREDS = torch.cat([PREDS, preds.view(1,-1)],dim=0)

P = torch.mode(PREDS,0)[0]
P = P.to(torch.int)
runDrugPrediction(P, Nodes, "./", "", nSam, outnames, True)




