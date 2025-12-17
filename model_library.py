import os, sys, time, glob, fcntl, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_sparse import SparseTensor
import pytorch_lightning as pl
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from igraph import *
from torch_geometric.data import Batch
from torch_geometric.data import DataLoader as gDataLoader
from torch_geometric.utils import (
    to_undirected, dropout_adj, add_self_loops, degree
)
Dtype = 'float32'
device = 'cpu'

def get_adj(row, col, value, N, norm=False, asymm_norm=False, set_diag=True, remove_diag=False):
    print(set_diag)
    adj = SparseTensor(row=row, col=col, value=value, sparse_sizes=(N, N))
    if set_diag:
        print('... setting diagonal entries')
        adj = adj.set_diag()
    elif remove_diag:
        print('... removing diagonal entries')
        adj = adj.remove_diag()
    else:
        print('... keeping diag elements as they are')
    if norm==True:
      if not asymm_norm:
        print('... performing symmetric normalization')
        deg = adj.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
      else:
        print('... performing asymmetric normalization')
        deg = adj.sum(dim=1).to(torch.float)
        deg_inv = deg.pow(-1.0)
        deg_inv[deg_inv == float('inf')] = 0
        adj = deg_inv.view(-1, 1) * adj
    else:
        print('... no normalization')
    #
    adj = adj.to_scipy(layout='csr')
    #
    return adj

def expandAndNormalizeEmbedding(gD, normalization):
  num_nodes = gD[0]['op_embedding'][0].shape[0]
  Xtrain = torch.zeros(len(gD)*gD[0]['op_embedding'][0].shape[0], (2*len(gD[0]['op_embedding'])+len(gD[0]['properties'])) )
  print('All feature train shapes:')
  print(Xtrain.shape)
  for i in range(len(gD)):
    start = i*num_nodes
    end = (i+1)*num_nodes
    op_dict = gD[i]
    Xtrain[start:end,:] = torch.cat([torch.cat(op_dict['op_embedding'],dim=1), torch.cat(op_dict['properties'],dim=1)], axis=1)
  Xtrain = Xtrain.float()
  # aggregation: sum, mean, max, min, median, std, 25%, 75%, difference between sucssessive columns
  Xtrain = torch.cat((Xtrain, torch.sum(Xtrain,1).view(-1,1), torch.mean(Xtrain,1).view(-1,1), torch.max(Xtrain,1).values.view(-1,1), torch.min(Xtrain,1).values.view(-1,1), torch.median(Xtrain,1).values.view(-1,1), torch.quantile(Xtrain, 0.25, dim=1, keepdim=True), torch.quantile(Xtrain, 0.75, dim=1, keepdim=True), (Xtrain[:,torch.arange(4,22,2)] - Xtrain[:,torch.arange(2,20,2)]), (Xtrain[:,torch.arange(5,22,2)] - Xtrain[:,torch.arange(3,20,2)]) ) ,1)
  # polynomical features:
  pf = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
  Xtrain = torch.from_numpy(pf.fit_transform(Xtrain)).float()
  print('feature synthesis done',flush=True)
  print('train shapes:')
  print(Xtrain)
  print(Xtrain.shape)
  if normalization=='minMax':
    mi,ma,r = torch.load('GNN_normalization_factors_minMax_{}.pt'.format(str(Xtrain.shape[1])) )
    Xtrain = (Xtrain - mi)/r
  Xtrain = Xtrain.float()
  return Xtrain

def makeX2(X1, T2, Nodes):
  gen = [i.split('__', 1)[0] for i in T2]
  moa = [1 if i.split('__', 1)[1]=='act' else 2 for i in T2]
  X2 = np.zeros(len(Nodes))
  X2[[Nodes.index(i) for i in gen]] = moa
  X2 = np.tile(X2,(X1.shape[0],1))
  if len(X1.shape)==1: X1.shape=(1,X1.shape[0])
  if len(X2.shape)==1: X2.shape=(1,X2.shape[0])
  print(X1.shape); print(X2.shape)
  X1dim = X1.shape[1]
  X2dim = X2.shape[1]
  return X1, X2, X1dim

class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        assert self.x.size(0) == self.y.size(0)
    def __len__(self):
        return self.x.size(0)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def prepareRealdataDataLoader(X1_file, DETF_file, removeColFile, normalization):
  directed_asymm_norm = True
  directed_set_diag = False
  directed_remove_diag = False
  num_propagations = 5
  num_node_features = 2
  num_hops = 11
  Nodes = open('nodeList_adjacency_PKN_sorted_GNN_nodeLabel.csv', 'r').read().splitlines()
  num_nodes = len(Nodes)
  # Graph
  PKN = np.loadtxt("adjacency_PKN.csv.gz", delimiter=",",dtype=Dtype)
  g = Graph.Weighted_Adjacency(PKN)
  PKN = PKN.T
  e = np.where(PKN!=0)
  edge_index = torch.cat((torch.unsqueeze(torch.from_numpy(e[0]),0), torch.unsqueeze(torch.from_numpy(e[1]),0) ), axis=0)
  np.sum(PKN[np.array(edge_index[0]),np.array(edge_index[1])]==0)
  # Reactome count
  RE = np.loadtxt("adjacency_Reactome_count.csv.gz", delimiter=",",dtype=Dtype)
  e2 = np.where(RE!=0)
  edge_index_RE = torch.cat((torch.unsqueeze(torch.from_numpy(e2[0]),0), torch.unsqueeze(torch.from_numpy(e2[1]),0) ), axis=0)
  np.sum(RE[np.array(edge_index_RE[0]),np.array(edge_index_RE[1])]==0)
  DETFs = []
  X1 = np.loadtxt(X1_file, delimiter=",")
  text_file = open(DETF_file, "r"); lines = text_file.read().split("\n");  DETFs = [l for l in lines if l!='']; DETFs = [x for x in DETFs if x.split("__")[0] in Nodes]
  ## graph embedding ##
  row, col = edge_index
  rowRE, colRE = edge_index_RE
  edgeRE = torch.tensor(RE[rowRE,colRE])
  adjRE = get_adj(rowRE, colRE, edgeRE, num_nodes, norm=False, asymm_norm=directed_asymm_norm, set_diag=directed_set_diag, remove_diag=directed_remove_diag) # only once
  I_TFs = open('TF_indices.csv', 'r').read().splitlines()
  I_TFs = [int(i) for i in I_TFs]
  col_to_remove = torch.load(removeColFile)
  X1, X2, num_nodes = makeX2(X1, DETFs, Nodes) # just 1 input statei
  tensor_X1 = torch.Tensor(X1)
  tensor_X2 = torch.Tensor(X2)
  # graph embedding
  gD = createGraphEmbedding(num_nodes, tensor_X1, tensor_X2, None, None, adjRE, num_propagations, g, edge_index, row, col, directed_asymm_norm, directed_set_diag, directed_remove_diag, I_TFs, num_hops)
  # expand, synthesize and normalize embedding
  Xdata = expandAndNormalizeEmbedding(gD, normalization)
  # remvoe columns
  Xdata = Xdata[:,~col_to_remove]
  Xdata = torch.nan_to_num(Xdata, nan=0.0)
  print(Xdata)
  print(Xdata.shape)
  # dataloader
  num_features = Xdata.shape[1]
  query_dataset = SimpleDataset(Xdata, torch.zeros([Xdata.shape[0]])) # dummy class
  query_loader = DataLoader(query_dataset, batch_size=1000000, shuffle=False)
  return(query_loader)

def createGraphEmbedding(num_nodes, tensor_X1, tensor_X2, tensor_y, tensor_y2, adjRE, num_propagations, g, edge_index, row, col, directed_asymm_norm, directed_set_diag, directed_remove_diag, I_TFs, num_hops):
  all_idx = torch.tensor([i for i in range(num_nodes)])
  gD = []
  for i in range(tensor_X1.shape[0]):
    op_dict = {}
    batch_x = torch.stack([tensor_X1[i], tensor_X2[i]],dim=1)
    batch_x2 = torch.stack([tensor_X1[i], tensor_X2[i]],dim=1) # for opposite direction
    edge_multiple = batch_x[:,0][np.array(edge_index[0])] * batch_x[:,0][np.array(edge_index[1])]
    edge_multiple = edge_multiple / edge_multiple.max().item() # scaling
    adj = get_adj(row, col, edge_multiple, num_nodes, norm=False, asymm_norm=directed_asymm_norm, set_diag=directed_set_diag, remove_diag=directed_remove_diag)
    adj2 = get_adj(row, col, edge_multiple, num_nodes, norm=False, asymm_norm=directed_asymm_norm, set_diag=directed_set_diag, remove_diag=directed_remove_diag)
    # For reactome network  
    absoluteDEG = torch.abs(batch_x[:,1])
    # For shortest paths
    g.es['weight'] = np.round(np.array(1 / (edge_multiple + 1)),3)  # inverse edge weights for shortest paths
    degs = torch.where(batch_x[:,1]!=0)[0].tolist() 
    detfs = list(set(I_TFs) & set(degs))
    if len(detfs)==0:
      detfs = degs
    elif len(detfs) > 10:
      detfs = np.array(detfs)[np.random.randint(0,len(detfs),10)].tolist()
    if tensor_y==None:
      op_dict['label'] = []
    else:
      op_dict['label'] = torch.stack([tensor_y[i].to(torch.long), tensor_y2[i].to(torch.long)],dim=1)
    op_dict['op_embedding'] = []
    op_dict['op_embedding'].append(batch_x[all_idx].to(torch.float))
    print('Diffusing node features')
    for _ in tqdm(range(num_propagations)):
      # 1. forward    
      batch_x = adj @ batch_x
      op_dict['op_embedding'].append(torch.from_numpy(batch_x[all_idx]))
      # 2. transposed
      batch_x2 = adj2 @ batch_x2
      op_dict['op_embedding'].append(torch.from_numpy(batch_x2[all_idx]))
    # Reactome overlap
    op_dict['properties'] = []
    op_dict['properties'].append(torch.tensor(adjRE @ absoluteDEG).reshape(-1,1) )
    # network properties
    op_dict['properties'].append(torch.tensor(g.strength(weights=g.es['weight'], mode='in')).reshape(-1,1) )
    op_dict['properties'].append(torch.tensor(g.strength(weights=g.es['weight'], mode='out')).reshape(-1,1) )
    print("Computing shortest paths to DETFs...")
    s = g.shortest_paths(source=detfs, target=None,weights=g.es['weight'], mode='in') # the heavier the longer
    s = np.array(s)
    s[s==0] = 'inf'
    # reachability ratio (how many can each gene reach / total DEGs)
    rr = (s.shape[0] - np.sum(np.isinf(s),0)) / s.shape[0]
    op_dict['properties'].append(torch.tensor(rr.reshape(-1,1)))
    # average SP to DEGs
    s[np.isinf(s)] = 100
    op_dict['properties'].append(torch.tensor(np.mean(s,0).reshape(-1,1)))
    # Louvain community label
    g_undirected = g.as_undirected()
    g_undirected.es['weight'] = (torch.abs(edge_multiple) + 1e-6).cpu().numpy() 
    community_labels = g_undirected.community_multilevel(weights=g_undirected.es['weight']).membership
    op_dict['properties'].append(torch.tensor(community_labels).reshape(-1, 1))
    # Nonlinear Transformations
    signed_log_transformed = torch.sign(op_dict['op_embedding'][-1]) * torch.log(torch.abs(op_dict['op_embedding'][-1]) + 1e-6) # signed Log 
    sqrt_features = torch.sqrt(torch.clamp(op_dict['op_embedding'][-1], min=0))  # Square root
    quadrart_features = torch.pow(torch.clamp(op_dict['op_embedding'][-1], min=0), 1/4)
    op_dict['op_embedding'].extend([signed_log_transformed, sqrt_features, quadrart_features])
    # Pseudo-Temporal Analysis
    temporal_change = [ op_dict['op_embedding'][t] - op_dict['op_embedding'][t-1] for t in range(1, num_hops-1) ]
    op_dict['op_embedding'].extend(temporal_change)
    # Node Similarity
    cos_sim = F.cosine_similarity(torch.tensor(batch_x).unsqueeze(0), torch.tensor(batch_x).unsqueeze(1), dim=2)
    del batch_x, batch_x2
    op_dict['properties'].append(cos_sim.mean(dim=1).reshape(-1, 1))    
    gD.append(op_dict)
  return gD

class VAE_decrease(nn.Module):
    def __init__(self, in_channels, initial_hidden_size, num_layers, dropout, latent_dimension):
        super(VAE_decrease, self).__init__()
        # Encoder Layer Sizes (decreasing)
        self.encoder_sizes = self._generate_hidden_sizes(initial_hidden_size, num_layers, decreasing=True)
        self.decoder_sizes = self.encoder_sizes[::-1]  # Decoder Layer Sizes (reverse of encoder)
        # Encoder
        self.encoder_layers = nn.ModuleList()
        prev_channels = in_channels
        for hidden_size in self.encoder_sizes:
            self.encoder_layers.append(self._build_layer(prev_channels, hidden_size, dropout, batch_norm=True))
            prev_channels = hidden_size
        self.fc_mu = nn.Linear(self.encoder_sizes[-1], latent_dimension)
        self.fc_logvar = nn.Linear(self.encoder_sizes[-1], latent_dimension)
        # Decoder
        self.decoder_layers = nn.ModuleList()
        prev_channels = latent_dimension
        for hidden_size in self.decoder_sizes:
            self.decoder_layers.append(self._build_layer(prev_channels, hidden_size, dropout, batch_norm=False))  # No BatchNorm in decoder
            prev_channels = hidden_size
        # Final output layer
        self.output_layer = nn.Linear(self.decoder_sizes[-1], in_channels)
    def _generate_hidden_sizes(self, initial_size, num_layers, decreasing=True):
        """Generate hidden layer sizes with an intermediate non-power-of-2 layer, then powers of 2"""
        sizes = [initial_size]
        # Find the largest power of 2 smaller than initial_size
        largest_power_of_2 = 2 ** (initial_size.bit_length() - 1)
        if largest_power_of_2 == initial_size:
            largest_power_of_2 //= 2  # If it's already a power of 2, go one step lower
        # Add the largest power of 2
        sizes.append(largest_power_of_2)
        # Progressively decrease in powers of 2
        size = largest_power_of_2 // 2
        for _ in range(num_layers - 2): 
            sizes.append(size)
            size = max(size // 2, 8)  # Ensure it doesn't go below 8
        return sizes
    def _build_layer(self, in_features, out_features, dropout, batch_norm):
        """Creates a single fully connected layer with activation & dropout"""
        layers = [nn.Linear(in_features, out_features)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_features))  
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)
    def encode(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        logvar = torch.clamp(logvar, min=-5, max=5)  # Prevent extreme variance collapse
        return mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z):
        for layer in self.decoder_layers:
            z = layer(z)
        return torch.sigmoid(self.output_layer(z))  # Output constrained to [0,1]
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def evaluate_ffn(y_true, y_pred):
    # Convert tensors to numpy if needed
    y_true = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else np.array(y_true)
    y_pred = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else np.array(y_pred)
    # Handle Class 3 as wildcard: prediction of 1 or 2 is considered correct
    y_true_adj = y_true.copy()
    class3_mask = y_true == 3
    if np.any(class3_mask):
        correct_mask = class3_mask & ((y_pred == 1) | (y_pred == 2))
        y_true_adj[correct_mask] = y_pred[correct_mask]
        y_true_adj[class3_mask & (y_pred == 0)] = -1  # Mark as invalid
    # Filter out invalid labels (-1)
    valid_mask = y_true_adj != -1
    y_true_adj = y_true_adj[valid_mask]
    y_pred = y_pred[valid_mask]
    # Derived stats
    labels = np.unique(np.concatenate((y_true_adj, y_pred)))
    cm = confusion_matrix(y_true_adj, y_pred, labels=labels)
    TP = np.diag(cm).sum()
    FP = cm.sum(axis=0).sum() - TP
    FN = cm.sum(axis=1).sum() - TP
    TN = cm.sum() - (TP + FP + FN)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    fpr = FP / (FP + TN + 1e-8)
    fnr = FN / (FN + TP + 1e-8)
    return {"f1": f1, "fpr": fpr, "fnr": fnr}

class FFN_decay(pl.LightningModule):
    def __init__(self, in_channels, initial_size, out_channels, num_layers, dropout=0.0, decay_factor=0.7, lr=0.005, l1_lambda=0.0, l2_lambda=0.0):
        super(FFN_decay, self).__init__()
        self.save_hyperparameters()
        self.hidden_layer_sizes = self._generate_hidden_sizes(num_layers, initial_size, decay_factor)
        self.layers = nn.ModuleList()
        prev_channels = in_channels
        for hidden_channels in self.hidden_layer_sizes:
            self.layers.append(nn.Sequential(
                nn.Linear(prev_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels, momentum=0.9),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            ))
            prev_channels = hidden_channels
        self.layers.append(nn.Linear(prev_channels, out_channels))
        self.lr = lr
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.best_val_metrics = {"f1": 0, "fpr": 1, "fnr": 1}
        self.best_test_metrics = {"f1": 0, "fpr": 1, "fnr": 1}
    def _generate_hidden_sizes(self, num_layers, initial_size, decay_factor=0.7):
        hidden_sizes = []
        size = initial_size
        for _ in range(num_layers):
            hidden_sizes.append(size)
            size = max(int(size * decay_factor), 8)
        return hidden_sizes
    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, nn.Sequential):
                for sub_layer in layer:
                    if hasattr(sub_layer, 'reset_parameters'):
                        sub_layer.reset_parameters()
            elif hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        return self.layers[-1](x)
    def step(self, batch, mode="train"):
        x, labels = batch
        logits = self(x)
        loss = F.cross_entropy(logits, labels, reduction='mean')
        if self.l1_lambda > 0:
            l1_penalty = sum(p.abs().sum() for p in self.parameters() if p.requires_grad)
            loss = loss + self.l1_lambda * l1_penalty
        preds = logits.argmax(dim=1)
        eval_results = evaluate_ffn(labels, preds)  # Updated function
        self.log_dict({
            f"{mode}_loss": loss.item(),
            f"{mode}_f1": eval_results["f1"],
            f"{mode}_fpr": eval_results["fpr"],
            f"{mode}_fnr": eval_results["fnr"]
        }, sync_dist=True, prog_bar=True)
        return loss, eval_results
    def training_step(self, batch, batch_idx):
        loss, eval_results = self.step(batch, mode="train")
        total_batches = self.trainer.num_training_batches
        print(f"Batch {batch_idx+1}/{total_batches}: Loss = {loss.item():.6f}", end="\r")
        return loss
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, eval_results = self.step(batch, mode="val")
        return {"loss": loss, **eval_results, "dataloader_idx": dataloader_idx}
    def validation_epoch_end(self, outputs):
        if isinstance(outputs[0], list):
            val_outputs = outputs[0]
            test_outputs = outputs[1] if len(outputs) > 1 else []
        else:
            val_outputs, test_outputs = outputs, []
        def aggregate_metrics(output_list, prefix):
            if len(output_list) == 0:
                return {f"{prefix}_{key}": 0 for key in ["f1", "fpr", "fnr"]}
            return {
                f"{prefix}_{key}": torch.tensor([float(x[key]) for x in output_list], dtype=torch.float32).mean().item()
                for key in ["f1", "fpr", "fnr"]
            }
        val_metrics = aggregate_metrics(val_outputs, "val")
        test_metrics = aggregate_metrics(test_outputs, "test")
        self.log_dict(val_metrics, sync_dist=True, prog_bar=True)
        self.log_dict(test_metrics, sync_dist=True, prog_bar=True)
        train_metrics = {key: self.trainer.logged_metrics.get(f"train_{key}", 0) for key in ["f1", "fpr", "fnr"]}
        val_metrics = {key: val_metrics.get(f"val_{key}", 0) for key in ["f1", "fpr", "fnr"]}
        test_metrics = {key: test_metrics.get(f"test_{key}", 0) for key in ["f1", "fpr", "fnr"]}
        print(f"Epoch {self.current_epoch}: "
              f"train_f1 = {train_metrics['f1']:.4f}, fpr = {train_metrics['fpr']:.4f}, fnr = {train_metrics['fnr']:.4f} | "
              f"val_f1 = {val_metrics['f1']:.4f}, fpr = {val_metrics['fpr']:.4f}, fnr = {val_metrics['fnr']:.4f} | "
              f"test_f1 = {test_metrics['f1']:.4f}, fpr = {test_metrics['fpr']:.4f}, fnr = {test_metrics['fnr']:.4f}")
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_lambda)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def exportPredictedProteins(outfile, sigPs):
  with open(outfile,'w') as wr:
    for i in range(len(sigPs)):
      wr.write(sigPs[i] +"\n")

def wait_for_file(directory, filename, interval=1, timeout=60):
    file_path = os.path.join(directory, filename)
    # Pre-check if file already exists
    if os.path.exists(file_path):
        #print(f"File {filename} already exists")
        return True
    #print(f"Waiting for {filename} to appear in {directory}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(file_path):
            #print(f"File {filename} detected")
            return True
        time.sleep(interval)
    print("Timeout: File did not appear.")
    return False

def compute_BES_score(pemp, hypergeo_pval, rank_hy, pi0=0.99, w_posterior=20.0, w_hypergeo=1.0, w_hypergeo_rank=1.0, pemp_penalty_multiplier=1.0):
    # Convert to arrays
    pemp = np.array(pemp, dtype=float)
    hypergeo_pval = np.array(hypergeo_pval, dtype=float)
    rank_hy = np.array(rank_hy, dtype=float)
    # Identify invalid scores: zero jaccard or hypergeo p = 1 (not 0!)
    mask_zero = hypergeo_pval == 1.0
    # Posterior and penalty
    posterior = 1 - pi0 * pemp
    pemp_soft_penalty = np.exp(-pemp_penalty_multiplier * (1 - pemp))
    s_posterior = posterior * pemp_soft_penalty
    # Normalized rank components (1 = best)
    rank_hy_norm = 1 - (rank_hy - np.min(rank_hy)) / (np.max(rank_hy) - np.min(rank_hy) + 1e-8)
    # BES score
    BES = (w_posterior * s_posterior + w_hypergeo * (1 - hypergeo_pval) + w_hypergeo_rank * rank_hy_norm)
    # Apply hard zero override
    BES[mask_zero] = 0.0
    # Ranking: higher score = better
    BES_rank = pd.Series(BES).rank(ascending=False, method='min').values
    BES_rank[BES == 0.0] = len(BES)
    return BES, BES_rank


