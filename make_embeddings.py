import os, sys, torch 
import numpy as np
from model_library import *

removeColFile = 'removed_columns_minMax.pt'
norma = 'minMax'
X1_file='X_data1.csv.gz'
for cl in ['cl6','cl4']:
  DETF_file='targets_TFs_'+cl+'-cl7.txt'
  query_dataloader = prepareRealdataDataLoader(X1_file, DETF_file, removeColFile, norma)
  for batch in query_dataloader:
    print(batch) # batch[1] is dummy class with all 0
  # export
  torch.save(batch[0],'query_data_'+cl+'cl7.pt')



