#%%
import torch
#from torch import nn
import numpy as np
# from transformers import BertConfig
#import matplotlib.pyplot as plt
from medbert.dataloader.mlm_plos_loader import MLM_PLOS_Loader
from torch.utils.data import random_split
import pandas as pd
#%%
# ids = torch.LongTensor(np.concatenate([np.arange(100), np.zeros(10)]))
# embedding = nn.Embedding(len(np.unique(ids))+1, 2, padding_idx=0)
# embed_ids = embedding(ids)
#print(embed_ids)
# print(embed_ids[-10:])
# plt.scatter(embed_ids.detach().numpy()[:,0], embed_ids.detach().numpy()[:,1])
#for id, embed_id in zip(ids, embed_ids):
 #   plt.annotate(f"{id}", (embed_id[0], embed_id[1]))
#ls = [3,4,8]
#type((np.array(ls)>7).any().astype(int))
# get working directory
import os
from os.path import join,split
curdir = os.getcwd()
curdir = split(split(curdir)[0])[0]
vocab = torch.load(join(curdir,"data\\raw\\simulated\\example_data_vocab.pt"))
data = pd.DataFrame({'codes':[[1,2],[2,3],[3,5],[1,4],[2,5]], 'segments':[1,2,3,4,5], 'los':[1,2,3,4,5]})

train_size = int(0.7 * len(data))

test_size = len(data) - train_size
#split data into train and test using torch.utils.data.random_split
dataset = MLM_PLOS_Loader(data, vocab)

train_dataset, test_dataset = random_split(dataset, [.5, .5])
#%%

train_size = len(train_dataset)
train_dataset.indices
test_size = len(test_dataset)
test_dataset.indices