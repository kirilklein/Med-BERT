#%%
#import torch
#from torch import nn
import numpy as np
#import matplotlib.pyplot as plt
#%%
# ids = torch.LongTensor(np.concatenate([np.arange(100), np.zeros(10)]))
# embedding = nn.Embedding(len(np.unique(ids))+1, 2, padding_idx=0)
# embed_ids = embedding(ids)
#print(embed_ids)
# print(embed_ids[-10:])
# plt.scatter(embed_ids.detach().numpy()[:,0], embed_ids.detach().numpy()[:,1])
#for id, embed_id in zip(ids, embed_ids):
 #   plt.annotate(f"{id}", (embed_id[0], embed_id[1]))
ls = [3,4,8]
(np.array(ls)>7).any().astype(int)