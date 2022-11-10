#%%
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
#%%
print()
ids = torch.LongTensor(np.arange(10000))
embedding = nn.Embedding(len(np.unique(ids))+1, 2)
embed_ids = embedding(ids)
print(embed_ids)
plt.scatter(embed_ids.detach().numpy()[:,0], embed_ids.detach().numpy()[:,1])
#for id, embed_id in zip(ids, embed_ids):
 #   plt.annotate(f"{id}", (embed_id[0], embed_id[1]))