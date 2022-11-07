from medbert.dataloader.MLM import MLMLoader
from os.path import join
import torch

sim_data = "data\\raw\\simulated"
data_file = join(sim_data, "example_data_tokenized.pt")
vocab_file = join(sim_data, "example_data_vocab.pt")
data = torch.load(data_file)
vocab = torch.load(vocab_file)
loader = MLMLoader(vocab, data, max_len=100)
print("Length of dataset ", len(loader))
for i, b in enumerate(loader):
    if i==1:
        break
    print(b)