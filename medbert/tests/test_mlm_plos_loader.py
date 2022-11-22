from medbert.features.dataset import MLM_PLOS_Dataset
from os.path import join
import torch

sim_data = "data\\raw\\simulated"
data_file = join(sim_data, "example_data_tokenized.pt")
vocab_file = join(sim_data, "example_data_vocab.pt")
data = torch.load(data_file)
vocab = torch.load(vocab_file)
dataset = MLM_PLOS_Dataset(data, vocab, max_len=100)
loader = torch.utils.data.DataLoader(train_dataset,   # type: ignore
                batch_size=64, shuffle=True)
print("Length of dataset ", len(loader))
for i, b in enumerate(loader):
    if i==1:
        break
    for k,v in b.items():
        print(k, v, len(v))