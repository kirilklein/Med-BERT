from medbert.models import MLM_pretraining
from os.path import join
import torch

sim_data = "data\\raw\\simulated"
data_file = join(sim_data, "example_data_tokenized.pt")
vocab_file = join(sim_data, "example_data_vocab.pt")

MLM_pretraining.main(data_file, vocab_file, 1, 100)