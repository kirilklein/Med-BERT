from medbert.models import MLM_pretraining
from os.path import join


sim_data = "data\\raw\\simulated"
data_file = join(sim_data, "example_data_tokenized.pt")
vocab_file = join(sim_data, "example_data_vocab.pt")
MLM_pretraining.main(vocab_file, data_file,30)