from medbert.models import mlm_plos_pretraining
from os.path import join


sim_data = "data\\raw\\simulated"
data_file = join(sim_data, "example_data_tokenized.pt")
vocab_file = join(sim_data, "example_data_vocab.pt")

mlm_plos_pretraining.main(data_file=data_file, vocab_file=vocab_file,
        save_path="models/mlm_pretrained/test.pt", 
        epochs=1, 
        batch_size=64, 
        max_len=100,
        config_file="configs\\pretrain_config.json",
        checkpoint_freq=1)