from os.path import join, split
import torch
from transformers import BertForPreTraining, BertConfig
import numpy as np
import json
from medbert.dataloader.embeddings import BertEmbeddings
from medbert.dataloader.mlm_plos_loader import MLMLoader


class Encoding():
    def __init__(self, model_path, tokenized_data_path, from_checkpoint=False):
        self.model_path = model_path
        self.model_dir = split(model_path)[0]
        self.config = self.get_config()
        self.tokenized_data_path = tokenized_data_path
        self.vocab = self.get_vocab()
        self.from_checkpoint = from_checkpoint
        self.model = self.get_model()
        self.input = torch.load(tokenized_data_path)
        self.input_loader = MLMLoader(self.input, self.vocab, self.config)

    def get_config(self):
        with open(join(self.model_dir, "config.json"), 'r') as f:
            config_dic = json.load(f)
        config = BertConfig(**config_dic)
        return config

    def get_vocab(self):
        data_dir = split(split(self.tokenized_data_path)[0])[0]
        vocab_path = join(data_dir, 'vocab',split(self.tokenized_data_path)[1])
        vocab = torch.load(vocab_path)
        return vocab

    def get_model(self):
        if self.from_checkpoint:
            checkpoint_path = join(self.model_dir, "checkpoint.pt")
            checkpoint = torch.load(checkpoint_path)
            config = checkpoint['config']
            model = BertForPreTraining(config)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model = torch.load(self.model_path)
        return model
        

    def get_encoding(self):
        with torch.no_grad():
            for i, batch in enumerate(self.input_loader):
            last_hidden_states = self.model(self.inp_vec)
        features = last_hidden_states[0][:, 0, :].numpy()
        return features