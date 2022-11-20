from os.path import join, split
import json
import torch
from transformers import BertForPreTraining, BertConfig
import numpy as np


class Encoding():
    def __init__(self, model_path, tokenized_data_path, from_checkpoint=False):
        self.model_path = model_path
        self.model_dir = split(model_path)[0]
        self.tokenized_data_path = tokenized_data_path
        self.vocab = self.get_vocab()
        self.from_checkpoint = from_checkpoint
        self.model = self.get_model()
        self.input = self.get_input()

    def get_vocab(self):
        tok_data_dir = split(self.tokenized_data_path)[0]
        vocab_path = join(tok_data_dir, split(self.tokenized_data_path)[1][-12:]+"_vocab.pt")
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
            last_hidden_states = self.model(self.input)
        features = last_hidden_states[0][:, 0, :].numpy()
        return features

    def get_input(self):
        input = np.zeros((10,10))
        return input