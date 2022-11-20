from os.path import join, split
import torch
from transformers import BertForPreTraining, BertConfig
import numpy as np
import pandas as pd
import json
import torch
from medbert.dataloader.embeddings import BertEmbeddings
from medbert.dataloader.mlm_plos_loader import MLM_PLOS_Loader
from medbert.common import pytorch


class Encoding():
    def __init__(self, model_path, tokenized_data_path, from_checkpoint=False):
        self.model_path = model_path
        self.model_dir = split(model_path)[0]
        self.config = self.get_config()
        self.tokenized_data_path = tokenized_data_path
        self.vocab = self.get_vocab()
        self.from_checkpoint = from_checkpoint
        self.model = self.get_model()
        self.data = pd.DataFrame(torch.load(tokenized_data_path))
        self.input_dataset = MLM_PLOS_Loader(self.data, self.vocab, self.config.max_length)
        self.input_loader = torch.utils.data.DataLoader(self.input_dataset,   # type: ignore
                                                batch_size=64, shuffle=False)

    def get_config(self):
        with open(join(self.model_dir, "config.json"), 'r') as f:
            config_dic = json.load(f)
        config = BertConfig(**config_dic)
        return config

    def get_vocab(self):
        data_dir = split(split(self.tokenized_data_path)[0])[0]
        vocab_path = join(data_dir, 'vocab',split(self.tokenized_data_path)[1])
        print(f"Load vocab from {vocab_path}")
        vocab = torch.load(vocab_path)
        return vocab

    def get_model(self):
        if self.from_checkpoint:
            checkpoint_path = join(self.model_dir, "checkpoint.pt")
            checkpoint = torch.load(checkpoint_path)
            config = checkpoint['config']
            model = BertForPreTraining(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Load model from {checkpoint_path}")
        else:
            model = torch.load(self.model_path)
            print(f"Load model from {self.model_path}")
        return model
        

    def get_encoding(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        embeddings = BertEmbeddings(config=self.config)
        self.model.eval()
        feat_ls = []
        print("Start encoding...")
        with torch.no_grad():
            for i, batch in enumerate(self.input_loader):
                print(len(batch))
                batch = pytorch.batch_to_device(batch, device)
                # get embeddings
                embedding_output = embeddings(batch['codes'], batch['segments'])
                print(embedding_output)
                # process
                outputs = self.model(inputs_embeds=embedding_output, ) 
                features = outputs[0][:, 0, :].numpy()
                print(features.shape)
                feat_ls.append(features)
        return feat_ls