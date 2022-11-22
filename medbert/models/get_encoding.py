from os.path import join, split
import typer
import torch
from transformers import BertForPreTraining, BertConfig
import numpy as np
import pandas as pd
import json
import torch
from medbert.features.embeddings import BertEmbeddings
from medbert.features.mlm_plos_dataset import MLM_PLOS_Dataset
from medbert.models.utils import Encoder
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
        self.input_dataset = MLM_PLOS_Dataset(self.data, self.vocab, self.config.max_length)
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


def main(
    data_file : str = typer.Argument(..., help="Tokenized data"),
    model_path : str = typer.Argument(..., help=".pt containing the model"),
    vocab_file : str = typer.Option(None, help=".pt vocab dic"),
    batch_size : int = typer.Option(128, help="Batch size"),
    max_len : int = typer.Option(None, help="maximum number of tokens in seq"),
    max_num_seg : int = typer.Option(100, help="maximum number of segments in seq"),
    from_checkpoint : bool = typer.Option(False, help="Load model from checkpoint")
    ):
    data = torch.load(data_file)
    if isinstance(max_num_seg, type(None)):
        max_num_seg = int(np.max([max(segs) for segs in data['segments']]))
    data = pd.DataFrame(data) 
    pat_ids = data['pats']
    if isinstance(vocab_file, type(None)):
        vocab_file = join(split(split(data_file)[0])[0], 'vocab', split(data_file)[1])
    vocab = torch.load(vocab_file)
    dataset = MLM_PLOS_Dataset(data, vocab, max_len) 
    encoder = Encoder(dataset, model_path, from_checkpoint=from_checkpoint, 
                batch_size=batch_size, pat_ids=pat_ids)
    encoder()
    
if __name__=='__main__':
    typer.run(main)