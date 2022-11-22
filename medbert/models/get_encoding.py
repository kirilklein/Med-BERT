from os.path import join, split
import typer
import torch
import numpy as np
import pandas as pd
import torch
from medbert.features.mlm_plos_dataset import MLM_PLOS_Dataset
from medbert.models.utils import Encoder


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
    encoder = Encoder(dataset, model_path, pat_ids, from_checkpoint=from_checkpoint, 
                batch_size=batch_size)
    encoder()
    
if __name__=='__main__':
    typer.run(main)