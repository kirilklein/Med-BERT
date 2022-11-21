from transformers import BertForPreTraining, BertConfig
from medbert.features.mlm_plos_dataset import MLM_PLOS_Dataset
from medbert.models import utils
import torch
import typer
import json
from torch.utils.data import random_split
import pandas as pd
from os.path import join
import numpy as np


def main(
    data_file : str = typer.Argument(..., help="Tokenized data"),
    vocab_file : str = typer.Argument(..., help=".pt vocab dic"),
    model_dir : str = typer.Argument(..., help="Directory to save model"),
    epochs : int = typer.Argument(..., help="Number of epochs"),
    batch_size : int = typer.Option(32, help="Batch size"),
    load_model : bool = typer.Option(False, help="Load saved model"),
    max_len : int = typer.Option(None, help="maximum number of tokens in seq"),
    max_num_seg : int = typer.Option(100, help="maximum number of segments in seq"),
    config_file : str = typer.Option(join('configs','pretrain_config.json'), 
        help="Location of the config file"),
    checkpoint_freq : int = typer.Option(5, help="Frequency of checkpoints in epochs"),
    from_checkpoint : bool = typer.Option(False, help="Load model from checkpoint")
    ):
    args = locals()
    typer.echo(f"Arguments: {args}")

    data = torch.load(data_file)
    if isinstance(max_num_seg, type(None)):
        max_num_seg = int(np.max([max(segs) for segs in data['segments']])) + 1 # +1 for padding

    data = pd.DataFrame(data) 
    vocab = torch.load(vocab_file)
    with open(config_file) as f:
            config_dic = json.load(f)
    config = BertConfig(vocab_size=len(vocab), **config_dic)
    if not load_model:
        print("Initialize new model")
        model = BertForPreTraining(config)
    else:
        print(f"Load saved model from {model_dir}")
        model = torch.load(join(model_dir, 'model.pt'))
    config.vocab_size = len(vocab)
    config.pad_token_id = vocab['PAD']
    config.seg_vocab_size = max_num_seg
    typer.echo(f"Config: {vars(config)}")

    dataset = MLM_PLOS_Dataset(data, vocab, max_len)
    print(f"Use {config.validation_size*100}% of data for validation")
    train_dataset, val_dataset = random_split(dataset, 
                    [1-config.validation_size, config.validation_size],
                    generator=torch.Generator().manual_seed(42))
    
    trainer = utils.CustomPreTrainer(train_dataset, val_dataset, model, epochs, 
                batch_size, model_dir, checkpoint_freq=checkpoint_freq, 
                from_checkpoint=from_checkpoint, config=config, args=args)
    trainer()
    trainer.save_model()
    
if __name__=='__main__':
    typer.run(main)