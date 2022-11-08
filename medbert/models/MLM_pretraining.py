from transformers import BertForMaskedLM, BertConfig
from medbert.dataloader.MLM import MLMLoader
from . import utils
import torch
import typer
import json
import os


app = typer.Typer(name="demo", add_completion=False, help="This is a demo app.")

@app.command()
def main(data_file : str = typer.Argument(..., help="Tokenized data"),
    vocab_file : str = typer.Argument(..., help=".pt vocab dic"),
    save_dir : str = typer.Argument(...),
    epochs : int = typer.Argument(...),
    batch_size : int = typer.Option(16),
    max_len : int = typer.Option(512, help="maximum number of tokens in seq"),
    config_file : str = typer.Option("configs\\mlm_config.json", 
        help="Location of the config file"),
    ):
    #TODO Check intermediate size, is it 64?
    config_file = str(config_file)
    with open(config_file) as f:
        config_dic = json.load(f)
    data = torch.load(data_file)
    vocab = torch.load(vocab_file)
    #model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    max_num_visits=30 # vocab_size has to be at least as large as the largest input id
    config = BertConfig(vocab_size=len(vocab)+max_num_visits, **config_dic) 
    model = BertForMaskedLM(config)
    dataset = MLMLoader(data, vocab, max_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    trainer = utils.CustomMLMTrainer(dataset, model, epochs, batch_size, save_dir)
    trainer()
    torch.save(model, save_dir)
if __name__=='__main__':
    typer.run(main)

