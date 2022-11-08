from transformers import BertForMaskedLM, BertConfig
from medbert.dataloader.MLM import MLMLoader
from . import utils
import torch
import typer
import json
import os


app = typer.Typer(name="pretraining", add_completion=False, help="MLM Pretraining")
@app.command()
def main(data_file : str = typer.Argument(..., help="Tokenized data"),
    vocab_file : str = typer.Argument(..., help=".pt vocab dic"),
    save_path : str = typer.Argument(...),
    epochs : int = typer.Argument(...),
    batch_size : int = typer.Option(16),
    load_path : str = typer.Argument(None, help=".pt containing the model"),
    max_len : int = typer.Option(512, help="maximum number of tokens in seq"),
    config_file : str = typer.Option("configs\\mlm_config.json", 
        help="Location of the config file"),
    checkpoint_freq : int = typer.Option(5, help="Frequency of checkpoints in epochs")
    ):
    #TODO Check intermediate size, is it 64?
    
    data = torch.load(data_file)
    vocab = torch.load(vocab_file)
    #model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    max_num_visits=30 # vocab_size has to be at least as large as the largest input id
    
    if isinstance(load_path, type(None)):
        print("Initialize new model")
        #config_file = str(config_file)
        with open(config_file) as f:
            config_dic = json.load(f)
        config = BertConfig(vocab_size=len(vocab)+max_num_visits, **config_dic) 
        model = BertForMaskedLM(config)
    else:
        print(f"Load saved model from {load_path}")
        model = torch.load(load_path)
    dataset = MLMLoader(data, vocab, max_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    trainer = utils.CustomMLMTrainer(dataset, model, epochs, batch_size, save_path)
    trainer()
    torch.save(model, save_path)
    print(f"Trained model saved to {save_path}")
if __name__=='__main__':
    typer.run(main)

