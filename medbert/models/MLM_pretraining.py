from transformers import BertForMaskedLM
from medbert.dataloader.MLM import MLMLoader
from . import utils
import torch
import typer


def main(data_file : str = typer.Argument(..., help="Tokenized data"),
    vocab_file : str = typer.Argument(..., help=".pt vocab dic"),
    save_dir : str = typer.Argument(...),
    epochs : int = typer.Argument(...),
    batch_size : int = typer.Option(16),
    max_len : int = typer.Option(512, help="maximum number of tokens in seq"),
    ):
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    data = torch.load(data_file)
    vocab = torch.load(vocab_file)
    dataset = MLMLoader(data, vocab, max_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    trainer = utils.CustomMLMTrainer(dataset, model, epochs, batch_size, save_dir)
    trainer()

