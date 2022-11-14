from transformers import BertForPreTraining, BertConfig
from medbert.dataloader.mlm_plos_loader import MLM_PLOS_Loader
from . import utils
import torch
import typer
import json



app = typer.Typer(name="pretraining", add_completion=False, help="MLM Pretraining")
@app.command()
def main(data_file : str = typer.Argument(..., help="Tokenized data"),
    vocab_file : str = typer.Argument(..., help=".pt vocab dic"),
    save_path : str = typer.Argument(...),
    epochs : int = typer.Argument(...),
    batch_size : int = typer.Option(16),
    load_path : str = typer.Argument(None, help=".pt containing the model"),
    max_len : int = typer.Option(512, help="maximum number of tokens in seq"),
    max_num_seg : int = typer.Option(100, help="maximum number of segments in seq"),
    config_file : str = typer.Option("configs\\mlm_config.json", 
        help="Location of the config file"),
    checkpoint_freq : int = typer.Option(5, help="Frequency of checkpoints in epochs"),
    from_checkpoint : bool = typer.Option(False, help="Load model from checkpoint")
    ):
    data = torch.load(data_file)
    vocab = torch.load(vocab_file)
    with open(config_file) as f:
            config_dic = json.load(f)
    config = BertConfig(vocab_size=len(vocab), **config_dic)
    if isinstance(load_path, type(None)):
        print("Initialize new model")
        model = BertForPreTraining(config)
    else:
        print(f"Load saved model from {load_path}")
        model = torch.load(load_path)
    config.vocab_size = len(vocab)
    config.seg_vocab_size = max_num_seg
    dataset = MLM_PLOS_Loader(data, vocab, max_len)
    trainer = utils.CustomPreTrainer(dataset, model, epochs, batch_size, 
                save_path, checkpoint_freq=checkpoint_freq, 
                from_checkpoint=from_checkpoint, config=config)
    trainer()
    torch.save(model, save_path)
    print(f"Trained model saved to {save_path}")
if __name__=='__main__':
    typer.run(main)