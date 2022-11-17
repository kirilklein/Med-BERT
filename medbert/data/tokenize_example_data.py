from os.path import join, split
import pickle as pkl
from medbert.dataloader import tokenizer  
import typer
import torch


def main(
    input_data_path: str = typer.Argument(..., 
        help="pickle list in the form [[pid1, los1, codes1, visit_nums1], ...]"),
    vocab_save_path: str = typer.Option(
        None, help="Path to save vocab, must end with .pt"),
    out_data_path: str = typer.Option(
        None, help="Path to save tokenized data, must end with .pt"), 
    max_len: int = 
        typer.Option(100, help="maximum number of tokens to keep for each visit"),
    ):

    with open(input_data_path, 'rb') as f:
        data = pkl.load(f)

    Tokenizer = tokenizer.EHRTokenizer()
    tokenized_data_dic = Tokenizer.batch_encode(data, max_len=max_len)
    if isinstance(vocab_save_path, type(None)):
        data_dir = split(split(input_data_path)[0])[0]
        vocab_save_path = join(join(data_dir, 'tokenized', split(input_data_path)[1][:-4]+"_vocab.pt"))
    Tokenizer.save_vocab(vocab_save_path)
    if isinstance(out_data_path, type(None)):
        data_dir = split(split(input_data_path)[0])[0]
        out_data_path = join(join(data_dir, 'tokenized', split(input_data_path)[1][:-4]+"_tokenized.pt"))
    print(f"Save tokenized data to {out_data_path}")
    torch.save(tokenized_data_dic, out_data_path)
    
if __name__ == "__main__":
    typer.run(main)