from os.path import join, split
import pickle as pkl
from medbert.dataloader import tokenizer  
import typer
import torch


def main(
    input_data_file: str = typer.Argument(..., 
        help="pickle list in the form [[pid1, los1, codes1, visit_nums1], ...]"),
    vocab_save_name: str = typer.Option(None, help="Path to save vocab, must end with .pt"),
    max_len: int = 
        typer.Option(100, help="maximum number of tokens to keep for each visit"),
    ):

    with open(input_data_file, 'rb') as f:
        data = pkl.load(f)

    Tokenizer = tokenizer.EHRTokenizer()
    tokenized_data_dic = Tokenizer.batch_encode(data, max_len=max_len)
    if isinstance(vocab_save_name, type(None)):
        vocab_save_name = join(input_data_file[:-4]+"_vocab.pt")
    Tokenizer.save_vocab(vocab_save_name)
    save_dest = join(input_data_file[:-4]+"_tokenized.pt")
    print(f"Save tokenized data to {save_dest}")
    torch.save(tokenized_data_dic, save_dest)
    
if __name__ == "__main__":
    typer.run(main)