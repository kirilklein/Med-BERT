from os.path import join
import pickle as pkl
from medbert.features import tokenizer  
import typer
import torch


def main(
    input_data_file: str = typer.Argument(..., 
        help="pickle list in the form [[pid1, los1, codes1, visit_nums1], ...]"),
    vocab_save_name: str = typer.Argument(...),
    truncation: int = 
        typer.Option(100, help="maximum number of tokens to keep for each visit"),
    ):

    with open(input_data_file, 'rb') as f:
        data = pkl.load(f)

    Tokenizer = tokenizer.EHRTokenizer()
    tokenized_data_dic = Tokenizer.batch_encode(data)
    Tokenizer.save_vocab(vocab_save_name)
    torch.save(tokenized_data_dic, join(input_data_file[:-4]+"_tokenized.pt"))
    
if __name__ == "__main__":
    typer.run(main)