import sys
import os
from os.path import join
script_dir = os.path.dirname(os.path.abspath(__file__))
medbert_dir = os.path.dirname(script_dir)
if medbert_dir not in sys.path:
    sys.path.append(os.path.dirname(medbert_dir))
import pickle as pkl
from med_bert.data import tokenizer  
import typer


def main(input_data_file: str = typer.Argument(..., 
                        help="pickle list in the form [[pid1, los1, codes1, visit_nums1], ...]"),
        vocab_save_name: str = typer.Argument(...),
        truncation: int = typer.Option(100, help="maximum number of tokens to keep for each visit"),
    ):
    with open(input_data_file, 'rb') as f:
        data = pkl.load(f)
    Tokenizer = tokenizer.EHRTokenizer()
    output_seqs = Tokenizer.batch_encode([d[2] for d in data], padding=True, truncation=truncation)
    Tokenizer.save_vocab(vocab_save_name)
    data = [(d[0], d[1], seq, d[3]+(truncation-len(d[3]))*[0]) for d, seq in zip(data, output_seqs)]

    with open(join(input_data_file[:-4]+"_tokenized.pkl"), 'wb') as f:
        pkl.dump(data, f)

if __name__ == "__main__":
    typer.run(main)