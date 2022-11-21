import torch
import typer
import pickle as pkl
from os.path import join, split

class EHRTokenizer():
    def __init__(self, vocabulary=None):
        if isinstance(vocabulary, type(None)):
            self.vocabulary = {
                'PAD':0,
                'MASK':1,
                'UNK':2,
            }
        else:
            self.vocabulary = vocabulary

    def __call__(self, seq):
        return self.batch_encode(seq)

    def encode(self, seq):
        for code in seq:
            if code not in self.vocabulary:
                self.vocabulary[code] = len(self.vocabulary)
        return [self.vocabulary[code] for code in seq]

    def batch_encode(self, seqs, max_len=None):
        if isinstance(max_len, type(None)):
            max_len = max([len(seq) for seq in seqs])
        pat_ids = [seq[0] for seq in seqs]
        los_seqs = [seq[1] for seq in seqs]
        code_seqs = [seq[2] for seq in seqs] # icd codes
        visit_seqs = [seq[3] for seq in seqs]
        if isinstance(max_len, type(None)):
            max_len = max([len(seq) for seq in code_seqs])    
        output_code_seqs = []
        output_visit_seqs = []
        # let's do the padding later
        for code_seq, visit_seq in zip(code_seqs, visit_seqs):
            # truncation
            if len(code_seq)>max_len:
                code_seq = code_seq[:max_len]
                visit_seq = visit_seq[:max_len]
            # Tokenizing
            tokenized_code_seq = self.encode(code_seq)
            output_code_seqs.append(tokenized_code_seq)
            output_visit_seqs.append(visit_seq)
        tokenized_data_dic = {'pats':pat_ids, 'los':los_seqs, 'codes':output_code_seqs, 
                            'segments':output_visit_seqs}
        return tokenized_data_dic

    def save_vocab(self, dest):
        print(f"Writing vocab to {dest}")
        torch.save(self.vocabulary, dest)


def main(
    input_data_path: str = typer.Argument(..., 
        help="pickle list in the form [[pid1, los1, codes1, visit_nums1], ...]"),
    vocab_save_path: str = typer.Option(None, help="Path to save vocab, must end with .pt"),
    out_data_path: str = typer.Option(None, help="Path to save tokenized data, must end with .pt"),
    max_len: int = 
        typer.Option(None, help="maximum number of tokens to keep for each visit"),
    ):

    with open(input_data_path, 'rb') as f:
        data = pkl.load(f)

    Tokenizer = EHRTokenizer()
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
