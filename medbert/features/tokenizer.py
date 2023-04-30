import torch
import typer
import pickle as pkl
from os.path import join, split
from tqdm import tqdm

class EHRTokenizer():
    def __init__(self, vocabulary=None, config=None):
        self.frozen = False
        self.config = config
        if isinstance(vocabulary, type(None)):
            self.vocabulary = {
                '[PAD]':0,
                '[MASK]':1,
                '[UNK]':2,
            }
            if config.sep_tokens:
                self.vocabulary['[SEP]'] = len(self.vocabulary)
            if config.cls_token:
                self.vocabulary['[CLS]'] = len(self.vocabulary)
        else:
            self.vocabulary = vocabulary

    def __call__(self, features):
        return self.batch_encode(features)

    def encode(self, seq):
        for code in seq:
            if code not in self.vocabulary:
                if not self.frozen:
                    self.vocabulary[code] = len(self.vocabulary)
                
        return [self.vocabulary.get(code, self.vocabulary['[UNK]']) for code in seq]

    def batch_encode(self, features):
        tokenized_data_dic = {'pats': [], 'los': [], 'concepts': [], 'segments': [], 'attention_mask': []}
        concepts = [seq[2] for seq in features] # icd codes
        pad_len = max([len(concept_seq) for concept_seq in concepts])   
        if pad_len > self.config.truncation:
            pad_len = self.config.truncation
        target_seqs = []
        out_visit_seqs = []
        for patient in tqdm(features, desc="Tokenizing", total=len(concepts)):
            # truncation
            concept_seq = patient[2]
            visit_seq = patient[3]
            if len(concept_seq)>self.config.truncation:
                concept_seq = concept_seq[:self.config.truncation]
                visit_seq = visit_seq[:self.config.truncation]
            
            # Tokenizing
            tokenized_concept_seq = self.encode(concept_seq)
            attention_mask = [1] * len(concept_seq)
            # Pad tokenized code seq
            if self.config.padding:
                tokenized_concept_seq = self.pad(tokenized_concept_seq, pad_len)
                visit_seq = self.pad(visit_seq, pad_len, pad_token=0)
                attention_mask = self.pad(attention_mask, pad_len, pad_token=0)

            target_seqs.append(tokenized_concept_seq)
            out_visit_seqs.append(visit_seq)
            tokenized_data_dic['concepts'].append(target_seqs) 
            tokenized_data_dic['segments'].append(out_visit_seqs)
            tokenized_data_dic['attention_mask'].append(attention_mask)

        tokenized_data_dic['pats'] = [seq[0] for seq in features]
        tokenized_data_dic['los'] = [seq[1] for seq in features]
        return tokenized_data_dic
    
    def pad(self, seq: dict,  pad_len: int, pad_token: int=None):
        if pad_token is None:
            pad_token = self.vocabulary['[PAD]']
        difference = pad_len - len(seq)
        if difference > 0:
            return seq + [pad_token] * difference
        else:
            return seq

    def freeze_vocab(self):
        self.frozen = True

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
