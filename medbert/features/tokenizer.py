import torch
from tqdm import tdqm

class EHRTokenizer():
    def __init__(self, vocabulary=None):
        if isinstance(vocabulary, type(None)):
            self.vocabulary = {
                'PAD': 0,
                'MASK': 1,
                'UNK':2,
                #'[CLS]': 1, 
                #'[SEP]': 2,
                #'[UNK]': 3,
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

    def batch_encode(self, seqs, padding=True, truncation=512):
        pat_ids = [seq[0] for seq in seqs]
        los_seqs = [seq[1] for seq in seqs]
        code_seqs = [seq[2] for seq in seqs] # icd codes
        visit_seqs = [seq[3] for seq in seqs]

        if isinstance(truncation, type(None)):
            max_len = max([len(seq) for seq in code_seqs])
        else:
            max_len = truncation
        
        output_code_seqs = []
        output_visit_seqs = []
        for code_seq, visit_seq in zip(code_seqs, visit_seqs):
            # Tokenizing
            tokenized_code_seq = self.encode(code_seq)
            # Padding
            if padding:
                if len(tokenized_code_seq)>max_len:
                    tokenized_code_seq = tokenized_code_seq[:max_len]
                difference = max_len - len(tokenized_code_seq)
                padded_code_seq = tokenized_code_seq \
                    + [self.vocabulary['PAD']] * difference
                padded_visit_seq = visit_seq \
                    + [self.vocabulary['PAD']] * difference
            else: 
                padded_code_seq = tokenized_code_seq
                padded_visit_seq = visit_seq
            output_code_seqs.append(padded_code_seq)
            output_visit_seqs.append(padded_visit_seq)
        tokenized_data_dic = {'ids':pat_ids, 'los':los_seqs, 'codes':code_seqs, 'segment_ids':visit_seqs}
        return tokenized_data_dic

    def save_vocab(self, dest):
        with open(dest, 'wb') as f:
            torch.save(self.vocabulary, f)

