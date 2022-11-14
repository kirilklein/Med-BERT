import torch


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

    def batch_encode(self, seqs, max_len=512):
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

