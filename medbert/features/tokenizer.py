import torch


class EHRTokenizer():
    def __init__(self, vocabulary=None):
        if vocabulary is None:
            self.new_vocab = True
            self.vocabulary = {
                '[PAD]': 0,
                '[CLS]': 1, 
                '[SEP]': 2,
                '[UNK]': 3,
                '[MASK]': 4,
            }
        else:
            self.new_vocab = False
            self.vocabulary = vocabulary

    def __call__(self, seq):
        return self.batch_encode(seq)

    def batch_encode(self, seqs, padding=True, truncation=None):
        code_seqs = [seq[2] for seq in seqs] # icd codes
        visit_seqs = [seq[3] for seq in seqs]

        if isinstance(truncation, type(None)):
            max_len = max([len(seq) for seq in code_seqs])
        else:
            max_len = truncation
        
        output_seqs = []

        for code_seq, visit_seq in zip(code_seqs, visit_seqs):
            # Tokenizing
            tokenized_code_seq = self.encode(code_seq)
            tokenized_visit_seq = 
            # Padding
            if padding:
                if len(tokenized_code_seq)>max_len:
                    padded_seq = tokenized_code_seq[:max_len]
                difference = max_len - len(tokenized_code_seq)
                padded_seq = tokenized_code_seq + [self.vocabulary['[PAD]']] * difference
            else: 
                padded_seq = tokenized_code_seq
            # Truncating
            truncated_seq = padded_seq[:truncation]

            output_seqs.append(truncated_seq)

        return output_seqs

    def encode(self, seq):
        if self.new_vocab:
            for code in seq:
                if code not in self.vocabulary:
                    self.vocabulary[code] = len(self.vocabulary)

        return [self.vocabulary['[CLS]']] + [self.vocabulary[code] for code in seq] + [self.vocabulary['[SEP]']]

    def save_vocab(self, dest):
        with open(dest, 'wb') as f:
            torch.save(self.vocabulary, f)

