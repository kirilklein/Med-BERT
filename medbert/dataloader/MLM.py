from torch.utils.data.dataset import Dataset
import numpy as np
from medbert.dataloader.utils import random_mask, seq_padding
import torch


class MLMLoader(Dataset):
    def __init__(self, vocab, data, max_len=512):
        self.vocab = vocab
        self.codes_all = data['codes']
        self.segments_all = data['segments']
        self.max_len = max_len

    def __getitem__(self, index):
        """
        return: code, position, segmentation, mask, label
        """
        codes = self.codes_all[index]
        segments = self.segments_all[index]
        # mask 0:len(code) to 1, padding to be 0
        mask = np.ones(self.max_len)
        mask[:len(codes)] = 0
        # mask 
        #TODO: do we need original codes?
        masked_codes, labels = random_mask(codes, self.vocab) 
        # pad code sequence, segments and label
        pad_codes = seq_padding(masked_codes, self.max_len, self.vocab)
        pad_segments = seq_padding(segments, self.max_len, self.vocab)
        pad_labels = seq_padding(labels, self.max_len, self.vocab)

        return torch.LongTensor(pad_codes), torch.LongTensor(pad_segments), \
               torch.LongTensor(mask), torch.LongTensor(pad_labels)

    def __len__(self):
        return len(self.codes_all)
