from torch.utils.data.dataset import Dataset
import numpy as np
from medbert.features.utils import random_mask, seq_padding
import torch


class MLM_PLOS_Dataset(Dataset):
    def __init__(self, data, vocab, max_len=None):
        self.vocab = vocab
        self.codes_all = data['codes']
        self.segments_all = data['segments']
        self.los_all = data['los']
        if isinstance(max_len, type(None)):
            self.max_len = int(np.max(np.array([len(code_ls) for code_ls in self.codes_all])))
        else:
            self.max_len = max_len

    def __getitem__(self, index):
        """
        return: code, position, segmentation, mask, label
        """
        codes = self.codes_all[index]
        segments = self.segments_all[index]
        los = self.los_all[index]
        plos = (np.array(los)>7).any()
        # mask 0:len(code) to 1, padding to be 0
        mask = np.ones(self.max_len)
        mask[len(codes):] = 0
        # mask 
        masked_codes, labels = random_mask(codes, self.vocab) 
        # pad code sequence, segments and label
        pad_codes = seq_padding(masked_codes, self.max_len, self.vocab)
        pad_segments = seq_padding(segments, self.max_len, self.vocab)
        pad_labels = seq_padding(labels, self.max_len, self.vocab)
        output_dic = {
            'codes':torch.LongTensor(pad_codes),
            'segments':torch.LongTensor(pad_segments),
            'attention_mask':torch.LongTensor(mask),
            'labels':torch.LongTensor(pad_labels),
            'plos':torch.LongTensor([plos])}
        return output_dic

    def __len__(self):
        return len(self.codes_all)

