import random
import numpy as np


#TODO dont mask cls and sep tokens
def random_mask(codes, vocab):
    """One code is masked per patient"""
    masked_codes = codes
    labels = len(codes) * [-100] # -100 is ignored by loss function
    mask_code = np.random.randint(len(codes))
    prob = np.random.uniform()    
    if prob < 0.8:
        masked_codes[mask_code] = vocab['MASK']
        labels[mask_code] = vocab['UNK']
    # 10% randomly change token to random token
    elif prob < 0.9:
        masked_codes[mask_code] = random.choice(list(vocab.values())[3:]) # first three tokens reserved!
    return masked_codes, labels

def seq_padding(seq, max_len, vocab):
    return seq + (max_len-len(seq)) * [vocab['PAD']]

#TODO torch.utils.data.random_split