
import numpy as np


def random_mask(idxs, vocab, mask_prob=0.15,
    special_tokens=['MASK', 'UNK', 'PAD'], seed=0):
    """mask code with 15% probability, 80% of the time replace with [MASK], 
        10% of the time replace with random token, 10% of the time keep original"""
    rng = np.random.default_rng(seed)
    masked_idxs = idxs.copy()
    special_idxs = [vocab[token] for token in special_tokens]
    labels = len(idxs) * [-100] # -100 is ignored by loss function
    
    for i, idx in enumerate(idxs):
        if idx in special_idxs:
            continue
        prob = rng.uniform()
        if prob<mask_prob:
            prob = rng.uniform()  
            # 80% of the time replace with [MASK] 
            if prob < 0.8:
                masked_idxs[i] = vocab['MASK']
            # 10% change token to random token
            elif prob < 0.9:
                masked_idxs[i] = rng.choice(list(vocab.values())[len(special_idxs):]) # first tokens are special!
            # 10% keep original
            labels[i] = idx
    return masked_idxs, labels

