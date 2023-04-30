from typing import List
from operator import itemgetter
import numpy as np

class Splitter:
    def __init__(self, ratios: List = [0.7, 0.2, 0.1]):
        """Ratios: train, val, test"""
        self.ratios = ratios

    def __call__(self, features: List[List]):
        # draw random indices
        indices = np.random.permutation(len(features))
        # split indices according to ratios
        train_idx = int(len(features) * self.ratios[0])
        val_idx = int(len(features) * self.ratios[1])
        # split features according to indices
        train = itemgetter(*indices[:train_idx])(features)
        val = itemgetter(*indices[train_idx:train_idx+val_idx])(features)
        test = itemgetter(*indices[train_idx+val_idx:])(features)
        
        return train, val, test