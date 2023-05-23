import torch
from os.path import join

class Splitter():
    def __init__(self, ratios: dict = {'train':0.7, 'val':0.2, 'test':0.1}) -> None:
        self.ratios = ratios
        self.ratios_list = [ratio for ratio in self.ratios.values()]
        self.splits = None
    def __call__(self, features: dict, )-> dict:
        return self.split_features(features)

    def split_features(self, features: dict, )-> dict:
        """
        Split features into train, validation and test sets
        """
        if round(sum(self.ratios_list), 5) != 1:
            raise ValueError(f'Sum of ratios ({self.ratios_list}) != 1 ({round(sum(self.ratios_list), 5)})')
        torch.manual_seed(0)

        N = len(features['concept'])

        self._split_indices(N)
        split_dic = {}
        for set_, split in self.splits.items():
            split_dic[set_] = {key: [values[s] for s in split] for key, values in features.items()}
        return split_dic

    def _split_indices(self, N: int)-> dict:
        indices = torch.randperm(N)
        self.splits = {}
        for set_, ratio in self.ratios.items():
            N_split = round(N * ratio)
            self.splits[set_] = indices[:N_split]
            indices = indices[N_split:]

        # Add remaining indices to last split - incase of rounding error
        if len(indices) > 0:
            self.splits[set_] = torch.cat((self.splits[set_], indices))

        print(f'Resulting split ratios: {[round(len(s) / N, 2) for s in self.splits.values()]}')
        
    def split_outcomes(self, outcomes: list)-> dict:
        outcomes_splits = {}
        for set_, split in self.splits.items():
            print(split)
            outcomes_splits[set_] = outcomes[split] 
        return outcomes_splits
    def save(self, dest: str):
        torch.save(self.splits, join(dest, 'splits.pt'))
