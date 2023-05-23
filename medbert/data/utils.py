import torch


class Splitter():
    def __init__(self, ratios: list = [0.7, 0.2, 0.1]) -> None:
        self.ratios = ratios

    def __call__(self, features: dict, ):
        return self.split_features(features)

    def split_features(self, features: dict, ):
        """
        Split features into train, validation and test sets
        """
        if round(sum(self.ratios), 5) != 1:
            raise ValueError(f'Sum of ratios ({self.ratios}) != 1 ({round(sum(ratios), 5)})')
        torch.manual_seed(0)

        N = len(features['concept'])

        splits = self._split_indices(N, self.ratios)

        for split in splits:
            yield {key: [values[s] for s in split] for key, values in features.items()}

    def _split_indices(self, N: int, ratios: list):
        indices = torch.randperm(N)
        splits = []
        for ratio in ratios:
            N_split = round(N * ratio)
            splits.append(indices[:N_split])
            indices = indices[N_split:]

        # Add remaining indices to last split - incase of rounding error
        if len(indices) > 0:
            splits[-1] = torch.cat((splits[-1], indices))

        print(f'Resulting split ratios: {[round(len(s) / N, 2) for s in splits]}')
        return splits