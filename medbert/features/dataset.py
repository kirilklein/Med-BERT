from torch.utils.data import Dataset
import torch




class BaseDataset(Dataset):
    def __init__(self, features: dict, **kwargs):
        self.features = features
        self.kwargs = kwargs
        self.max_segments = self.get_max_segments()

    def __len__(self):
        return len(self.features['concept'])

    def __getitem__(self, index):
        return {key: values[index] for key, values in self.features.items()}

    def get_max_segments(self):
        if 'segment' not in self.features:
            raise ValueError('No segment data found. Please add segment data to dataset')
        return max([max(segment) for segment in self.features['segment']]) + 1

    def load_vocabulary(self, vocabulary):
        if isinstance(vocabulary, str):
            return torch.load(vocabulary)
        elif isinstance(vocabulary, dict):
            return vocabulary
        else:
            raise TypeError(f'Unsupported vocabulary input {type(vocabulary)}')
        
    def convert_to_long(self, patient):
        """
        Converts all tensors in the patient to longs except abspos
        """
        return {
            key: value.long() for key, value in patient.items() if (isinstance(value, torch.Tensor) and (key != 'abspos'))}


class MLM_PLOS_Dataset(BaseDataset):
    def __init__(self, features: dict, **kwargs):
        super().__init__(features, **kwargs)
        self.plos = True
        self.min_los = self.kwargs.get('min_los', 0)

        if self.min_los==0:
            self.plos = False
        self.vocabulary = self.load_vocabulary(self.kwargs.get('vocabulary', 'vocabulary.pt'))
        self.masked_ratio = self.kwargs.get('masked_ratio', 0.3)
        if kwargs.get('ignore_special_tokens', True):
            self.n_special_tokens = len([token for token in self.vocabulary if token.startswith('[')])
        else:
            self.n_special_tokens = 0

    def __getitem__(self, index):
        patient = super().__getitem__(index)

        masked_concepts, target = self._mask(patient)
        if self.plos:
            patient['plos'] = self.get_plos(patient)
        patient['concept'] = masked_concepts
        patient['target'] = target
        patient = self.convert_to_long(patient)
        return patient

    def get_plos(self, patient: dict):
        """
        Returns the PLOS of the patient
        """
        return (patient['los'].clone().detach()>=self.kwargs['min_los']).any().long()
        
   
    def _mask(self, patient: dict):
        concepts = patient["concept"]

        N = len(concepts)

        # Initialize
        masked_concepts = torch.clone(concepts)
        target = torch.ones(N, dtype=torch.long) * -100

        # Apply special token mask and create MLM mask
        eligible_mask = masked_concepts >= self.n_special_tokens
        eligible_concepts = masked_concepts[eligible_mask]  # Ignore special tokens
        rng = torch.rand(len(eligible_concepts))  # Random number for each token
        masked = rng < self.masked_ratio  # Mask tokens with probability masked_ratio

        # Get masked MLM concepts
        selected_concepts = eligible_concepts[masked]  # Select set % of the tokens
        adj_rng = rng[masked].div(self.masked_ratio)  # Fix ratio to 0-100 interval

        # Operation masks
        rng_mask = adj_rng < 0.8  # 80% - Mask token
        rng_replace = (0.8 <= adj_rng) & (
            adj_rng < 0.9
        )  # 10% - replace with random word
        # rng_keep = adj_rng >= 0.9                             # 10% - keep token (Redundant)

        # Apply operations (Mask, replace, keep)
        selected_concepts = torch.where(
            rng_mask, self.vocabulary["[MASK]"], selected_concepts
        )  # Replace with [MASK]
        selected_concepts = torch.where(
            rng_replace,
            torch.randint(
                self.n_special_tokens, len(self.vocabulary), (len(selected_concepts),)
            ),
            selected_concepts,
        )  # Replace with random word
        # selected_concepts = torch.where(rng_keep, selected_concepts, selected_concepts)       # Redundant

        # Update outputs (nonzero for double masking)
        target[eligible_mask.nonzero()[:, 0][masked]] = eligible_concepts[
            masked
        ]  # Set "true" token
        masked_concepts[
            eligible_mask.nonzero()[:, 0][masked]
        ] = selected_concepts  # Sets new concepts

        return masked_concepts, target


class BinaryOutcomeDataset(BaseDataset): 
    def __init__(self, features: dict, outcomes: torch.tensor, vocabulary: {}, **kwargs):
        super().__init__(features, **kwargs)
        self.outcomes = outcomes
        self.vocabulary = vocabulary
    def __getitem__(self, index):
        patient = super().__getitem__(index)
        patient['target'] = self.outcomes[index].item()
        patient = self.convert_to_long(patient)
        # turn all into longs
        return patient
        