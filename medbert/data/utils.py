import glob
import os
from os.path import join, split

import pandas as pd
import torch


class ConceptLoader():
    def __call__(self, concepts=['diagnose', 'medication'], data_dir: str = 'formatted_data', patients_info: str = 'patients_info.csv'):
        return self.load(concepts=concepts, data_dir=data_dir, patients_info=patients_info)
    
    def load(self, concepts=['diagnose', 'medication'], data_dir: str = 'formatted_data', patients_info: str = 'patients_info.csv'):
        # Get all concept files
        concept_paths = os.path.join(data_dir, 'concept.*')
        path = glob.glob(concept_paths)
        # Filter out concepts files
        path = [p for p in path if (split(p)[1]).split('.')[1] in concepts]
        
        # Load concepts
        concepts = pd.concat([self._read_file(p) for p in path], ignore_index=True).drop_duplicates()
        
        concepts = concepts.sort_values('TIMESTAMP')

        # Load patient data
        patient_path = os.path.join(data_dir, patients_info)
        patients_info = self._read_file(patient_path)

        return concepts, patients_info

    def _read_file(self, file_path: str) -> pd.DataFrame:
        file_type = file_path.split(".")[-1]
        if file_type == 'csv':
            df = pd.read_csv(file_path)
        elif file_type == 'parquet':
            df = pd.read_parquet(file_path)

        for col in self.detect_date_columns(df):
            df[col] = df[col].apply(lambda x: x[:10] if isinstance(x, str) else x)
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[col] = df[col].dt.tz_localize(None)

        return df

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

class Censor():
    def __init__(self, censor_time: float = 0, )-> None:
        """Abspos before outcome to censor"""
        self.censor_time = censor_time

    def __call__(self, features: dict, outcomes: list)-> dict:
        return self.censor_features(features, outcomes)
    
    def censor_features(self, features: dict, outcomes: list)-> dict:
        """
        Censor features before outcome
        """
        censored_features = {}
        censored_outcomes = []
        censored_patients = []
        for patient, pat_outcome in self._patient_iterator(features, outcomes):
            censor_index = self._find_censor_index(patient, pat_outcome)
            if censor_index == 0: # remove patients with no observations before censor_time
                continue
            censored_patient = {key: v[:censor_index] for key, v in patient.items()}
            censored_patients.append(censored_patient)
            censored_outcomes.append(1 if pat_outcome < torch.inf else 0)

        censored_features = {key: [patient[key] for patient in censored_patients] for key in censored_patients[0]}
        return censored_features, censored_outcomes
        
    def _find_censor_index(self, patient, pat_outcome):
        """Censor index is the index of the last observation before the outcome - self.censor_time"""
        censor_abspos = pat_outcome-self.censor_time
        if censor_abspos < patient['abspos'][0]:
            return 0
        elif (torch.tensor(patient['abspos']) > censor_abspos).sum() == 0: # If no observations after censor time
            return len(patient['abspos'])
        else:
            censor_index = (torch.tensor(patient['abspos']) <= censor_abspos).nonzero()[-1].item()
            return censor_index
    
    def _patient_iterator(self, features: dict, outcomes: list)-> tuple:
        for i in range(len(features['concept'])):
            yield {key: values[i] for key, values in features.items()}, outcomes[i]