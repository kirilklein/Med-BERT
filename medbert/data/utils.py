import glob
import os
from datetime import datetime
from os.path import join, split

import dateutil
import pandas as pd
import torch

from .creators import BaseCreator


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
    @staticmethod
    def detect_date_columns(df: pd.DataFrame):
        date_columns = []
        for col in df.columns:
            if isinstance(df[col], datetime):
                continue
            if 'TIME' in col.upper() or 'DATE' in col.upper():
                try:
                    first_non_na = df.loc[df[col].notna(), col].iloc[0]
                    dateutil.parser.parse(first_non_na)
                    date_columns.append(col)
                except:
                    continue
        return date_columns

class FeatureMaker():
    def __init__(self, config):
        self.config = config

        self.features = {
            'concept': [],
        }

        self.order = {
            'concept': 0,
            'background': -1
        }
        self.creators = {creator.id: creator for creator in BaseCreator.__subclasses__() if creator.id in self.config.features.keys()}
        self.pipeline = self.create_pipeline()
        

    def __call__(self, concepts: pd.DataFrame, patients_info: pd.DataFrame):
        for creator in self.pipeline:
            concepts = creator(concepts, patients_info)
            concepts['CONCEPT'] = concepts['CONCEPT'].astype(str)
        features = self.create_features(concepts, patients_info)

        return features
    
    def create_pipeline(self):
        # Pipeline creation
        pipeline = []
        for id in self.config.features:
            creator = self.creators[id](self.config.features)
            pipeline.append(creator)
            if getattr(creator, 'feature', None) is not None:
                self.features[creator.feature] = []

        # Reordering
        pipeline_creators = [creator.feature for creator in pipeline if hasattr(creator, 'feature')]
        for feature, pos in self.order.items():
            if feature in pipeline_creators:
                creator = pipeline.pop(pipeline_creators.index(feature))
                pipeline.insert(pos, creator)

        return pipeline

    def create_features(self, concepts: pd.DataFrame, patients_info: pd.DataFrame) -> tuple:
        # Add standard info
        for pid, patient in concepts.groupby('PID'):
            for feature, value in self.features.items():
                value.append(patient[feature.upper()].tolist())

        # Add outcomes if in config
        
        info_dict = patients_info.set_index('PID').to_dict('index')
        origin_point = datetime(**self.config.features.abspos)
        # Add outcomes
        if hasattr(self.config, 'outcomes'):
            outcomes = []
            for pid, patient in concepts.groupby('PID'):
                for outcome in self.config.outcomes:
                    patient_outcome = info_dict[pid][f'{outcome}']
                    if pd.isna(patient_outcome):
                        outcomes.append(None)
                    else:
                        outcomes.append((patient_outcome - origin_point).total_seconds() / 60 / 60)

            return self.features, outcomes
        else:
            return self.features


class Excluder():
    def __call__(self, features: dict, outcomes: dict=None, k: int = 2) -> pd.DataFrame:
        return self.exclude_by_k(features, outcomes, k=k)
    
    @staticmethod
    def exclude_by_k(features: dict, outcomes: dict=None, k: int = 2) -> pd.DataFrame:
        kept_indices = []
        for i, concepts in enumerate(features['concept']):
            unique_codes = set([code for code in concepts if not code.startswith('[')])
            if len(unique_codes) >= k:
                kept_indices.append(i)

        for key, values in features.items():
            features[key] = [values[i] for i in kept_indices]
        if outcomes:
            for key, values in outcomes.items():
                outcomes[key] = [values[i] for i in kept_indices]
        if outcomes:
            return features, outcomes
        else:
            return features
        
class Cleaner():
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, concepts: pd.DataFrame, patients_info: pd.DataFrame) -> pd.DataFrame:
        # drop nans
        concepts = concepts.dropna(subset=['CONCEPT'])
        # Remove patients from patients_info that are not in concepts
        patients_info = patients_info[patients_info['PID'].isin(concepts['PID'])]
        # Remove concepts which don't have PID in patients_info
        concepts = concepts[concepts['PID'].isin(patients_info['PID'])]
        return concepts, patients_info        


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
        self.censor_time = censor_time * 365.25 * 24 # censor time in hours

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