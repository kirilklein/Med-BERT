import pandas as pd
from datetime import datetime
import itertools

class BaseCreator():
    def __init__(self, config: dict):
        self.config = config

    def __call__(self, concepts: pd.DataFrame, patients_info: pd.DataFrame):
        return self.create(concepts, patients_info)

class AgeCreator(BaseCreator):
    feature = id = 'age'
    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame):
        # Create PID -> BIRTHDATE dict
        birthdates = pd.Series(patients_info['BIRTHDATE'].values, index=patients_info['PID']).to_dict()
        # Calculate approximate age
        ages = (((concepts['TIMESTAMP'] - concepts['PID'].map(birthdates)).dt.days / 365.25) + 0.5).round()

        concepts['AGE'] = ages
        return concepts

class AbsposCreator(BaseCreator):
    """Add absolute position of concept in hours since origin point"""
    feature = id = 'abspos'
    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame):
        origin_point = datetime(**self.config.features.abspos)
        # Calculate hours since origin point
        abspos = (concepts['TIMESTAMP'] - origin_point).dt.total_seconds() / 60 / 60

        concepts['ABSPOS'] = abspos
        return concepts

class SegmentCreator(BaseCreator):
    feature = id = 'segment'
    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame):
        # segments = concepts.groupby('PID')['ADMISSION_ID'].transform(lambda x: pd.factorize(x)[0]+1)
        # rough estimation of segments based on 1d difference of timestamps
        concepts['TIMESTAMP'] = pd.to_datetime(concepts['TIMESTAMP'])
        concepts = concepts.sort_values(['PID', 'TIMESTAMP'])

        concepts['DIFF'] = concepts.groupby('PID')['TIMESTAMP'].diff()
        print("Using 3 days as a rough estimation of a new visit")
        concepts['NEW_SEGMENT'] = concepts['DIFF'] > pd.Timedelta(days=3) # 3 days is a rough estimation of a new segment
        concepts['SEGMENT'] = concepts.groupby('PID', group_keys=False)['NEW_SEGMENT'].apply(lambda x: x.astype(int).cumsum()) + 1
        concepts = concepts.drop(columns=['DIFF', 'NEW_SEGMENT'])

        # concepts['SEGMENT'] = segments
        return concepts
    
class LOSCreator(BaseCreator):
    feature = id = 'los'
    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame):
       # Create a groupby object
        grouped = concepts.groupby(['PID', 'SEGMENT'])

        # Calculate LOS for each group and assign it back to the original dataframe
        concepts['LOS'] = grouped['TIMESTAMP'].transform(lambda x: (x.max() - x.min()+pd.Timedelta(days=1))/ pd.Timedelta(days=1))
        return concepts

class BackgroundCreator(BaseCreator):
    id = 'background'
    prepend_token = "BG_"
    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame):
        # Create background concepts
        background = {
            'PID': patients_info['PID'].tolist() * len(self.config.features.background),
            'CONCEPT': itertools.chain.from_iterable(
                [(self.prepend_token + patients_info[col].astype(str)).tolist() for col in self.config.features.background])
        }

        if 'segment' in self.config.features:
            background['SEGMENT'] = 0

        if 'age' in self.config.features:
            background['AGE'] = -1

        if 'abspos' in self.config.features:
            origin_point = datetime(**self.config.features.abspos)
            start = (origin_point - patients_info['BIRTHDATE']).dt.total_seconds() / 60 / 60
            background['ABSPOS'] = start.tolist() * len(self.config.features.background)

        # background['AGE'] = -1

        # Prepend background to concepts
        background = pd.DataFrame(background)
        return pd.concat([background, concepts])


""" SIMPLE EXAMPLES """
class SimpleValueCreator(BaseCreator):
    id = 'value'
    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame):
        concepts['CONCEPT'] = concepts['CONCEPT'] + '_' + concepts['VALUE'].astype(str)

        return concepts
        
class QuartileValueCreator(BaseCreator):
    id = 'quartile_value'
    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame):
        quartiles = concepts.groupby('CONCEPT')['value'].transform(lambda x: pd.qcut(x, 4, labels=False))
        concepts['CONCEPT'] = concepts['CONCEPT'] + '_' + quartiles.astype(str)

        return concepts

