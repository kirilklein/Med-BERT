import numpy as np
import torch
import string 
import random
import typer
from os.path import split, join

class DataGenerator(super):
    def __init__(self, num_patients, min_num_visits, max_num_visits, 
        min_num_codes_per_visit, max_num_codes_per_visit, min_los, 
        max_los, num_codes):
        """
        Simulates data as lists:
            [pid, los_ls, all_visit_codes, visit_nums]
        min_los, max_los: Length os stay in the hospital,
        num_codes: total number of ICD10 codes to generate
        """
        self.num_patients = num_patients
        self.min_num_codes_per_visit = min_num_codes_per_visit
        self.max_num_codes_per_visit = max_num_codes_per_visit
        self.min_num_visits = min_num_visits
        self.max_num_visits = max_num_visits
        self.min_los = min_los
        self.max_los = max_los
        self.num_codes = num_codes
        self.codes = self.generate_randomICD10_codes(self.num_codes)

    def __call__(self):
        concepts_dic = {k:[] for k in ['los', 'concept', 'segment', 'age', 'abspos']}
        outcomes = []
        for pid in range(self.num_patients):
            out_dic = self.generate_ICD10_history()
            for k, v in out_dic.items():
                concepts_dic[k].append(v)
            outcomes.append(self.simulate_outcome(out_dic['abspos']))
        return concepts_dic, torch.tensor(outcomes)
    
    def generate_ICD10_history(self):
        
        num_visits = np.random.randint(self.min_num_visits, self.max_num_visits)
        num_codes_per_visit_ls = np.random.randint(self.min_num_codes_per_visit, 
            self.max_num_codes_per_visit, 
            size=num_visits)
        los_nums = np.random.randint(self.min_los, self.max_los, size=num_visits)
        
        # codes = self.generate_randomICD10_codes(self.num_codes)
        all_visit_codes = np.random.choice(self.codes, 
            size=np.sum(num_codes_per_visit_ls), replace=True).tolist()
        
        visit_nums = np.arange(1, num_visits+1) # should start with 1!
        visit_nums = np.repeat(visit_nums, num_codes_per_visit_ls).tolist()
        los_nums = np.repeat(los_nums, num_codes_per_visit_ls).tolist()
        # simulate random increasing ages in size of all_visit_codes where age within a visit stays the same
        ages = self.simulate_ages(visits=visit_nums, max_age=110)
        abspos = self.simulate_abspos(ages)
        return {'los':los_nums, 'concept':all_visit_codes, 'segment':visit_nums, 'age':ages, 'abspos':abspos}

    def generate_randomICD10_codes(self, n):
        letters = np.random.choice([char for char in string.ascii_uppercase], 
            size=n, replace=True)
        numbers = np.random.choice(np.arange(1000), size=n, replace=True)
        codes = [letter + str(number).zfill(3)[:2] + '.' + str(number)[-1] for \
            letter, number in zip(letters, numbers)]
        return codes

    def simulate_ages(self, visits, max_age=110):
        ages = []
        current_age = random.randint(0, max_age-10)
        for i in range(len(visits)):
            if i > 0 and visits[i] != visits[i-1]:
                current_age += random.randint(0, max_age - current_age)
            ages.append(current_age)
        return ages
    def simulate_abspos(self, ages):
        abspos = []
        current_pos = random.random()*100 + float(ages[0])
        for i in range(len(ages)):
            if i>0:
                current_pos += random.random()*0.1 + float(ages[i] - ages[i-1])
            abspos.append(current_pos)
            
        return abspos

    def simulate_outcome(self, abspos):
        outcome = random.randint(0, 1)
        if outcome == 1:
            outcome = random.choice(abspos)
        else:
            outcome = torch.inf
        return outcome
        

def main(num_patients : int = typer.Argument(...), 
        save_name: str = typer.Argument(..., 
        help="name of the file to save the data to, should end with .pkl"),
        min_num_visits: int = typer.Option(2),
        max_num_visits: int = typer.Option(10),
        min_num_codes_per_visit: int = typer.Option(1),
        max_num_codes_per_visit: int = typer.Option(5),
        min_los: int = typer.Option(1),
        max_los: int = typer.Option(30),
        num_codes: int = typer.Option(500)):
    generator = DataGenerator(num_patients, min_num_visits, max_num_visits, 
        min_num_codes_per_visit, max_num_codes_per_visit, 
        min_los, max_los, num_codes)
    patients, outcomes = generator()
    torch.save(patients, save_name)
    torch.save(outcomes, join(split(save_name)[0], 'synthetic_outcomes.pt'))


if __name__ == '__main__':
    typer.run(main)