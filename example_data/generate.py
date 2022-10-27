import numpy as np
import pickle as pkl
import multiprocessing as mp
import string 
import random





class DataGenerater(super):
    def __init__(self, n_patients, min_num_codes, max_num_codes, min_num_visits, max_num_visits):
        self.n_samples = n_patients
    
    def generate_ICD10_history(self):
        pass
    def generate_randomICD10_codes(self, n):
        letters = np.random.choice([char for char in string.ascii_uppercase], size=n, replace=True)
        numbers = np.random.choice(np.arange(1000), size=n, replace=True)
        codes = [letter + str(number).zfill(3)[:2] + '.' + str(number)[-1] for letter, number in zip(letters, numbers)]
        return codes
    def simulate_visit(self):
        pass

def test():
    generator = DataGenerater(1000, 5, 10, 5, 10)
    print(generator.generate_randomICD10_codes(100))
if __name__ == '__main__':
    test()