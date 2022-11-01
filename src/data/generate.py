import numpy as np
import pickle as pkl
import multiprocessing as mp
import string 
import typer





class DataGenerater(super):
    def __init__(self, num_patients, min_num_visits, max_num_visits, 
        min_num_codes_per_visit, max_num_codes_per_visit, min_los, max_los, num_codes):
        self.num_patients = num_patients
        self.min_num_codes_per_visit = min_num_codes_per_visit
        self.max_num_codes_per_visit = max_num_codes_per_visit
        self.min_num_visits = min_num_visits
        self.max_num_visits = max_num_visits
        self.min_los = min_los
        self.max_los = max_los
        self.num_codes = num_codes

    def generate_ICD10_history(self, pid):
        codes = self.generate_randomICD10_codes(self.num_codes)
        num_visits = np.random.randint(self.min_num_visits, self.max_num_visits)
        num_codes_per_visit_ls = np.random.randint(self.min_num_codes_per_visit, self.max_num_codes_per_visit, 
            size=num_visits)
        los_ls = np.random.randint(self.min_los, self.max_los, size=num_visits).tolist()
        all_visit_codes = np.random.choice(codes, size=np.sum(num_codes_per_visit_ls), replace=True).tolist()
        visit_nums = np.arange(0, num_visits)
        visit_nums = np.repeat(visit_nums, num_codes_per_visit_ls).tolist()
        return [pid, los_ls, all_visit_codes, visit_nums]

    def generate_randomICD10_codes(self, n):
        letters = np.random.choice([char for char in string.ascii_uppercase], size=n, replace=True)
        numbers = np.random.choice(np.arange(1000), size=n, replace=True)
        codes = [letter + str(number).zfill(3)[:2] + '.' + str(number)[-1] for letter, number in zip(letters, numbers)]
        return codes

    def simulate_data(self):
        for pid in range(self.num_patients):
            yield self.generate_ICD10_history(pid)
def main(num_patients : int = typer.Argument(...), save_name: str = typer.Argument(..., 
        help="name of the file to save the data to, should end with .pkl")):
    generator = DataGenerater(num_patients, 2, 10, 1, 10, 1, 30, 10000)
    with open(save_name, 'wb') as f:
        pkl.dump([hist for hist in generator.simulate_data()], f)
def test():
    generator = DataGenerater(100, 2, 10, 1, 10, 1, 30, 10000)
    print([hist for hist in generator.simulate_data()][-1])
if __name__ == '__main__':
    typer.run(main)