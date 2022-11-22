from medbert.data import generate
import numpy as np

def test_generate(num_patients=10,
        min_num_visits=2,
        max_num_visits=5,
        min_num_codes_per_visit=1,
        max_num_codes_per_visit=5,
        min_los=1,
        max_los=30,
        num_codes=1000,):
    generator = generate.DataGenerator(
        num_patients, min_num_visits, max_num_visits, 
        min_num_codes_per_visit, max_num_codes_per_visit, 
        min_los, max_los, num_codes
    )
    data = [pat for pat in generator.simulate_data()]
    assert len(data) == num_patients
    assert np.min(np.array([min(d[1]) for d in data])) >= min_los
    assert np.max(np.array([max(d[1]) for d in data])) <= max_los
    codes = []
    for d in data:
        codes = codes + d[2]
    assert len(np.unique(np.array(codes), return_counts=False)) <= num_codes
    assert np.min(np.array([min(d[-1]) for d in data])) <= min_num_visits
    assert np.max(np.array([max(d[-1]) for d in data])) <= max_num_visits



