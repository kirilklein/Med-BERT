from medbert.features import tokenizer
import pickle
import os

def test_tokenizer(max_len=20):
    assert os.path.exists("data\\raw\\example_data.pkl"), "Generate example data first" 
    with open("data\\raw\\example_data.pkl", 'rb') as f:
        data = pickle.load(f)
    num_patients = len(data)
    Tokenizer = tokenizer.EHRTokenizer()
    tokenized_data = Tokenizer.batch_encode(data, max_len=max_len)
    for k, v in tokenized_data.items():
        assert k in ['pats', 'los', 'codes', 'segments']
        assert len(v) == num_patients
        if k != 'pats':
            flat_ls = [item for sublist in v for item in sublist]
            assert all((isinstance(entry, int) for entry in flat_ls))  
