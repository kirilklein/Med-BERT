from medbert.features import tokenizer
import pickle

with open("data\\raw\\simulated\\example_data.pkl", 'rb') as f:
    data = pickle.load(f)

Tokenizer = tokenizer.EHRTokenizer()
tokenized_data = Tokenizer.batch_encode(data)
print(tokenized_data[0])
