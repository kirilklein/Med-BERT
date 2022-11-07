from medbert.dataloader import tokenizer
import pickle

with open("data\\raw\\simulated\\example_data.pkl", 'rb') as f:
    data = pickle.load(f)

Tokenizer = tokenizer.EHRTokenizer()
tokenized_data = Tokenizer.batch_encode(data, max_len=20)
print("First two ")
print('pats: ', tokenized_data['pats'][:2])
print('los: ', tokenized_data['los'][:2])
print('codes: ', tokenized_data['codes'][:2])
print('segments: ', tokenized_data['segments'][:2])
