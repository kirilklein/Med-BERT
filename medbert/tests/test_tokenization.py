from medbert.features import tokenizer
import pickle

with open("data\\raw\\simulated\\example_data.pkl", 'rb') as f:
    data = pickle.load(f)

Tokenizer = tokenizer.EHRTokenizer()
tokenized_data = Tokenizer.batch_encode(data, truncation=20)
print("First two ")
print('ids: ', tokenized_data['ids'][:2])
print('los: ', tokenized_data['los'][:2])
print('codes: ', tokenized_data['codes'][:2])
print('segment_ids: ', tokenized_data['segment_ids'][:2])
