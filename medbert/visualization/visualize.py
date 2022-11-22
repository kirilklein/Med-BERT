import os
from os.path import split
import torch
from medbert.features.dataset import PatientDatum
from medbert.features.embeddings import BertEmbeddings
from bertviz import model_view, head_view


class BertViz:
    def __init__(self, model_dir, data_path, vocab_path=None):
        self.model = torch.load(model_dir)
        self.model.eval()
        self.data = torch.load(data_path)
        if isinstance(vocab_path, type(None)):
            vocab_path = os.path.join(split(split(data_path)[0])[0], 'vocab', split(data_path)[1])
        self.vocab = torch.load(vocab_path)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}        

    def embedding_view(self, pat_id):
        patient_datum = PatientDatum(self.data, self.vocab, pat_id)
        patient_loader = torch.utils.data.DataLoader(patient_datum, batch_size=1, shuffle=False) # type: ignore #
        input = [d for d in patient_loader][0]
        embeddings = BertEmbeddings(config=self.model.config)
        embedded_input = embeddings(input['codes'], input['segments'])
        model_outputs = self.model(inputs_embeds=embedded_input, output_attentions=True)
        attention = model_outputs[-1] 
        input_tokens = [self.inv_vocab[i] for i in self.data['codes'][pat_id]]
        return attention, input_tokens

    def head_view(self, pat_id):
        attention, input_tokens = self.embedding_view(pat_id)
        head_view(attention, input_tokens)
        
    def model_view(self, pat_id):    
        attention, input_tokens = self.embedding_view(pat_id)
        model_view(attention, input_tokens)
        
    
    
