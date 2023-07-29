from transformers import BertForPreTraining
from features.embeddings import EhrEmbeddings

class EHRBertForPretraining(BertForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.bert.embeddings = EhrEmbeddings(config)
