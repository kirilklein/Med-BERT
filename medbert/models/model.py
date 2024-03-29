from features.embeddings import EhrEmbeddings
from transformers import (BertForMaskedLM, BertForPreTraining,
                          BertForSequenceClassification)


class EHRBertForPretraining(BertForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.bert.embeddings = EhrEmbeddings(config)

class EHRBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.bert.embeddings = EhrEmbeddings(config)

class EHRBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.bert.embeddings = EhrEmbeddings(config)
