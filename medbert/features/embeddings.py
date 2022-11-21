from torch import nn

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, segment, age
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, 
                                            config.hidden_size, 
                                            padding_idx=config.pad_token_id)
        self.segment_embeddings = nn.Embedding(config.seg_vocab_size, 
                                            config.hidden_size,
                                            padding_idx=config.pad_token_id)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, code_ids, seg_ids):
        embeddings = self.word_embeddings(code_ids)
        segment_embed = self.segment_embeddings(seg_ids)
        embeddings = embeddings + segment_embed
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings