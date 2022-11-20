import 

class Encoding():
    def __init__(self, model_name, model_path, tokenizer_path, max_len):
        self.model_name = model_name
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.max_len = max_len

    def get_tokenizer(self):
        if self.tokenizer_path is not None:
            tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)
        else:
            tokenizer = BertTokenizer.from_pretrained(self.model_name)
        return tokenizer

    def get_model(self):
        if self.model_path is not None:
            model = BertModel.from_pretrained(self.model_path)
        else:
            model = BertModel.from_pretrained(self.model_name)
        return model

    def get_encoding(self, text):
        tokenizer = self.get_tokenizer()
        model = self.get_model()
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)
        features = last_hidden_states[0][:, 0, :].numpy()
        return features

    def get_encoding_batch(self, texts):
        tokenizer = self.get_tokenizer()
        model = self.get_model()
        encoded = tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)
        features = last_hidden_states[0][:, 0, :].numpy()
        return features