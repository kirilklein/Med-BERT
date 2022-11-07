from transformers import AdamW
from transformers import BertForMaskedLM
from medbert.dataloader.MLM import MLMLoader
import torch
from tqdm import tqdm
import typer


def main(data_file : str = typer.Argument(..., help="Tokenized data"),
    vocab_file : str = typer.Argument(..., help=".pt vocab dic"),
    epochs : int = typer.Argument(...),
    max_len : int = typer.Option(512, help="maximum number of tokens in seq"),
    ):
    
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # and move our model over to the selected device
    model.to(device)
    # activate training mode
    model.train()

    # optimizer
    optim = AdamW(model.parameters(), lr=5e-5)

    data = torch.load(data_file)
    vocab = torch.load(vocab_file)
    
    dataset = MLMLoader(data, vocab, max_len)
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
   
    for epoch in range(epochs):
        print(f"epoch {epoch}/{epochs}")
        loop = tqdm(loader, leave=True)
        for i, batch in enumerate(loop):
            if i>1:
                break
            # initialize calculated grads
            optim.zero_grad()
            # put all tensore batches required for training
            input_ids = (batch['codes'] + batch['segments']).to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # process
            #mask_token_index = (input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
            logits = model(input_ids, attention_mask=attention_mask).logits
            predicted_token_id = logits[0, 2].argmax(axis=-1)
            print(predicted_token_id)
            #print(outputs.shape)