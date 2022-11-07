from transformers import BertForMaskedLM
from medbert.dataloader.MLM import MLMLoader
import torch
from tqdm import tqdm
import typer


def main(data_file : str = typer.Argument(..., help="Tokenized data"),
    vocab_file : str = typer.Argument(..., help=".pt vocab dic"),
    epochs : int = typer.Argument(...),
    batch_size : int = typer.Option(16),
    max_len : int = typer.Option(512, help="maximum number of tokens in seq"),
    ):
    
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device) # and move our model over to the selected device
    model.train() # activate training mode

    optim = torch.optim.AdamW(model.parameters(), lr=5e-5) # optimizer

    data = torch.load(data_file)
    vocab = torch.load(vocab_file)
    dataset = MLMLoader(data, vocab, max_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
   
    for epoch in range(epochs):
        print(f"epoch {epoch}/{epochs}")
        loop = tqdm(loader, leave=True)
        for i, batch in enumerate(loop):
            if i>2:
                break
            # initialize calculated grads
            optim.zero_grad()
            # put all tensore batches required for training
            input_ids = (batch['codes'] + batch['segments']).to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # process
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            # extract loss
            loss = outputs.loss
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            loop.set_postfix(loss=loss.item())
            