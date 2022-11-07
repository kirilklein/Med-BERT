from transformers import BertForMaskedLM
from medbert.dataloader.MLM import MLMLoader
from transformers import Trainer
import torch
from tqdm import tqdm
import typer


def main(data_file : str = typer.Argument(..., help="Tokenized data"),
    vocab_file : str = typer.Argument(..., help=".pt vocab dic"),
    save_dir : str = typer.Argument(...),
    epochs : int = typer.Argument(...),
    batch_size : int = typer.Option(16),
    max_len : int = typer.Option(512, help="maximum number of tokens in seq"),
    ):
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    data = torch.load(data_file)
    vocab = torch.load(vocab_file)
    dataset = MLMLoader(data, vocab, max_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    trainer = CustomMLMTrainer(dataset, model, epochs, batch_size, save_dir)
    trainer()

class CustomMLMTrainer(Trainer):
    def __init__(self, dataset, model, epochs, batch_size, save_dir, lr=5e-5, 
                optimizer=torch.optim.AdamW):
        self.dataset = dataset
        self.model = model
        self.epochs= epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optimizer
        self.save_dir = save_dir
    def __call__(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device) # and move our model over to the selected device
        self.model.train() # activate training mode
        optim = self.optimizer(self.model.parameters(), lr=self.lr) # optimizer
        loader = torch.utils.data.DataLoader(self.dataset, 
                batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
            loop = tqdm(loader, leave=True)
            for i, batch in enumerate(loop):
                # initialize calculated grads
                optim.zero_grad()
                # put all tensore batches required for training
                input_ids = (batch['codes'] + batch['segments']).to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                # process
                outputs = self.model(input_ids, 
                            attention_mask=attention_mask, labels=labels)                
                # extract loss
                loss = outputs.loss
                # calculate loss for every parameter that needs grad update
                loss.backward()
                # update parameters
                optim.step()
                loop.set_description(f"epoch {epoch}/{self.epochs}")
                loop.set_postfix(loss=loss.item())
