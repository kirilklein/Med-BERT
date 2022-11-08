from transformers import Trainer
import torch
from tqdm import tqdm


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
