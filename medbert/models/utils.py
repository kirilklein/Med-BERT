from transformers import Trainer
import torch
from tqdm import tqdm
import os 
from os.path import join, split

class CustomMLMTrainer(Trainer):
    def __init__(self, dataset, model, epochs, batch_size, save_path, lr=5e-5, 
                optimizer=torch.optim.AdamW, checkpoint_freq=5):
        self.dataset = dataset
        self.model = model
        self.epochs= epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optimizer
        self.save_path = save_path
        self.checkpoint_freq = checkpoint_freq
    def __call__(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device) # and move our model over to the selected device
        self.model.train() # activate training mode
        optim = self.optimizer(self.model.parameters(), lr=self.lr) # optimizer
        loader = torch.utils.data.DataLoader(self.dataset, 
                batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
            loop = tqdm(loader, leave=True)
            for batch in loop:
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
                print(loss.item())
                # update parameters
                optim.step()
                loop.set_description(f"epoch {epoch}/{self.epochs}")
                loop.set_postfix(loss=loss.item())
            if epoch%self.checkpoint_freq==0:
                print("Checkpoint")
                self.save_checkpoint(epoch, self.model, optim, loss)
            self.save_history(epoch, loss)

    def save_checkpoint(self, epoch, model, optim, loss):
        checkpoint_path = join(split(self.save_path)[0], 
                    f"checkpoint_{split(self.save_path)[1]}")
        torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optim.state_dict(),
            'loss':loss.item(),
        }, checkpoint_path)
    
    def save_history(self, epoch, loss):
        hist_path = join(split(self.save_path)[0], 
                f"history_{split(self.save_path)[1][:-3]}.txt")
        if not os.path.exists(hist_path):
            with open(hist_path, 'w') as f:
                f.write(f"epoch loss")    
        with open(hist_path, 'a+') as f:
            f.write(f"{str(epoch)} {str(loss.item):.4f}")
