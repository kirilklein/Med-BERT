from transformers import Trainer
from medbert.dataloader.embeddings import BertEmbeddings
import torch
from tqdm import tqdm
import os 
from os.path import join, split

class CustomPreTrainer(Trainer):
    def __init__(self, dataset, model, epochs, batch_size, save_path, lr=5e-5, 
                optimizer=torch.optim.AdamW, checkpoint_freq=5, from_checkpoint=False,
                config=None):
        self.dataset = dataset
        self.model = model
        self.epochs= epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optimizer
        self.save_path = save_path
        self.checkpoint_freq = checkpoint_freq
        self.from_checkpoint = from_checkpoint
        self.embeddings = BertEmbeddings(config=config)
    def __call__(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device) # and move our model over to the selected device
        optim = self.optimizer(self.model.parameters(), lr=self.lr) # optimizer
        # Here trainloader and tesloader should be defined

        # testloader = torch.utils.data.DataLoader(self.dataset, 
                # batch_size=self.batch_size,
        trainloader = torch.utils.data.DataLoader(self.dataset, 
                batch_size=self.batch_size, shuffle=True)
        if self.from_checkpoint:
            self.model, optim = self.load_from_checkpoint(self.model, optim)
        self.model.train() # activate training mode  
        for epoch in range(self.epochs):
            loop = tqdm(trainloader, leave=True)
            for batch in loop:
                # initialize calculated grads
                optim.zero_grad()
                # put all tensore batches required for training
                #TODO: pass segments as token_type_ids
                code_ids = batch['codes'].to(device)
                segment_ids = batch['segments'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                plos_label = batch['plos'].to(device)
                embedding_output = self.embeddings(code_ids, segment_ids)
                # process
                outputs = self.model(input_embeds=embedding_output, 
                            attention_mask=attention_mask, labels=labels,
                            next_sentence_label=plos_label)                
                # extract loss
                loss = outputs.loss
                # calculate loss for every parameter that needs grad update
                loss.backward()
                # update parameters
                optim.step()
                loop.set_description(f"epoch {epoch}/{self.epochs}")
                loop.set_postfix(loss=loss.item())
            self.save_history(epoch, loss)
            if epoch%self.checkpoint_freq==0:
                print("Checkpoint")
                self.save_checkpoint(epoch, self.model, optim, loss)
            

    def save_checkpoint(self, epoch, model, optim, loss):
        checkpoint_path = join(split(self.save_path)[0], 
                    f"{split(self.save_path)[1][:-3]}_checkpoint.pt")
        torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optim.state_dict(),
            'loss':loss.item(),
        }, checkpoint_path)
    
    def save_history(self, epoch, loss):
        hist_path = join(split(self.save_path)[0], 
                f"{split(self.save_path)[1][:-3]}_history.txt")
        if not os.path.exists(hist_path):
            with open(hist_path, 'w') as f:
                f.write(f"epoch loss\n")    
        with open(hist_path, 'a+') as f:
            f.write(f"{epoch} {loss.item():.4f}\n")

    def load_from_checkpoint(self, model, optim):
        checkpoint_path = join(split(self.save_path)[0], 
                    f"{split(self.save_path)[1][:-3]}_checkpoint.pt")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optim