from transformers import Trainer
from medbert.dataloader.embeddings import BertEmbeddings
import torch
from tqdm import tqdm
import os 
from os.path import join, split

class CustomPreTrainer(Trainer):
    def __init__(self, train_dataset, val_dataset, model, epochs, batch_size, save_path, lr=5e-5, 
                optimizer=torch.optim.AdamW, checkpoint_freq=5, from_checkpoint=False,
                config=None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
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
        
        trainloader = torch.utils.data.DataLoader(self.val_dataset, 
                batch_size=self.batch_size, shuffle=True)
        valloader = torch.utils.data.DataLoader(self.val_dataset, 
                        batch_size=self.batch_size,shuffle=True)
        if self.from_checkpoint:
            self.model, optim = self.load_from_checkpoint(self.model, optim)
        self.model.train() # activate training mode  
        for epoch in range(self.epochs):
            train_loop = tqdm(trainloader, leave=True)
            for batch in train_loop:
                # initialize calculated grads
                optim.zero_grad()
                # put all tensore batches required for training
                code_ids = batch['codes'].to(device)
                segment_ids = batch['segments'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                plos_label = batch['plos'].to(device)
                embedding_output = self.embeddings(code_ids, segment_ids)
                # process
                outputs = self.model(inputs_embeds=embedding_output, 
                            attention_mask=attention_mask, labels=labels,
                            next_sentence_label=plos_label)                
                # extract loss
                train_loss = outputs.loss
                # calculate loss for every parameter that needs grad update
                train_loss.backward()
                # update parameters
                optim.step()
                train_loop.set_description(f"epoch {epoch}/{self.epochs}")
                train_loop.set_postfix(loss=train_loss.item())
            # validation
            # TODO: validation every few epochs
            val_loop = tqdm(valloader, leave=True)
            self.model.eval()
            for val_batch in val_loop:
                # put all tensor batches required for training
                code_ids = val_batch['codes'].to(device)
                segment_ids = val_batch['segments'].to(device)
                attention_mask = val_batch['attention_mask'].to(device)
                labels = val_batch['labels'].to(device)
                plos_label = val_batch['plos'].to(device)
                embedding_output = self.embeddings(code_ids, segment_ids)
                # process
                outputs = self.model(inputs_embeds=embedding_output, 
                            attention_mask=attention_mask, labels=labels,
                            next_sentence_label=plos_label)                
                # extract loss
                val_loss = outputs.loss
                val_loop.set_postfix({"val_lss":val_loss.item()})
            self.save_history(epoch, train_loss.item(), val_loss.item())
            if epoch%self.checkpoint_freq==0:
                print("Checkpoint")
                self.save_checkpoint(epoch, self.model, optim, 
                                    train_loss.item(), val_loss.item())
            

    def save_checkpoint(self, epoch, model, optim, train_loss, val_loss):
        checkpoint_path = join(split(self.save_path)[0], 
                    f"{split(self.save_path)[1][:-3]}_checkpoint.pt")
        torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optim.state_dict(),
            'train_loss':train_loss,
            'val_loss':val_loss
        }, checkpoint_path)
    
    def save_history(self, epoch, train_loss, val_loss=None):
        hist_path = join(split(self.save_path)[0], 
                f"{split(self.save_path)[1][:-3]}_history.txt")
        if not os.path.exists(hist_path):
            with open(hist_path, 'w') as f:
                f.write(f"epoch loss\n")    
        with open(hist_path, 'a+') as f:
            f.write(f"{epoch} {train_loss:.4f} {val_loss:.4f}\n")

    def load_from_checkpoint(self, model, optim):
        checkpoint_path = join(split(self.save_path)[0], 
                    f"{split(self.save_path)[1][:-3]}_checkpoint.pt")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optim