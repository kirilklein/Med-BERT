from transformers import Trainer, BertConfig, BertForPreTraining
from medbert.features.embeddings import BertEmbeddings
from medbert.common import common, pytorch
import torch
from tqdm import tqdm
import os 
from os.path import join, split
import json
import numpy as np


class CustomPreTrainer(Trainer):
    def __init__(self, train_dataset, val_dataset, model, epochs, 
                batch_size, model_dir, lr=5e-5, optimizer=torch.optim.AdamW, 
                checkpoint_freq=5, from_checkpoint=False, config=None, args=None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = model
        self.epochs= epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optimizer
        self.model_dir = model_dir
        self.checkpoint_freq = checkpoint_freq
        self.from_checkpoint = from_checkpoint
        self.config = config
        self.embeddings = BertEmbeddings(config=config)
        self.args = args
    def __call__(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        optim = self.optimizer(self.model.parameters(), lr=self.lr) # optimizer
        trainloader = torch.utils.data.DataLoader(self.train_dataset,   # type: ignore
                batch_size=self.batch_size, shuffle=True)
        valloader = torch.utils.data.DataLoader(self.val_dataset,   # type: ignore
                        batch_size=self.batch_size*2, shuffle=True)
        if self.from_checkpoint:
            self.model, optim = self.load_from_checkpoint(self.model, optim)
        self.model.to(device) # and move our model over to the selected device
        self.model.train() # activate training mode  
        for epoch in range(self.epochs):
            train_loop = tqdm(trainloader, leave=True)
            for i, batch in enumerate(train_loop):
                # initialize calculated grads
                optim.zero_grad()
                # put all tensore batches required for training
                batch = pytorch.batch_to_device(batch, device)
                # get embeddings
                embedding_output = self.embeddings(batch['codes'], batch['segments'])
                # process
                outputs = self.model(inputs_embeds=embedding_output, 
                            attention_mask=batch['attention_mask'], 
                            labels=batch['labels'],
                            next_sentence_label=batch['plos'])                
                # extract loss
                train_loss = outputs.loss
                # calculate loss for every parameter that needs grad update
                train_loss.backward()
                # update parameters
                optim.step()
                train_loop.set_description(f"epoch {epoch}/{self.epochs} Training")
                train_loop.set_postfix(loss=train_loss.item())
                self.save_history(epoch, i, train_loss.item())
            # validation
            # TODO: validation every few epochs
            val_loop = tqdm(valloader, leave=True)
            self.model.eval()
            val_loss_avg = 0
            with torch.no_grad():
                for val_batch in val_loop:
                    # put all tensor batches required for training
                    val_batch = pytorch.batch_to_device(val_batch, device)
                    # get embeddings
                    embedding_output = self.embeddings(val_batch['codes'], 
                                                        val_batch['segments'])
                    # process
                    outputs = self.model(inputs_embeds=embedding_output, 
                                attention_mask=val_batch['attention_mask'], 
                                labels=val_batch['labels'],
                                next_sentence_label=val_batch['plos'])                
                    # extract loss
                    val_loss = outputs.loss
                    val_loss_avg += val_loss.item()/len(valloader)
                    val_loop.set_description(f"Validation")
                    val_loop.set_postfix({"val_loss":val_loss.item()})
            self.save_history(epoch, i, train_loss.item(), val_loss_avg) # type: ignore
            if epoch%self.checkpoint_freq==0:
                print("Checkpoint")
                self.save_checkpoint(epoch, self.model, optim, 
                                    train_loss.item(), val_loss.item()) # type: ignore
            #TODO introduce training scheduler

    def save_checkpoint(self, epoch, model, optim, train_loss, val_loss):
        checkpoint_path = join(self.model_dir, "checkpoint.pt") 
        torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optim.state_dict(),
            'train_loss':train_loss,
            'val_loss':val_loss,
            'config':self.config,
        }, checkpoint_path)
    
    def save_history(self, epoch, batch, train_loss, val_loss=-100):
        hist_path = join(self.model_dir, "history.txt")
        common.create_directory(self.model_dir)
        if not os.path.exists(hist_path):
            with open(hist_path, 'w') as f:
                f.write(f"epoch batch train_loss val_loss\n")    
        with open(hist_path, 'a+') as f:
            f.write(f"{epoch} {batch} {train_loss:.4f} {val_loss:.4f}\n")

    def load_from_checkpoint(self, model, optim):
        checkpoint_path = join(self.model_dir, "checkpoint.pt")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if not isinstance(optim, type(None)):
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            return model, optim
        else:
            return model
    
    def save_model(self):
        common.create_directory(self.model_dir)
        torch.save(self.model, join(self.model_dir, "model.pt"))
        print(f"Trained model saved to {self.model_dir}")
        with open(join(self.model_dir, 'config.json'), 'w') as f:
            json.dump(vars(self.config), f)
        with open(join(self.model_dir, 'log.json'), 'w') as f:
            json.dump(self.args, f)


class Encoder(CustomPreTrainer):
    def __init__(self, dataset, model_dir, from_checkpoint=False, batch_size=128):
        self.model_dir = model_dir
        with open(join(self.model_dir, 'config.json'), 'r') as f:
            config_dic = json.load(f)
        self.config = BertConfig(**config_dic)
        super().__init__(train_dataset=dataset, val_dataset=None, model=None,
                        epochs=None, batch_size=batch_size, model_dir=model_dir,   
                        from_checkpoint=from_checkpoint, config=self.config)
        if not from_checkpoint:
            print(f"Load saved model from {model_dir}")
            self.model = torch.load(join(model_dir, 'model.pt'))
        else:
            model = BertForPreTraining(self.config)
            self.model = self.load_from_checkpoint(model, None)
    def __call__(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if self.from_checkpoint:
            self.model = self.load_from_checkpoint(self.model, None)
        self.model.to(device) # type: ignore # and move our model over to the selected device
        self.model.eval()  # type: ignore
        loader = torch.utils.data.DataLoader(self.train_dataset,  # type: ignore
                                    batch_size=self.batch_size, shuffle=False)  
        loop = tqdm(loader, leave=True)                        
        pat_vecs = []
        pat_ids = []
        for batch in loop:
            # put all tensore batches required for training
            batch = pytorch.batch_to_device(batch, device)
            # get embeddings
            embedding_output = self.embeddings(batch['codes'], batch['segments'])
            # process
            outputs = self.model(inputs_embeds=embedding_output,   # type: ignore
                        attention_mask=batch['attention_mask'],  
                        labels=batch['labels'],
                        next_sentence_label=batch['plos'], 
                        output_hidden_states=True) # type: ignore                
            loop.set_description(f"Inference")
            for hidden_state in outputs.hidden_states[-1]:
                pat_vec = hidden_state[hidden_state!=0].mean(dim=1)
                pat_vecs.append(pat_vec)
                pat_ids = pat_ids + batch['pat_id']
            # print('len', len(outputs.hidden_states))
            # print(outputs.hidden_states[0].shape)
            # print(outputs.hidden_states[-1].shape)
            # break
            # print(len(outputs[0]))
            # print(outputs[0].shape)
        pat_vecs = np.concatenate(pat_vecs, axis=0)
        print(len(pat_ids), pat_vecs.shape)
        return pat_ids, pat_vecs
