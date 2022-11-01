from cmath import pi
from torch.utils.data import Dataset
import random
import torch
import tqdm
from torch import nn
from transformers import BertConfig, BertModel
from transformers import AdamW

class MLMDataset(Dataset):
    def __init__(self, codes, segments, vocab):
        self.codes = codes
        self.segments = segments
        self.vocab = vocab

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, index):

        seq = self.codes[index]
        N = len(seq)
        masked_seq = [self.vocab['[CLS]']] + [0]*(N-1)
        masked_pos = [-100] * N           # -1 is auto-ignored in loss function
        
        for i in range(1, N):
            rng = random.random()
            if rng < 0.15:              # Select 15% of the tokens
                rng /= 0.15             # Fix ratio to 0-100 interval
                if rng < 0.8:           # 80% - Mask token
                    masked_seq[i] = self.vocab['[MASK]']
                elif 0.8 <= rng < 0.9:  # 10% - replace with random word
                    masked_seq[i] = random.randint(1, max(self.vocab.values()))
                else:                   # 10% - Do nothing        
                    masked_seq[i] = seq[i]
                masked_pos[i] = 0       # Unignore this token in loss function
            else:
                masked_seq[i] = seq[i]   
        

        return self.codes[index], self.segments[index], masked_seq, masked_pos

def pipeline(trainset, testset, vocab_file='example_data/example_data_vocab.pt', 
    criterion= nn.CrossEntropyLoss(), num_epochs=1, batch_size=32):
    vocab = torch.load(vocab_file)
    configuration = BertConfig(vocab_size=len(vocab), max_position_embeddings=100)
    model = BertModel(configuration)
    # activate training mode
    model.train()
    # initialize optimizer
    optim = AdamW(model.parameters(), lr=1e-4)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)                                          
    epochs = 2

    for epoch in range(num_epochs):
        running_loss = 0.0
        # setup loop with TQDM and dataloader
        loop = tqdm(trainloader, leave=True)
        for i, batch in tqdm(enumerate(trainloader, 0)):
            inputs, labels = batch
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all tensor batches required for training
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            # process
            outputs = model(input_ids, attention_mask=attention_mask,
                            labels=labels)

            # forward + backward + optimize
        
            # extract loss
            loss = outputs.loss
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
if __name__ == "__main__":
    pipeline()
    


