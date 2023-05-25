import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from trainer.trainer import EHRTrainer



class EHRSimpleTrainer(EHRTrainer):
    def __init__(self, 
        model: torch.nn.Module,
        train_dataset: Dataset = None,
        test_dataset: Dataset = None,
        val_dataset: Dataset = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler.StepLR = None,
        args: dict = {},
        cfg: DictConfig = None,
    ):
        super().__init__(model, train_dataset, test_dataset, val_dataset, optimizer, scheduler, args, cfg)
        # Freeze the model


    def train(self, **kwargs):
        self.update_attributes(**kwargs)
        self.validate_training()

        dataloader = self.setup_training()

        for epoch in range(self.args.epochs):
            train_loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Train Loop')
            epoch_loss = []
            step_loss = 0
            for i, batch in train_loop:
                self.optimizer.zero_grad()
                batch = self.to_device(batch)
                # Train step
                outputs = self.forward_pass(batch)
                outputs.loss.backward() # calculate gradients
                self.optimizer.step()
                step_loss += outputs.loss.item()
                tqdm.write(f'Train loss {(i+1)}: {step_loss /(i+1)}')
                epoch_loss.append(step_loss)
                
                if self.scheduler is not None:
                    self.scheduler.step()
            # Validate (returns None if no validation set is provided)
            val_loss = self.validate()
             # Save epoch checkpoint
            self.save_checkpoint(id=f'epoch{epoch}_end', train_loss=epoch_loss, val_loss=val_loss, final_step_loss=epoch_loss[-1])
            # Print epoch info
            self.info(f'Epoch {epoch} train loss: {sum(epoch_loss) / (len(train_loop))}')
            self.info(f'Epoch {epoch} val loss: {val_loss}')

    def forward_pass(self, batch: dict):
        return self.model.forward(batch)

    def validate(self):
        if self.val_dataset is None:
            return None, None

        self.model.eval()
        dataloader = DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=self.args.collate_fn)
        val_loop = tqdm(dataloader, total=len(dataloader), desc='Validation')
        val_loss = 0
        for batch in val_loop:
            batch = self.to_device(batch)
            with torch.no_grad():
                outputs = self.forward_pass(batch)
                val_loss += outputs.loss.item()

        return val_loss / len(val_loop)

    def save_setup(self):
        OmegaConf.save(config=self.args, f=os.path.join(self.run_folder, 'config.yaml'))

