import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig, OmegaConf
from trainer import EHRTrainer
from tqdm import tqdm

class EHRPerturb(EHRTrainer):
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
        
    def train(self):
        dataloader = self.setup_training()
        for epoch in range(self.args.epochs):
            train_loop = tqdm(enumerate(dataloader), total=len(dataloader))
            train_loop.set_description(f'Train {epoch}')
            epoch_loss = []
            step_loss = 0
            for i, batch in train_loop:
                original_output = self.forward_pass(batch)
                pass

    def forward_pass(self, batch: dict):
        batch = self.to_device(batch)
        return self.model(
            input_ids=batch['concept'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['segment'] if 'segment' in batch else None,
            position_ids=batch['age'].long() if 'age' in batch else None,
        )
# TODO: continue here