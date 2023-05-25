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
        # Freeze the model
        for param in self.model.parameters():
            param.requires_grad = False
        self.noise_simulator = GaussianNoise(self.model, self.cfg)

    def train(self):
        dataloader = self.setup_training()
        for epoch in range(self.args.epochs):
            train_loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Train Loop')
            epoch_loss = []
            step_loss = 0
            for i, batch in train_loop:
                # Accumulate gradients
                step_loss += self.train_step(batch).item()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()
            self.save_checkpoint(id=f'epoch{epoch}_step{(i+1)}', train_loss=step_loss)
        # Validate (returns None if no validation set is provided)
        val_loss = self.validate()
    
    
    def train_step(self, batch: dict):
        original_output = self.bert_forward_pass(batch)        
        perturbed_embeddings = self.noise_simulator(batch)
        perturbed_output = self.bert_forward_pass(batch, perturbed_embeddings)
        loss = self.perturbation_loss(original_output, perturbed_output)
        loss.backward()
        return loss

    def validate(self):
        if self.val_dataset is None:
            return None, None

        self.model.eval()
        dataloader = DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=self.args.collate_fn)
        val_loop = tqdm(dataloader, total=len(dataloader), desc='Validation')
        val_loss = 0
        for batch in val_loop:
            original_output = self.bert_forward_pass(batch)        
            perturbed_embeddings = self.noise_simulator(batch)
            perturbed_output = self.bert_forward_pass(batch, perturbed_embeddings)
            val_loss += self.perturbation_loss(original_output, perturbed_output)

        return val_loss / len(val_loop)

    def perturbation_loss(self, original_output, perturbed_output):
        """Calculate the perturbation loss"""
        return 0

    def bert_forward_pass(self, batch: dict, embeddings: torch.Tensor = None):
        """Forward pass through the model, optionally with embeddings"""
        batch = self.to_device(batch)
        concept = batch['concept'] if embeddings is None else None
        segment = batch['segment'] if 'segment' in batch and embeddings is None else None
        position = batch['age'] if 'age' in batch and embeddings is None else None
        output = self.model(
            input_embeds=embeddings if embeddings is not None else None,
            input_ids=concept,
            attention_mask=batch['attention_mask'],
            token_type_ids=segment,
            position_ids=position,
        )
        return output
        

class GaussianNoise(torch.module):
    """Simulate Gaussian noise with trainable sigma to add to the embeddings"""
    def __init__(self, model, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = model # BERT model
        self.num_strata = self.get_num_strata()
        self.num_age_groups = self.num_strata // 2
        self.sigmas = self.initialize()

    def __call__(self, batch: dict)->torch.Tensor:
        """Simulate Gaussian noise for the batch"""
        embeddings = self.get_summed_embeddings(batch)
        indices = self.get_stratum_indices(batch, self.num_age_groups)
        gaussian_noise = self.simulate_noise(indices)
        perturbed_embeddings = embeddings + gaussian_noise
        return perturbed_embeddings

    def initialize(self):
        """Initialize the noise module"""
        num_concepts = len(self.model.bert.embeddings.word_embeddings.weight.data)
        num_strata = self.get_num_strata()
        # initialize learnable parameters
        self.sigmas = torch.nn.Parameter(torch.ones(num_concepts, num_strata)) # are ones ok?

    def simulate_noise(self, indices: torch.Tensor):
        """Simulate Gaussian noise using the sigmas"""
        sigmas = self.sigmas[indices]
        gaussian_noise = torch.normal(mean=torch.ones_like(sigmas), std = sigmas)
        return gaussian_noise

    def get_num_strata(self):
        """Calculate the number of strata for age based on min_age, max_age and age_window"""
        min_age = self.cfg.stratification.get('min_age', 40)
        max_age = self.cfg.stratification.get('max_age', 80)
        age_window = self.cfg.stratification.get('age_window', 5)
        num_age_strata = int((max_age - min_age) / age_window)
        return 2 * num_age_strata # 2 genders
    
    def get_stratum_indices(self, batch):
        """Get the stratum indices for the batch"""
        age_strata = torch.floor_divide(batch['age'], self.num_age_groups)
        stratum_indices = age_strata * (batch['gender']+1)+self.num_age_groups
        return stratum_indices
    
    def get_summed_embeddings(self, batch: dict):
        """Get the summed embeddings of concept, segment and position embeddings"""
        batch = self.to_device(batch)
        embeddings = self.model.bert.embeddings
        concept_emb = embeddings.word_embeddings(batch['concept']) 
        token_type_emb = embeddings.token_type_embeddings(batch['segment']) if 'segment' in batch else torch.zeros_like(concept_emb)
        position_emb = embeddings.position_embeddings(batch['age']) 
        summed_embeddings = concept_emb + token_type_emb + position_emb
        return summed_embeddings