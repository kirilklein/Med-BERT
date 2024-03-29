import torch
from features.embeddings import PerturbedEHREmbeddings

class PerturbationModel(torch.nn.Module):
    def __init__(self, bert_model, cfg):
        super().__init__()
        self.cfg = cfg
        self.lambda_ = self.cfg.get('lambda', 0.1)
        self.bert_model = bert_model
        self.K = self.bert_model.bert.embeddings.concept_embeddings.weight.data.shape[1] # hidden dimensions?
        self.freeze_bert()
        self.noise_simulator = GaussianNoise(bert_model, cfg)

        self.embeddings = self.bert_model.bert.embeddings

        self.embeddings_perturb = PerturbedEHREmbeddings(self.bert_model.config)
        self.embeddings_perturb.set_parameters(self.embeddings)
        

    def forward(self, batch: dict):
        original_output = self.bert_forward_pass(batch)  
        perturbed_embeddings = self.embeddings_perturb(batch, self.noise_simulator)
        perturbed_output = self.bert_forward_pass(batch, perturbed_embeddings)
        loss = self.perturbation_loss(original_output, perturbed_output)
        outputs = ModelOutputs(predictions=original_output.logits, perturbed_predictions=perturbed_output.logits, loss=loss)
        return outputs

    def freeze_bert(self):
        for param in self.bert_model.parameters():
            param.requires_grad = False

    def bert_forward_pass(self, batch: dict, embeddings: torch.Tensor = None):
        """Forward pass through the model, optionally with embeddings"""
        concept = batch['concept'] if embeddings is None else None
        segment = batch['segment'] if 'segment' in batch and embeddings is None else None
        position = batch['age'] if 'age' in batch and embeddings is None else None
        output = self.bert_model(
            inputs_embeds=embeddings if embeddings is not None else None,
            input_ids=concept,
            attention_mask=batch['attention_mask'],
            token_type_ids=segment,
            position_ids=position,
        )
        return output  
    
    def perturbation_loss(self, original_output, perturbed_output):
        """Calculate the perturbation loss 
        as presented in https://proceedings.mlr.press/v97/guan19a.html"""
        logits = original_output.logits[:,1]
        perturbed_logits = perturbed_output.logits[:,1]
        squared_diff = (logits - perturbed_logits)**2
        sigmas = self.noise_simulator.sigmas
        first_term = -torch.log(sigmas).sum()
        second_term = 1/(self.K*self.lambda_)*(squared_diff/logits.std())
        loss = first_term + second_term
        return loss.mean()

class GaussianNoise(torch.nn.Module):
    """Simulate Gaussian noise with trainable sigma to add to the embeddings"""
    def __init__(self, bert_model, cfg):
        super().__init__()
        self.cfg = cfg
        self.bert_model = bert_model # BERT model
        self.min_age = self.cfg.stratification.get('min_age', 40)
        self.max_age = self.cfg.stratification.get('max_age', 80)
        self.age_window = self.cfg.stratification.get('age_window', 5)
        self.num_strata = self.get_num_strata()
        self.num_age_groups = self.num_strata // 2
        self.initialize()
        self.strata_dict = self.get_strata_dict()

    def initialize(self):
        """Initialize the noise module"""
        num_concepts = len(self.bert_model.bert.embeddings.concept_embeddings.weight.data)
        num_strata = self.get_num_strata()
        # initialize learnable parameters
        # the last column is to map all the ones outside the age range
        self.sigmas = torch.nn.Parameter(torch.ones(num_concepts, num_strata+1))

    def simulate_noise(self, concepts, indices: torch.Tensor, embeddings: torch.Tensor):
        """Simulate Gaussian noise using the sigmas"""
        extended_indices = indices.unsqueeze(-1)
        extended_concept = concepts.unsqueeze(-1)
        selected_sigmas = self.sigmas[extended_concept, extended_indices]
        # Reparameterization trick: Sample noise from standard normal, then scale by selected sigma
        std_normal_noise = torch.randn_like(embeddings)
        scaled_noise = std_normal_noise * selected_sigmas
        
        return scaled_noise

    def get_num_strata(self):
        """Calculate the number of strata for age based on min_age, max_age and age_window"""
        num_age_strata = int((self.max_age - self.min_age) / self.age_window)
        return 2 * num_age_strata # 2 genders
    
    def get_stratum_indices(self, age: torch.Tensor, gender: torch.Tensor):
        """Get the stratum indices for the batch"""
        age_mask = (age >= self.min_age) & (age <= self.max_age)
        # we map ages starting from min_age to 0, 1, 2, ... based on age_window
        age_strata = torch.floor_divide(age-self.min_age, self.age_window) 
        # groups run from 0 to num_age_groups-1, and then num_age_groups to 2*num_age_groups-1
        stratum_indices = age_strata + self.num_age_groups * gender
        stratum_indices = torch.where(age_mask, stratum_indices, -1) # by default everyone else is in the last stratum
        return stratum_indices
    
    def get_strata_dict(self):
        """Get the strata dictionary"""
        strata_dict = {i:{} for i in range(self.num_strata)}
        sample_batch = {}
        sample_age = torch.arange(0,110,1).repeat(2,1)
        sample_gender = torch.arange(0,2,1).repeat(110,1).transpose(0,1)
        for i in range(self.num_strata):
            stratum_indices = self.get_stratum_indices(sample_age, sample_gender)
            strata_dict[i]['age'] = sample_age[stratum_indices == i]
            strata_dict[i]['gender'] = sample_gender[stratum_indices == i]
        return strata_dict


class ModelOutputs:
    def __init__(self, predictions=None, perturbed_predictions=None, loss=None):
        self.predictions = predictions
        self.perturbed_predictions = perturbed_predictions
        self.loss = loss