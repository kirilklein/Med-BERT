from hydra import initialize, compose
from omegaconf import OmegaConf
from data.utils import Splitter, ConceptLoader, FeatureMaker, Excluder, Censor, Cleaner
from features.tokenizer import EHRTokenizer
from features.dataset import BinaryOutcomeDataset
import torch
from os.path import join
import os

def main():
    with initialize(config_path='../configs'):
        cfg = compose(config_name='data_finetune.yaml')
    
    # save configs in dataset_config.working_dir!
    
    """
        Loads data
        Censors
        Creates features
        Excludes patients with <k concepts
        Splits data
        Tokenize
        To dataset
    """

    print("Loading concepts...")
    concepts, patients_info = ConceptLoader()(**cfg.loader)
    print("Cleaning concepts...")
    concepts, patients_info = Cleaner(cfg)(concepts, patients_info)
    print("Creating feature sequences")
    features, outcomes = FeatureMaker(cfg)(concepts, patients_info)
    
  
    print("Censoring")
    features, outcomes = Censor(censor_time=cfg.censor_time)(features, outcomes) # censor time in years
    
    print("Exclude patients with <k concepts")
    features, outcomes = Excluder()(features, outcomes, k=cfg.min_concepts)

    splitter = Splitter(ratios=cfg.split_ratios)
    data_splits = splitter(features)
    outcomes_splits = splitter.split_outcomes(torch.tensor(outcomes))
    if not os.path.exists(cfg.out_dir):
        os.makedirs(cfg.out_dir)
    splitter.save(cfg.out_dir)
    train, test, val = data_splits['train'], data_splits['test'], data_splits['val']
    train_out, test_out, val_out = outcomes_splits['train'], outcomes_splits['test'], outcomes_splits['val']
    
    print("Tokenizing")
    tokenizer = EHRTokenizer(config=cfg.tokenizer)
    train_encoded = tokenizer(train)
    tokenizer.freeze_vocabulary()
    tokenizer.save_vocab(join(cfg.out_dir, 'vocabulary.pt'))
    test_encoded = tokenizer(test)
    val_encoded = tokenizer(val)
  
    print("Dataset with Binary Outcome")
    train_dataset = BinaryOutcomeDataset(train_encoded, train_out, vocabulary=tokenizer.vocabulary, **cfg.dataset)
    test_dataset = BinaryOutcomeDataset(test_encoded, test_out, vocabulary=tokenizer.vocabulary, **cfg.dataset)
    val_dataset = BinaryOutcomeDataset(val_encoded, val_out, vocabulary=tokenizer.vocabulary, **cfg.dataset)
    print("Saving datasets")
    torch.save(train_dataset, join(cfg.out_dir, 'dataset.train'))
    torch.save(test_dataset, join(cfg.out_dir, 'dataset.test'))
    torch.save(val_dataset,  join(cfg.out_dir, 'dataset.val'))
    print("Saving configs")
    OmegaConf.save(config=cfg, f=join(cfg.out_dir, 'data.yaml'))
    


if __name__ == '__main__':
    main()

