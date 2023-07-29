import os
from os.path import join

import torch
from data.utils import ConceptLoader, Excluder, FeatureMaker, Splitter
from features.tokenizer import EHRTokenizer
from hydra import compose, initialize
from omegaconf import OmegaConf


def main():
    with initialize(config_path='../configs'):
        cfg = compose(config_name='data_pretrain.yaml')
    
    # save configs in dataset_config.working_dir!
    
    """
        Loads data
        Creates features
        Excludes patients with <k concepts
        Splits data
        Tokenize
        To dataset
    """

    print("Loading concepts...")
    concepts, patients_info = ConceptLoader()(**cfg.loader)
   
    print("Creating feature sequences")
    features = FeatureMaker(cfg)(concepts, patients_info)
    
    print("Exclude patients with <k concepts")
    features = Excluder()(features, k=cfg.min_concepts)

    print("Splitting data")
    splitter = Splitter(ratios=cfg.split_ratios)
    splits = splitter(features)
    if not os.path.exists(cfg.out_dir):
        os.makedirs(cfg.out_dir)
    splitter.save(cfg.out_dir)
    train, test, val = splits['train'], splits['test'], splits['val']

    torch.save(train, join(cfg.out_dir, 'train.pt'))
    torch.save(test, join(cfg.out_dir, 'test.pt'))
    torch.save(val, join(cfg.out_dir, 'val.pt'))
    
    print("Tokenizing")
    tokenizer = EHRTokenizer(config=cfg.tokenizer)
    train_encoded = tokenizer(train)
    tokenizer.freeze_vocabulary()
    tokenizer.save_vocab(join(cfg.out_dir, 'vocabulary.pt'))
    test_encoded = tokenizer(test)
    val_encoded = tokenizer(val)

    torch.save(train_encoded, join(cfg.out_dir, 'train_encoded.pt'))
    torch.save(test_encoded, join(cfg.out_dir, 'test_encoded.pt'))
    torch.save(val_encoded, join(cfg.out_dir, 'val_encoded.pt'))

    print("Saving config")
    OmegaConf.save(config=cfg, f=join(cfg.out_dir, 'data.yaml'))
    


if __name__ == '__main__':
    main()

