from hydra import initialize, compose
from omegaconf import OmegaConf
from data.utils import Splitter, ConceptLoader, FeatureMaker, Excluder
from features.tokenizer import EHRTokenizer
from features.dataset import MLM_PLOS_Dataset
import torch
from os.path import join

def main():
    with initialize(config_path='../configs'):
        cfg = compose(config_name='data_pretrain.yaml')
    
    # save configs in dataset_config.working_dir!
    
    
    """
        Loads data
        Finds outcomes
        Censors
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
    splitter.save(cfg.out_dir)
    train, test, val = splits['train'], splits['test'], splits['val']

    # Tokenize
    tokenizer = EHRTokenizer(config=cfg.tokenizer)
    train_encoded = tokenizer(train)
    tokenizer.freeze_vocabulary()
    tokenizer.save_vocab(join(cfg.out_dir, 'vocabulary.pt'))
    test_encoded = tokenizer(test)
    val_encoded = tokenizer(val)
    # To dataset
    train_dataset = MLM_PLOS_Dataset(train_encoded, vocabulary=tokenizer.vocabulary, **cfg.dataset)
    test_dataset = MLM_PLOS_Dataset(test_encoded, vocabulary=tokenizer.vocabulary, **cfg.dataset)
    val_dataset = MLM_PLOS_Dataset(val_encoded, vocabulary=tokenizer.vocabulary, **cfg.dataset)
    torch.save(train_dataset, join(cfg.out_dir, 'dataset.train'))
    torch.save(test_dataset, join(cfg.out_dir, 'dataset.test'))
    torch.save(val_dataset,  join(cfg.out_dir, 'dataset.val'))
    
    OmegaConf.save(config=cfg, f=join(cfg.out_dir, 'data.yaml'))
    


if __name__ == '__main__':
    main()

