from hydra import initialize, compose
from omegaconf import OmegaConf
from data.utils import Splitter
from features.tokenizer import EHRTokenizer
from features.dataset import MLM_PLOS_Dataset
import torch
from os.path import join

def main():
    with initialize(config_path='../configs'):
        tokenizer_config = compose(config_name='tokenizer.yaml')
        dataset_config = compose(config_name='dataset_finetune.yaml')
    
    # save configs in dataset_config.working_dir!
    
    
    """
        Tokenize
        To dataset
    """

    # Overwrite nans and incorrect values
    features = torch.load(dataset_config.in_data) # list of lists
    outcomes = torch.load(dataset_config.in_outcomes) # list of lists
    splitter = Splitter(ratios=dataset_config.split_ratios)
    data_splits = splitter(features)
    outcomes_splits = splitter.split_outcomes(outcomes)
    splitter.save(dataset_config.out_dir)
    train, test, val = data_splits['train'], data_splits['test'], data_splits['val']
    train_out, test_out, val_out = outcomes_splits['train'], outcomes_splits['test'], outcomes_splits['val']
    assert False
    # Tokenize
    tokenizer = EHRTokenizer(config=tokenizer_config)
    train_encoded = tokenizer(train)
    tokenizer.freeze_vocabulary()
    tokenizer.save_vocab(join(dataset_config.out_dir, 'vocabulary.pt'))
    test_encoded = tokenizer(test)
    val_encoded = tokenizer(val)
    # To dataset
    train_dataset = MLM_PLOS_Dataset(train_encoded, vocabulary=tokenizer.vocabulary, **dataset_config)
    test_dataset = MLM_PLOS_Dataset(test_encoded, vocabulary=tokenizer.vocabulary, **dataset_config)
    val_dataset = MLM_PLOS_Dataset(val_encoded, vocabulary=tokenizer.vocabulary, **dataset_config)
    torch.save(train_dataset, join(dataset_config.out_dir, 'dataset.train'))
    torch.save(test_dataset, join(dataset_config.out_dir, 'dataset.test'))
    torch.save(val_dataset,  join(dataset_config.out_dir, 'dataset.val'))
    
    OmegaConf.save(config=tokenizer_config, f=join(dataset_config.out_dir, 'tokenizer.yaml'))
    OmegaConf.save(config=dataset_config, f=join(dataset_config.out_dir, 'dataset.yaml'))
    


if __name__ == '__main__':
    main()

