from hydra import initialize, compose
from omegaconf import OmegaConf
from data.utils import Splitter
from features.tokenizer import EHRTokenizer
import torch
from os.path import join

def main():
    with initialize(config_path='../configs'):
        tokenizer_config = compose(config_name='tokenizer.yaml')
        dataset_config = compose(config_name='dataset.yaml')
    
    # save configs in dataset_config.working_dir!
    
    
    """
        Tokenize
        To dataset
    """

    # Overwrite nans and incorrect values
    features = torch.load('../data/features/synthetic.pt') # list of lists
    train, val, test = Splitter(ratios=dataset_config.split_ratios)(features)
    # Tokenize
    print(tokenizer_config)
    tokenizer = EHRTokenizer(config=tokenizer_config)
    encoded_train = tokenizer(train)
    tokenizer.freeze_vocab()
    tokenizer.save_vocab(join(dataset_config.working_dir, 'vocabulary.pt'))
    encoded_test = tokenizer(test)
    encoded_val = tokenizer(val)
    print(encoded_test['concepts'])
    # To dataset
    # train_dataset = MLMDataset(encoded_train, vocabulary=tokenizer.vocabulary)
    # test_dataset = MLMDataset(encoded_test, vocabulary=tokenizer.vocabulary)
    # val_dataset = MLMDataset(encoded_val, vocabulary=tokenizer.vocabulary)
    # torch.save(train_dataset, 'dataset.train')
    # torch.save(test_dataset, 'dataset.test')
    # torch.save(val_dataset, 'dataset.val')
    
    OmegaConf.save(config=tokenizer_config, f=join(dataset_config.working_dir, 'tokenizer.yaml'))
    OmegaConf.save(config=dataset_config, f=join(dataset_config.working_dir, 'dataset.yaml'))
    


if __name__ == '__main__':
    main()

