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
        dataset_config = compose(config_name='dataset.yaml')
    
    # save configs in dataset_config.working_dir!
    data_dir = "../data/processed/synthetic"
    
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
    patient = {key: values[3] for key, values in encoded_train.items()}
    print(patient)
    print({key: len(values) for key, values in patient.items()})

    # To dataset
    train_dataset = MLM_PLOS_Dataset(encoded_train, vocabulary=tokenizer.vocabulary, **dataset_config)
    # test_dataset = MLMDataset(encoded_test, vocabulary=tokenizer.vocabulary)
    # val_dataset = MLMDataset(encoded_val, vocabulary=tokenizer.vocabulary)
    torch.save(train_dataset, join(data_dir, 'dataset.train'))
    patient = train_dataset.__getitem__(2)
    print({key: len(values) for key, values in patient.items()})
    print(patient)
    # torch.save(test_dataset, join(data_dir, 'dataset.test'))
    # torch.save(val_dataset,  join(data_dir, 'dataset.val'))
    
    OmegaConf.save(config=tokenizer_config, f=join(dataset_config.working_dir, 'tokenizer.yaml'))
    OmegaConf.save(config=dataset_config, f=join(dataset_config.working_dir, 'dataset.yaml'))
    


if __name__ == '__main__':
    main()

