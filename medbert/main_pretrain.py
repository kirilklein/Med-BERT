from os.path import join

import torch
from features.dataset import MLM_PLOS_Dataset
from hydra import compose, initialize
from models.model import EHRBertForPretraining, EHRBertForMaskedLM
from torch.optim import AdamW
from trainer.trainer import EHRTrainer
from transformers import BertConfig

def get_model(bertconfig, cfg):
    if cfg.dataset.get('plos', False):
        print("Using EHRBertForPretraining (MLM+PLOS)")
        return EHRBertForPretraining(bertconfig)
    else:
        print("Using EHRBertForMaskedLM (MLM)")
        return EHRBertForMaskedLM(bertconfig)

def main():
    with initialize(config_path='../configs'):
        cfg: dict = compose(config_name='pretrain.yaml')

    data_dir = cfg.data_dir
    train_encoded = torch.load(join(data_dir, 'train_encoded.pt'))
    val_encoded = torch.load(join(data_dir, 'val_encoded.pt'))
    vocabulary = torch.load(join(data_dir, 'vocabulary.pt'))

    train_dataset = MLM_PLOS_Dataset(train_encoded, vocabulary=vocabulary, **cfg.dataset)
    val_dataset = MLM_PLOS_Dataset(val_encoded, vocabulary=vocabulary, **cfg.dataset)

    bertconfig = BertConfig(
            vocab_size=len(train_dataset.vocabulary),
            type_vocab_size=int(train_dataset.max_segments),
            **cfg.get('model', {}),
        )
    model = get_model(bertconfig, cfg)
    opt = cfg.get('optimizer', {})
    optimizer = AdamW(
        model.parameters(),
        lr=opt.get('lr', 1e-4),
        weight_decay=opt.get('weight_decay', 0.01),
        eps=opt.get('epsilon', 1e-8),
    )
    trainer = EHRTrainer( 
        model=model, 
        optimizer=optimizer,
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        args=cfg.get('trainer_args', {}),
        cfg=cfg,
    )
    trainer.train()


if __name__ == '__main__':
    main()