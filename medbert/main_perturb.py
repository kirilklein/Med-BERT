from os.path import split

import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from torch.optim import AdamW
from trainer.perturb import EHRPerturb
from transformers import BertConfig, BertForSequenceClassification


def main():
    with initialize(config_path='../configs'):
        cfg: dict = compose(config_name='perturb.yaml')

    print('Loading datasets...')
    train_dataset = torch.load(cfg.get('train_dataset', 'dataset.train'))
    val_dataset = torch.load(cfg.get('val_dataset', 'dataset.val'))

    print(f'Loading finetuned BERT model from {cfg.model_path}')
    model_dir = split(cfg.model_path)[0]
    config = BertConfig.from_pretrained(model_dir) 
    model = BertForSequenceClassification(config)
    model.load_state_dict(torch.load(cfg.model_path)['model_state_dict'], strict=False)

    model.eval() # we don't want to train the model, just use it to get the embeddings

    print('Setting up optimizer...')
    opt = cfg.get('optimizer', {})
    optimizer = AdamW(
        model.parameters(),
        lr=opt.get('lr', 1e-4),
        weight_decay=opt.get('weight_decay', 0.01),
        eps=opt.get('epsilon', 1e-8),
    )

    print('Starting training...')
    trainer = EHRPerturb( 
        bert_model=model, 
        optimizer=optimizer,
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        args=cfg.get('trainer_args', {}),
        cfg=cfg,
    )
    trainer.train()