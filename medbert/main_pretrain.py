import torch
from hydra import initialize, compose
from hydra.utils import instantiate
from torch.optim import AdamW

from trainer.trainer import EHRTrainer
from transformers import BertForPreTraining
from transformers import BertConfig

def main():
    with initialize(config_path='../configs'):
        cfg: dict = compose(config_name='pretrain.yaml')

    train_dataset = torch.load(cfg.get('train_dataset', 'dataset.train'))
    val_dataset = torch.load(cfg.get('val_dataset', 'dataset.val'))
    model = BertForPreTraining(
        BertConfig(
            vocab_size=len(train_dataset.vocabulary),
            type_vocab_size=int(train_dataset.max_segments),
            **cfg.get('model', {}),
        )
    )
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