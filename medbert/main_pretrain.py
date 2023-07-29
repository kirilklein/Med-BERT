import torch
from features.dataset import MLM_PLOS_Dataset
from hydra import compose, initialize
from hydra.utils import instantiate
from torch.optim import AdamW
from trainer.trainer import EHRTrainer
from transformers import BertConfig, BertForPreTraining
from os.path import join


def main():
    with initialize(config_path='../configs'):
        cfg: dict = compose(config_name='pretrain.yaml')


        print("Dataset with MLM and prolonged length of stay")
    data_dir = cfg.data_dir
    train_encoded = torch.load(join(data_dir, 'train_encoded.pt'))
    val_encoded = torch.load(join(data_dir, 'val_encoded.pt'))
    vocabulary = torch.load(join(data_dir, 'vocabulary.pt'))

    train_dataset = MLM_PLOS_Dataset(train_encoded, vocabulary=vocabulary, **cfg.dataset)
    val_dataset = MLM_PLOS_Dataset(val_encoded, vocabulary=vocabulary, **cfg.dataset)

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