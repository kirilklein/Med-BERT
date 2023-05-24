import torch
from hydra import initialize, compose
from hydra.utils import instantiate
from torch.optim import AdamW

from trainer.trainer import EHRFineTune
from transformers import BertForSequenceClassification
from transformers import BertConfig
from os.path import split

def main():
    with initialize(config_path='../configs'):
        cfg: dict = compose(config_name='finetune.yaml')

    train_dataset = torch.load(cfg.get('train_dataset', 'dataset.train'))
    val_dataset = torch.load(cfg.get('val_dataset', 'dataset.val'))

    print('Loading BERT model...')
    model_dir = split(cfg.model_path)[0]
    
    # Load the config from file
    config = BertConfig.from_pretrained(model_dir) 
    model = BertForSequenceClassification(config)
    model.load_state_dict(torch.load(cfg.model_path)['model_state_dict'], strict=False)

    opt = cfg.get('optimizer', {})
    optimizer = AdamW(
        model.parameters(),
        lr=opt.get('lr', 1e-4),
        weight_decay=opt.get('weight_decay', 0.01),
        eps=opt.get('epsilon', 1e-8),
    )
    
    trainer = EHRFineTune( 
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