from os.path import split

import torch
from hydra import compose, initialize
from torch.optim import AdamW
from trainer.perturb import EHRSimpleTrainer
from models.perturb import PerturbationModel
from models.model import EHRBertForSequenceClassification
from transformers import BertConfig
from features.dataset import BinaryOutcomeDataset
from os.path import join

def main():
    with initialize(config_path='../configs'):
        cfg: dict = compose(config_name='perturb.yaml')

    print('Loading datasets...')
    data_dir = cfg.data_dir
    train_encoded = torch.load(join(data_dir, 'train_encoded.pt'))
    train_out = torch.load(join(data_dir, 'train_outcomes.pt'))
    val_encoded = torch.load(join(data_dir, 'val_encoded.pt'))
    val_out = torch.load(join(data_dir, 'val_outcomes.pt'))
    vocabulary = torch.load(join(data_dir, 'vocabulary.pt'))

    print("Dataset with Binary Outcome")
    train_dataset = BinaryOutcomeDataset(train_encoded, train_out, vocabulary=vocabulary, **cfg.get('dataset', {}))
    val_dataset = BinaryOutcomeDataset(val_encoded, val_out, vocabulary=vocabulary, **cfg.get('dataset', {}))

    print(f'Loading finetuned BERT model from {cfg.model_path}')
    model_dir = split(cfg.model_path)[0]
    config = BertConfig.from_pretrained(model_dir) 
    bert_model = EHRBertForSequenceClassification(config)
    load_result = bert_model.load_state_dict(torch.load(cfg.model_path)['model_state_dict'], strict=False)
    print("missing keys", load_result.missing_keys)
    bert_model.eval() # we don't want to train the bertmodel, just use it to get the embeddings

    model = PerturbationModel(bert_model, cfg)

    print('Setting up optimizer...')
    opt = cfg.get('optimizer', {})
    optimizer = AdamW(
        model.parameters(),
        lr=opt.get('lr', 1e-4),
        weight_decay=opt.get('weight_decay', 0.01),
        eps=opt.get('epsilon', 1e-8),
    )

    print('Starting training...')
    trainer = EHRSimpleTrainer( 
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