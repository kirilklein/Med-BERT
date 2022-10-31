import sys
import os
from os.path import join
script_dir = os.path.dirname(os.path.abspath(__file__))
medbert_dir = os.path.dirname(script_dir)
if medbert_dir not in sys.path:
    sys.path.append(os.path.dirname(medbert_dir))

import torch
from transformers import BertConfig
from torch.utils.data import DataLoader

from med_bert.data.preprocess import split_testsets
from med_bert.dataset.MLM import MLMDataset
from med_bert.model.model import BertEHRModel

import typer 


def MLM_pretraining(vocabulary_file, data_file):
    train, test = split_testsets(data_file)
    train_codes, train_segments = train
    test_codes, test_segments = test

    with open(vocabulary_file, 'rb') as f:
        vocabulary = torch.load(f)

    # Find max segments
    max_segments = max(max([max(segment) for segment in train_segments]), max([max(segment) for segment in test_segments]))

    config = BertConfig(
        vocab_size=len(vocabulary),              
        max_position_embeddings=512,    # Change?
        type_vocab_size=max_segments    # Should be smarter
    )

    model = BertEHRModel(config)
    

    train_dataset = MLMDataset(train_codes, train_segments, vocabulary)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    loss = torch.nn.CrossEntropyLoss(ignore_index=-100)

    for batch in train_dataloader:
        codes, segments, masked_seq, masked_pos = batch
        output = model(input_ids=masked_seq, token_type_ids=segments)


def main(vocab_file: str = typer.Argument(..., help="path to vocabulary file")):
    MLM_pretraining(vocab_file)
if __name__ == "__main__":
    typer.run(main)