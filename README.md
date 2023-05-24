# Pytorch implementation of [Med_BERT](https://www.nature.com/articles/s41746-021-00455-y) using BERT from Huggingface Transformers. 
BERT is used to get embeddings of medical concepts.\\
As pretraining tasks MLM and prolonged length of stay in the hospital (e.g. >7 days) are employed. 
>Rasmy, Laila, et al. "Med-BERT: pretrained contextualized embeddings on large-scale structured electronic health records for disease prediction." NPJ digital medicine 4.1 (2021): 1-13.

## Reproducing
We use example data generated with https://github.com/synthetichealth/synthea.git and formatted using https://github.com/kirilklein/ehr_preprocess.git.
The data can be found in data/raw/synthea{size}
The main scripts are :    
    - python main_data_pretrain.py (config: dataset_pretrain.yaml)
    - python main_pretrain.py (configs: model.yaml, trainer.yaml)
    - python main_finetune.py (configs: finetune.yaml)
    - python main_perturb.py (configs: perturb.yaml)

