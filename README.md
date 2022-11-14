# Pytorch implementation of [Med_BERT](https://www.nature.com/articles/s41746-021-00455-y) based on Huggingface Transformers. 
The implementaiton is largely based on BERT, using ICD10 codes instead of words and visits instead of sentences. 
As pretraining tasks MLM and prolonged length of stay in the hospital (>7 days) are employed. The task will be fine-tuned for hospitalization/ICU prediction of COVID patients.
In contrast to the original paper, no priority will be assigned to codes within a visit.
## Reproducing

Run the following steps:
                       -If you want an example dataset:  
                                                      -python medbert\data\generate.py num_patients save_name  
                       -python medbert\data\tokenize_example_data.py input_data_file  
                       -python 