Pytorch implementation of MedBERT (Med-BERT: pretrained contextualized embeddings on largescale structured electronic health records for disease prediction Rasmy et. al) based on Huggingface Transformers. 
The implementaiton is largely based on BERT, using ICD10 codes instead of words and visits instead of sentences. 
As pretraining tasks MLM and length of stay in the hospital are employed. The task will be fine-tuned for hospitalization/ICU prediction of COVID patients.
In contrast to the original paper, no priority will be assigned to codes within a visit.

# generate.py: generate example data
# tokenize_example_data.py: turn icd codes into integers
# dataloader.MLM.py: mask out codes and pad