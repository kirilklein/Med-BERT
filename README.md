# Pytorch implementation of MedBERT (Med-BERT: pretrained contextualized embeddings on largescale structured electronic health records for disease prediction Rasmy et. al) based on Huggingface Transformers. 
The implementaiton is largely based on BERT, using ICD10 codes instead of words and visits instead of sentences. 
As pretraining tasks MLM and prolonged length of stay in the hospital (>7 days) are employed. The task will be fine-tuned for hospitalization/ICU prediction of COVID patients.
In contrast to the original paper, no priority will be assigned to codes within a visit.
## Reproducing

Run the following steps:
    If you want an example dataset:  
        python medbert\data\generate.py num_patients save_name  
    python medbert\data\tokenize_example_data.py input_data_file  
    python 