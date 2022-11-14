# Pytorch implementation of [Med_BERT](https://www.nature.com/articles/s41746-021-00455-y) based on Huggingface Transformers. 
The implementaiton is largely based on BERT, using ICD10 codes instead of words and visits instead of sentences. 
As pretraining tasks MLM and prolonged length of stay in the hospital (>7 days) are employed. The task will be fine-tuned for hospitalization/ICU prediction of COVID patients.
>Rasmy, Laila, et al. "Med-BERT: pretrained contextualized embeddings on large-scale structured electronic health records for disease prediction." NPJ digital medicine 4.1 (2021): 1-13.

## Reproducing

Run the following steps:

    If you want an example dataset:  
      python data\generate.py <num_patients> <save_name> 
    python data\tokenize_example_data.py <input_data_file>  
    python models\mlm_plos_pretraining.py <path_to_tokenized_data> <path_to_vocab_file> <path_to_save_model> <epochs> 
