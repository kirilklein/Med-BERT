# Pytorch implementation of [Med_BERT](https://www.nature.com/articles/s41746-021-00455-y) based on Huggingface Transformers. 
Using the bert model to get embeddings of medical concepts.
As pretraining tasks MLM and prolonged length of stay in the hospital (e.g. >7 days) are employed. 
>Rasmy, Laila, et al. "Med-BERT: pretrained contextualized embeddings on large-scale structured electronic health records for disease prediction." NPJ digital medicine 4.1 (2021): 1-13.

## Reproducing

Run the following steps:

    If you want an example dataset:  
      python data\generate.py <num_patients> <save_name> 
    Data should be provided as a dict of nested lists, where each outer list represents a patient:
      {
        'concept':[[c1,c2, c3,...], [...],[...],...], 
        'age':[[...],[...],...],
        'los':[[...],[...],...],
        'segment':[[...],[...],...],
        ...
      }
    You can also use https://github.com/kirilklein/ehr_preprocess.git to get data in this format from tabular data.

    - python main_data.py 
    - python main_pretrain.pt <path_to_tokenized_data> <path_to_vocab_file> <path_to_save_model> <epochs> 

