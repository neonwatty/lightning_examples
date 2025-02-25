---
dataset_info:
  config_name: en-es
  features:
  - name: id
    dtype: string
  - name: translation
    dtype:
      translation:
        languages:
        - en
        - es
  splits:
  - name: train
    num_bytes: 13529.294426019043
    num_examples: 50
  download_size: 10514
  dataset_size: 13529.294426019043
configs:
- config_name: en-es
  data_files:
  - split: train
    path: en-es/train-*
---



    # opus_books-sample-50
    Sample of 50 rows from the Helsinki-NLP/opus_books dataset.
    