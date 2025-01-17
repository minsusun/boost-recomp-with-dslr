# Quick Start

## Environment Configuration

- WSL Ubuntu 22.04
- Python 3.9
- CUDA 12.6

```shell
pip install -r requirements.txt
```

(Dumped list of packages I used to `requirements.txt` but due to other specs it may differ on each system)

## Data Preparation

Prepare data from the original author of RECOMP

```shell
pip install gdown
# Google Drive/recomp_data
gdown https://drive.google.com/drive/folders/1X-BHlZ_HG8tRL-7u70TZGKYZ3W14fnJn --folder -O data/
# Google Drive/recomp_training_data(Not used in here)
gdown https://drive.google.com/drive/folders/1Roahn6qQxB_zZ5j4ZtNm4GQk68m63nqn --folder -O data/
```

## Data Retrieval

Generate retrieved documents with [in-context-ralm](https://github.com/AI21Labs/in-context-ralm)

```shell
cd in-context-ralm
python prepare_retrieval_data.py \
    --retrieval_type sparse \
    --tokenizer_name gpt2 \
    --max_length 1024 \
    --dataset_path wikitext \
    --dataset_name wikitext-103-v1 \
    --dataset_split validation \
    --index_name wikipedia-dpr \
    --forbidden_titles_path ralm/retrievers/wikitext103_forbidden_titles.txt \
    --stride 32 \
    --output_file gpt2_wikitext_validation_retrieval_files_32 \
    --num_tokens_for_query 32 \
    --num_docs 16
mv gpt2_wikitext_validation_retrieval_files_32 ../
```

## Run Extractive Compressor

Run extractive compressor

```shell
python run_extractive_compresor.py \
    --input_data data/extractive_compressor_intputs/flan_ul2_nq_5shot_top_5_passage_new_msmarco_sent.json \
    --model_path  fangyuan/nq_extractive_compressor \
    --output_file outputs/flan_ul2_nq_5shot_top_5_passage_msmarco_sent_with_scores.json \
    --top_k -1 # consider all sentences
```

## Prompt Generation

Generate prompt from result of extractive compressor

```shell
python generate_prompt.py
    --data outputs/flan_ul2_nq_5shot_top_5_passage_msmarco_sent_with_scores.json \
    --prompt_type RECOMP
    --method DSLR
    --output outputs/prompt_nq_DSLR.json
    --top_k -1
```

## Task Run

Run task from generated prompt

(We used `google/flan-t5-small` instead of `google/flan-ul2` due to the limited resource)

```shell
python prompt_flan.py \
    --input_data_csv_file [input_data_file] \
    --output_data_csv_file [name_of_output_data]
```
