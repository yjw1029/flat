# Flat: Fast Llm-ATtack

Our approach, based on [llm-attacks](https://github.com/llm-attacks/llm-attacks), has been modified in the following ways:

1. Increased efficiency
* Support for multi-GPU
* Caching of hidden states for repeated computations
2. Algorithm modifications
* Single optimization now supports greedy modification of multiple tokens per step, effectively finding replacement tokens for each position by combining sample control search
* Implementation of the diversity optimization algorithm aimed at ensuring sufficiently low loss while keeping different local suffix tokens as distinct as possible (not finalized for submission).

## Enviroment

```bash
docker pull yjw1029/torch:2.0-llm-v7
```

## Optimization
For each attack (index=0...49), execute the following command
```bash
# Large model 
deepspeed --num_gpus 2 attack-sing.py --seed 2023 --sample_seed 2023 \
--data_file ../data/harmful_behaviors.csv \
--sample_indexes {index} --mini_batch_sizes 1024 1024 512 \
--batch_sizes 4096 2048 512 --loss_bounds 0.5 0.3 \
--log_file ../test_log/large/log_{index}.txt \
--suffix_jsonl_file ../test_log/large/log_{index}.jsonl \
--topk 256 --num_steps 500 --resume True \
--model_name_or_path PATH/TO/MODEL \
--enable_past_key_value --debug True \
--allow_non_ascii True --adv_string_init_file ../adv/uni_13b.json

# Base model
deepspeed --num_gpus 2 attack-sing.py --seed 2023 --sample_seed 2023 \
--data_file ../data/harmful_behaviors.csv \
--sample_indexes {index} --mini_batch_sizes 1024 512 \
--batch_sizes 4096 512 --loss_bounds 0.5 \
--log_file ../test_log/base/log_{index}.txt \
--suffix_jsonl_file ../test_log/base/log_{index}.jsonl \
--topk 256 --num_steps 500 --resume True \
--model_name_or_path PATH/TO/MODEL \
--enable_past_key_value --debug True \
--allow_non_ascii True --adv_string_init_file ../adv/uni_7b.txt
```

## Filter ineffective suffixes

Execute the following command (version=base/large)
```bash
cd eval

python build_case.py --data_path ../data/harmful_behaviors.csv \
--log_path ../test_log/{version} \
--output_path ../test_cases/{version}.json

# Generate response for each attack (index=0...49)
python llama_gen.py \
--model_dir PATH/TO/MODEL \
--test_case_file ../test_cases/{version}.json \
--generation_file ../test_generate/{version}/{index}.json \
--tensor_parallel_size 1 --start_index {index} --num_behaviors 1

# Merge generation
cd eval
python merge_gen.py --generation_dir ../test_generate/{version} \
--merged_generation_file ../test_merged_gen/{version}.json

# Select representitive generation for GPT-4 Evaluation
python select_response.py \
--response_path ../test_merged_gen/{version}.json \
--output_response_path ../test_filtered_gen/{version}.json \
--output_meta_path ../test_filter_meta/{version}.json

# Generate GPT-4 Evaluation
# We provide the evaluation file of large track at test_asr/large.jsonl
python eval_asr.py --gpt_config ../config/gpt4.yaml \
--generations_path ../test_filtered_gen/{version}.json \
--asr_path ../test_asr/{version}.jsonl \
--verbose --compute_asr


# build submission file
python build_submission_asr.py
submission_dir=../submission_large
cd ${submission_dir} && zip ../submission.zip ./*
```