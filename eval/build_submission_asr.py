# %%
from pathlib import Path
import json
import random
import jsonlines
import string
import pandas as pd

from utils import cut_refuse_pattern, cut_sure_pattern

# %%
# base or large
name = "large"

root_dir = Path("../")
suffix_path = root_dir / f"test_cases/{name}.json"
gen_path = root_dir / f"test_merged_gen/{name}.json"

asr_path = root_dir / f"test_asr/{name}.jsonl"
meta_path = root_dir / f"test_filter_meta/{name}.json"
submission_dir = f"../submission_{name}"

submission_path = Path(submission_dir) / "test_cases.json"

harmful_data = Path(root_dir / f"data/harmful_behaviors.csv")

# %%
asrs_rslt = []
with jsonlines.open(asr_path, "r") as reader:
    for obj in reader:
        asrs_rslt.append(obj)

# %%
with open(meta_path, "r") as f:
    meta = json.load(f)

with open(gen_path, "r") as f:
    gens = json.load(f)

with open(suffix_path, "r") as f:
    suffix = json.load(f)


df = pd.read_csv(harmful_data)
# %%

success_rslt_gpt4 = {}

for obj in asrs_rslt:
    behavior = obj["behavior"]
    if behavior not in success_rslt_gpt4:
        success_rslt_gpt4[behavior] = {"prompts": [], "responses": [], "suffixes": []}

    if obj["predictions"]:
        index = obj["index"]
        response = obj["response"]

        if len(cut_sure_pattern(cut_refuse_pattern(response))) < 20:
            # Filter too short responses (may only contains a title)
            continue

        # print(response.strip())
        select_index = meta[behavior]["selected_indices"][index]
        assert response in gens[behavior][select_index]

        most_sim_response_index = meta[behavior]["most_sim_response_index"]
        response_sim = meta[behavior]["response_sim"]
        for response_index, sim_index in enumerate(most_sim_response_index):
            if sim_index == index:
                if response_index == select_index or response_sim[response_index] > 0.9:
                    success_rslt_gpt4[behavior]["prompts"].append(
                        suffix[behavior][response_index]
                    )
                    success_rslt_gpt4[behavior]["responses"].append(
                        gens[behavior][response_index].strip()
                    )

# %%
for behavior in success_rslt_gpt4:
    print(len(success_rslt_gpt4[behavior]["prompts"]), behavior)

# %%

success_rslt = {k: v["prompts"] for k, v in success_rslt_gpt4.items()}


# %%
import json

with open("../../golden_7b.json", "w") as f:
    json.dump(success_rslt, f)

# %%
import torch
from transformers import AutoTokenizer

embedding_file = (
    "/home/t-jingweiyi/Jingwei/TDC/data/data/dev/llama2_7b_embedding_weights.pt"
)
tokenizer_path = "/home/t-jingweiyi/Jingwei/TDC/data/data/dev/base/model"

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path, padding_side="left", use_fast=False
)  # use_fast=False here for Llama
tokenizer.pad_token = tokenizer.eos_token

embedding_weights = torch.load(embedding_file)
embedding_layer = torch.nn.Embedding.from_pretrained(embedding_weights)
embedding_layer.requires_grad_(False)

# %%
from utils import get_embeddings, filter_results


def random_string(length=10):
    return "".join(random.choices(string.ascii_letters, k=length))


selected_samples = {}
empty_behavior_cnt = 0
less_50_num = 0
dup_sample_num = 0
for behavior in suffix:
    if behavior not in success_rslt or len(success_rslt[behavior]) == 0:
        print(behavior)
        empty_behavior_cnt += 1
        # selected_samples[behavior] = [random_string(20) for _ in range(50)]
        sent_embeddings = get_embeddings(suffix[behavior], embedding_layer, tokenizer)
        selected_samples[behavior] = filter_results(suffix[behavior], sent_embeddings)
    elif len(success_rslt[behavior]) > 50:
        sent_embeddings = get_embeddings(
            success_rslt[behavior], embedding_layer, tokenizer
        )
        selected_samples[behavior] = filter_results(
            success_rslt[behavior], sent_embeddings
        )
        # selected_samples[behavior] = random.sample(success_rslt[behavior], 50)
    else:
        # dup 50 if
        less_50_num += 1
        dup_sample_num += 50 - len(success_rslt[behavior])
        selected_samples[behavior] = success_rslt[behavior] + [
            success_rslt[behavior][0]
        ] * (50 - len(success_rslt[behavior]))
        selected_samples[behavior] = selected_samples[behavior][:50]

# %%
for behavior in selected_samples:
    assert (
        len(selected_samples[behavior]) == 50
    ), f"{behavior} have {len(selected_samples[behavior])}"

assert len(selected_samples) == 50, f"{len(selected_samples)} behaviors"

import os

Path(submission_path).parent.mkdir(exist_ok=True, parents=True)

# Save the predictions dictionary as a json file
with open(submission_path, "w") as f:
    json.dump(selected_samples, f)
