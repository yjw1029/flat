from pathlib import Path
import pandas as pd
import re
import jsonlines
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path", type=str, help="The data path of TDC harmful behavior"
    )
    parser.add_argument("--log_path", type=str, help="The log path.")
    parser.add_argument("--output_path", type=str, help="The output path.")

    args = parser.parse_args()
    return args


args = parse_args()

data_path = args.data_path
log_path = Path(args.log_path)
df = pd.read_csv(args.data_path)


def read_log(index=0):
    loss_bound = 0.5
    suffixes = []
    losses = []
    with jsonlines.open(log_path / f"log_{index}.jsonl", "r") as reader:
        for obj in reader:
            step = obj["step"]
            suffix = obj["suffix"]
            loss = obj["loss"]
            if loss < loss_bound:
                suffixes.append(suffix)
                losses.append(loss)
    return losses, suffixes


def read_log2(index=0):
    loss_bound = 0.5
    step_pattern = r"Step: (\d+)"
    loss_pattern = r"Current Loss: ([\d.]+)"
    suffix_pattern = r"Current Suffix: (.*) \|"
    suffixes = []
    losses = []
    with open(log_path / f"log_{index}.txt", "r") as f:
        for line in f:
            step_match = re.search(step_pattern, line)
            loss_match = re.search(loss_pattern, line)
            suffix_match = re.search(suffix_pattern, line)
            if step_match and loss_match and suffix_match:
                step = int(step_match.group(1))
                loss = float(loss_match.group(1))
                suffix = suffix_match.group(1).strip()
                if loss < loss_bound:
                    suffixes.append(suffix)
                    losses.append(loss)
    return losses, suffixes


cases = {}
for i in range(df.shape[0]):
    behavior = df.loc[i, "goal"]
    try:
        losses, suffixes = read_log(i)
    except:
        try:
            losses, suffixes = read_log2(i)
        except:
            losses, suffixes = [], []

    prompts = []
    for suffix, loss in zip(suffixes, losses):
        prompt = f"{behavior} {suffix}"
        if prompt not in prompts:
            prompts.append(prompt)
    cases[behavior] = prompts


Path(args.output_path).parent.mkdir(exist_ok=True, parents=True)
with open(args.output_path, "w") as f:
    json.dump(cases, f)
