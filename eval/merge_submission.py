import json
from pathlib import Path
from glob import glob

# 1. read all jsonlines the last line

log_path = "/home/t-jingweiyi/Jingwei/TDC/data/test_log_local/1106_7b_diverse_sam_wh_80"
behavior_path = "/home/t-jingweiyi/Jingwei/TDC/data/data/test/behaviors.json"

test_case_path = 
with open(behavior_path, "r") as f:
    behaviors = json.load(f)

test_cases = {}

for file in glob(log_path + "/*.jsonl"):
    file_name = Path(file).name

    index = int(file_name.split(".")[0])
    behavior = behaviors[index]
    with open(file, "r") as f:
        lines = f.readlines()
        line = json.loads(lines[-1].strip())

    suffixes = line["suffixes"]
    test_cases[behavior] = [f"{behavior} {s}" for s in suffixes]


with open("")
