import os
from pathlib import Path
import glob

import argparse
import re
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Running red teaming with baseline methods."
    )
    parser.add_argument(
        "--generation_dir",
        type=str,
        default="",
        help="The directory with generation files.",
    )
    parser.add_argument(
        "--merged_generation_file",
        type=str,
        default="",
        help="The output merged generation filed.",
    )
    args = parser.parse_args()

    return args


def parse_filename(filename):
    match = re.search(r"(\d+).json", filename)
    if match:
        return int(match.group(1))
    else:
        return None


def main(args):
    generation_files = glob.glob(os.path.join(args.generation_dir, "*.json"))

    index_file = []
    for file_name in generation_files:
        index = parse_filename(file_name)
        if index is None:
            continue
        index_file.append((index, os.path.join(args.generation_dir, file_name)))

    index_file.sort(key=lambda x: x[0])

    merge_result = {}
    for index, file in index_file:
        with open(file, "r") as f:
            result = json.load(f)
            merge_result.update(result)

    with open(args.merged_generation_file, "w") as f:
        json.dump(merge_result, f)


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.generation_dir):
        raise ValueError(f"Geneation dir {args.generation_dir} does not exists.")

    if not Path(args.merged_generation_file).parent.exists():
        Path(args.merged_generation_file).parent.mkdir(exist_ok=True, parents=True)

    main(args)
