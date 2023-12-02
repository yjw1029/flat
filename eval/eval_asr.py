import argparse
import json
from pathlib import Path
import transformers
import logging
from eval_utils import compute_success_rate


# Set this to disable warning messages in the generation mode.
transformers.utils.logging.set_verbosity_error()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Running red teaming with baseline methods."
    )
    parser.add_argument(
        "--gpt_config",
        type=str,
        default="./gpt35.yaml",
        help="The config file of GPT models.",
    )
    parser.add_argument(
        "--generations_path",
        type=str,
        default="./generations.json",
        help="The paths of saved generations. If this path exists, then we load it instead of generating from scratch",
    )
    parser.add_argument(
        "--asr_path",
        type=str,
        default="./asr.json",
        help="The path used for saving asrs.",
    )
    parser.add_argument(
        "--compute_asr",
        action="store_true",
        default=False,
        help="Whether to compute asr.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print out intermediate results",
    )
    args = parser.parse_args()

    return args


def merge_dicts(dicts):
    result = {}

    for d in dicts:
        for key, value in d.items():
            if key in result:
                result[key].extend(value)
            else:
                result[key] = value

    return result


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # ========== load input arguments ========== #
    args = parse_args()
    generations_path = args.generations_path
    gpt_config = args.gpt_config
    asr_path = args.asr_path

    if not Path(generations_path).parent.exists():
        raise ValueError(f"Geneation path {generations_path} does not exists.")

    if not Path(asr_path).parent.exists():
        Path(asr_path).parent.mkdir(exist_ok=True, parents=True)

    generations = json.load(open(generations_path, "r"))

    # ========== evaluate completions ========== #
    if args.compute_asr:
        # if not os.path.exists(asr_path):
        overall_success_rate, behavior_success_rates = compute_success_rate(
            generations, gpt_config, asr_path, verbose=True
        )

        logging.info(f"Overall ASR: {overall_success_rate}")


if __name__ == "__main__":
    main()
