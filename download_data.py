import argparse
from huggingface_hub import snapshot_download
import os
from pathlib import Path

# the dev phase uses original Llama 2 models
huggingface_model_names = {
    "dev": {
        "base": "meta-llama/Llama-2-7b-chat-hf",
        "large": "meta-llama/Llama-2-13b-chat-hf",
    },
    "test": "N/A",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download models for red teaming track."
    )
    parser.add_argument(
        "--subtrack",
        type=str,
        default="base",
        choices=["base", "large"],
        help="The subtrack to download models for",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="dev",
        choices=["dev", "test"],
        help="The phase of the competition",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face user access token (accessible in account settings)",
    )
    parser.add_argument("--output_dir", type=str, help="The path of output directory.")
    args = parser.parse_args()

    return args


def main():
    # ========== load input arguments ========== #
    args = parse_args()
    phase = args.phase
    subtrack = args.subtrack
    token = args.token

    output_dir = Path(args.output_dir)

    if phase == "test":
        print("Test data is not available yet")
        return

    model_name = huggingface_model_names[phase][subtrack]

    # ========== download model ========== #
    if not (output_dir / f"{phase}/{subtrack}/model").exists():
        print(f"Downloading model for {subtrack} subtrack of {phase} phase")
        red_team_model_path = output_dir / f"{phase}/{subtrack}/model"
        snapshot_download(
            repo_id=model_name,
            local_dir=red_team_model_path,
            token=token,
            local_dir_use_symlinks=False,
        )
        print("Done")
    else:
        print(f"Found {args.output_dir}/{phase}/{subtrack}; (skipping)")

    # ========== download embedding layer for diversity loss ========== #
    # run wget https://huggingface.co/justinphan3110/Llama-2-7b-embedding-layer/resolve/main/llama2_7b_embedding_weights.pt
    # in the ./data/{phase} folder
    if not (output_dir / f"{phase}/llama2_7b_embedding_weights.pt").exists():
        print(f"Downloading embedding layer for {phase} phase")
        os.system(
            f"wget https://huggingface.co/justinphan3110/Llama-2-7b-embedding-layer/resolve/main/llama2_7b_embedding_weights.pt -P {output_dir}/{phase}"
        )
        print("Done")
    else:
        print(
            f"Found {args.output_dir}/{phase}/llama2_7b_embedding_weights.pt; (skipping)"
        )


if __name__ == "__main__":
    main()
