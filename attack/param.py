import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproduction."
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="The data file with targets and prompt.",
    )
    parser.add_argument("--model_path", type=str, help="The LLM model path.")
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="The index of sample in data.",
    )
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--mini_batch_size", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--allow_non_ascii", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to inference every 10 steps.",
    )
    parser.add_argument("--enable_past_key_value", action="store_true", help="")
    parser.add_argument("--log_file", type=str, default="The file to log.")
    args = parser.parse_args()
    return args


def parse_args_lb():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproduction."
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="The data file with targets and prompt.",
    )
    parser.add_argument("--model_path", type=str, help="The LLM model path.")
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="The index of sample in data.",
    )

    parser.add_argument(
        "--len_control",
        type=int,
        default="20",
        help="The length of adversarial suffix.",
    )

    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[512])
    parser.add_argument("--loss_bounds", type=float, nargs="+", default=[])
    parser.add_argument("--mini_batch_sizes", type=int, nargs="+", default=[512])
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--allow_non_ascii", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to inference every 10 steps.",
    )
    parser.add_argument("--enable_past_key_value", action="store_true", help="")
    parser.add_argument("--log_file", type=str, default="The file to log.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="The temperatture for sample control.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.getenv("LOCAL_RANK", "0")),
        help="local rank",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=int(os.getenv("WORLD_SIZE", "1")),
        help="world_size",
    )
    args = parser.parse_args()
    return args
