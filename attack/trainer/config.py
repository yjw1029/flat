from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class TrainingArgs:
    seed: int = field(
        default=2023,
        metadata={"help": "The random seed for reproduction."},
    )
    sample_seed: int = field(
        default=2023,
        metadata={"help": "The random seed for reproduction."},
    )
    model_name_or_path: str = field(
        default=None, metadata={"help": "The model being attacked."}
    )

    temperature: float = field(
        default=1.0,
        metadata={"help": "The temperature used for sampling adversarial strings."},
    )
    topk: int = field(
        default=None,
        metadata={"help": "The topk tokens will be selected as sampling candidates."},
    )
    batch_sizes: List[int] = field(
        default_factory=lambda: [512],
        metadata={"help": "The exploration batch size per step."},
    )
    mini_batch_sizes: List[int] = field(
        default_factory=lambda: [512],
        metadata={"help": "The min batch size of sample for computing losses."},
    )
    loss_bounds: List[float] = field(
        default_factory=list,
        metadata={"help": "The loss bounds to change batch size and mini batch size."},
    )
    num_steps: int = field(
        default=500, metadata={"help": "The iteration number of steps"}
    )
    allow_non_ascii: bool = field(
        default=False,
        metadata={
            "help": "Whether to include non-ascii tokens in sampling candaidates."
        },
    )
    enable_past_key_value: bool = field(
        default=False,
        metadata={"help": "Whether to use past key value to decrease memory cost."},
    )

    adv_string_init_file: str = field(
        default=None,
        metadata={"help": "The file with initial adversarial suffix string."},
    )
    len_control: int = field(
        default=20,
        metadata={"help": "The length of adversarial suffix."},
    )

    local_rank: int = field(
        default=int(os.getenv("LOCAL_RANK", "0")),
        metadata={"help": "local rank."},
    )
    world_size: int = field(
        default=int(os.getenv("WORLD_SIZE", "1")),
        metadata={"help": "world size."},
    )

    log_file: str = field(
        default=None,
        metadata={"help": "The file to log."},
    )
    suffix_jsonl_file: str = field(
        default=None,
        metadata={"help": "The jsinline file to save suffixes."},
    )
    data_file: str = field(
        default=None,
        metadata={"help": "The data file to log."},
    )
    sample_indexes: List[int] = field(
        default_factory=lambda: [0],
        metadata={"help": "The sample indexes for optimization."},
    )

    resume: bool = field(
        default=False, metadata={"help": "Whether to resume from existing log file."}
    )
    debug: bool = field(
        default=False, metadata={"help": "Whether to enable debug mode."}
    )
    enable_stacking: bool = field(
        default=True,
        metadata={
            "help": "Whether to enable stacking which updates multi tokens per step."
        },
    )


@dataclass
class MultiTrainingArgs(TrainingArgs):
    progress_update_loss: float = field(
        default=None, metadata={"help": "The loss to progressively add new behavior."}
    )
    grad_batch_size: int = field(
        default=10,
        metadata={"help": "The max batch_size to compute grad (avoid OOM)."},
    )


@dataclass
class DiverseTrainingArgs(TrainingArgs):
    loss_upper_bound: float = field(
        default=0.2, metadata={"help": "The upper loss for sampling replacement."}
    )
    success_prompt_file: str = field(
        default=None,
        metadata={
            "help": "The file with prompts success attack the behavior. Used for intialization."
        },
    )
    grad_batch_size: int = field(
        default=10,
        metadata={"help": "The max batch_size to compute grad (avoid OOM)."},
    )
    embedding_file: str = field(
        default=None,
        metadata={
            "help": "The path of Llama 7b for computing local cosine diversity score."
        },
    )
    init_case_file: str = field(
        default=None,
        metadata={
            "help": "The submission casess of individual optimization, used as diversity starting point."
        },
    )
    num_adv_suffixes: int = field(
        default=50,
        metadata={"help": "The number of adversarial suffixes."},
    )

    max_diverse_theshold: float = field(
        default=0.3,
        metadata={"help": "The maximum loss threshold to exit diversity optimization."},
    )
    min_diverse_theshold: float = field(
        default=0.2,
        metadata={"help": "The minimum loss threshold to exit overall optimization."},
    )

    batch_sizes_ls: List[int] = field(
        default_factory=lambda: [512],
        metadata={"help": "The exploration batch size per step for step loss."},
    )
    mini_batch_sizes_ls: List[int] = field(
        default_factory=lambda: [512],
        metadata={
            "help": "The min batch size of sample for computing losses for step loss."
        },
    )
    loss_bounds_ls: List[float] = field(
        default_factory=list,
        metadata={
            "help": "The loss bounds to change batch size and mini batch size for step loss."
        },
    )
