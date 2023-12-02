import logging
import pandas as pd
from pathlib import Path
import jsonlines
import json

import transformers

from accelerate import Accelerator
from accelerate.utils import set_seed


from scheduler.base import MultiStepScheduler
from trainer.config import MultiTrainingArgs
from trainer.multi import ProgressiveTrainer


accelerator = Accelerator()
parser = transformers.HfArgumentParser((MultiTrainingArgs,))
args = parser.parse_args_into_dataclasses()[0]

Path(args.log_file).parent.mkdir(exist_ok=True, parents=True)
Path(args.suffix_jsonl_file).parent.mkdir(exist_ok=True, parents=True)


logger = logging.getLogger(__name__)

if args.local_rank == 0:
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        f"Rank:{args.local_rank} - %(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(
        logging.INFO
    )  # Set the minimum logging level for this handler
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(args.log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


set_seed(args.seed)

if args.resume and Path(args.suffix_jsonl_file).exists():

    def read_log():
        last_suffix, last_step, min_loss = None, 0, 10000
        with jsonlines.open(args.suffix_jsonl_file, "r") as reader:
            for obj in reader:
                step = obj["step"]
                suffix = obj["suffix"]
                min_loss_ = obj["min_loss"]
                if step > last_step:
                    last_step = step
                    last_suffix = suffix
                    min_loss = min_loss_
        return last_suffix, last_step, min_loss

    adv_string_init, adv_last_step, min_loss = read_log()
    if adv_string_init is not None:
        logger.info(
            f"Resume from {adv_last_step}. Adv string: {adv_string_init}. Min Loss: {min_loss}"
        )
        adv_last_step += 1
    else:
        if args.adv_string_init_file:
            with open(args.adv_string_init_file, "r") as f:
                adv_string_init = f.read()
        else:
            adv_string_init = " ".join(["!"] * args.len_control)
        adv_last_step = 0
        min_loss = 10000
        logger.info(f"New start from {adv_last_step}. Adv string: {adv_string_init}")
else:
    if args.adv_string_init_file:
        if args.adv_string_init_file.endswith(".txt"):
            with open(args.adv_string_init_file, "r") as f:
                adv_string_init = f.read()
        elif args.adv_string_init_file.endswith(".json"):
            with open(args.adv_string_init_file, "r") as f:
                adv_string_init = json.load(f)["suffix"]
    else:
        adv_string_init = " ".join(["!"] * args.len_control)
    adv_last_step = 0
    min_loss = 10000
    logger.info(f"New start from {adv_last_step}. Adv string: {adv_string_init}")


df = pd.read_csv(args.data_file)
sample = df.loc[args.sample_indexes]
user_prompts = sample["goal"].tolist()
targets = sample["target"].tolist()

llama2_prompt = """[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{instruction} [/INST] """

prompt_templates = [(llama2_prompt + " " + target) for target in targets]
# must add an empty token between target and prompt

logger.info(f"user_prompt: {user_prompts}")
logger.info(f"target: {targets}")
logger.info(f"temperature: {args.temperature}")


num_steps = args.num_steps
topk = args.topk
mini_batch_sizes = args.mini_batch_sizes
batch_sizes = args.batch_sizes
loss_bounds = args.loss_bounds
allow_non_ascii = args.allow_non_ascii

logger.info(f"allow_non_ascii: {allow_non_ascii}")
logger.info(f"topk: {topk}")
logger.info(f"num_steps: {num_steps}")

bs_scheduler = MultiStepScheduler(
    args.batch_sizes, args.mini_batch_sizes, args.loss_bounds, logger=logger
)

trainer = ProgressiveTrainer(
    args,
    init_adv_suffix=adv_string_init,
    user_prompts=user_prompts,
    prompt_templates=prompt_templates,
    targets=targets,
    bs_scheduler=bs_scheduler,
    min_loss=min_loss,
    adv_last_step=adv_last_step,
    logger=logger,
)

trainer.run()
