from typing import Any, Callable, Tuple, List
import sys
import gc

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
import fastchat.model
from peft import PeftModel

from .base import BaseModel


class LLMModel(BaseModel):
    def __init__(self, *, config_file: str = None, **kwargs):
        self.config = self.load_config(config_file)

        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()

        self.generation_config = self.load_generation_config()

    def get_conv_template(self):
        conv_template = fastchat.model.get_conversation_template(
            self.config["template_name"]
        )
        return conv_template

    def apply_lora(self):
        self.model = PeftModel.from_pretrained(
            self.model,
            self.config["lora_weights"],
            torch_dtype=torch.float16,
        )

    @torch.no_grad()
    def apply_delta(self):
        # load delta to cpu memory to avoid unecessary cuda memory usage
        delta = AutoModelForCausalLM.from_pretrained(
            self.config["delta_weights"],
            load_in_8bit=self.config["load_8bit"],
            torch_dtype=torch.float16,
            device_map={"": torch.device("cpu")},
            low_cpu_mem_usage=True,
        )

        for name, param in self.model.state_dict().items():
            assert name in delta.state_dict(), f"Weight {name} not in model parameters."
            param.data += delta.state_dict()[name].to(param.data.device)

        # need gc.collect() (issue https://github.com/huggingface/transformers/issues/22801)
        del delta
        gc.collect()
        torch.cuda.empty_cache()

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_name"],
            load_in_8bit=self.config["load_8bit"],
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            token=self.config.get("auth_token", None),
            trust_remote_code=self.config.get("trust_remote_code", False),
        )

        if "lora_weights" in self.config:
            self.apply_lora()

        if "delta_weights" in self.config:
            self.apply_delta()

        if not self.config["load_8bit"]:
            self.model.half()

        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        return self.model

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"],
            use_fast=False,
            token=self.config.get("auth_token", None),
            trust_remote_code=self.config.get("trust_remote_code", False),
        )
        self.tokenizer.padding_side = "left"
        return self.tokenizer

    def load_generation_config(self):
        # do_sample is set to False for gready non-probablistic sampling
        self.generation_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=512,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.generation_config

    def load_stopping_criteria(self, input_ids):
        return None

    @torch.no_grad()
    def generate(self, data):
        input_ids = torch.as_tensor(data["input_ids"]).cuda()

        stopping_criteria = self.load_stopping_criteria(input_ids)

        output_ids = self.model.generate(
            input_ids=input_ids,
            generation_config=self.generation_config,
            stopping_criteria=stopping_criteria,
        )

        output_ids = output_ids[:, input_ids.shape[1] :]

        responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        responses = self.post_process(responses)
        return responses

    def post_process(self, responses: List[str]):
        return [i.strip() for i in responses]

    def process_fn(
        self,
        example: Any,
        prompt_construct_fn: Callable[
            [
                Any,
            ],
            Tuple[str],
        ],
    ) -> Any:
        user_prompt = prompt_construct_fn(example)

        conv_template = self.get_conv_template()
        conv_template.append_message(conv_template.roles[0], user_prompt)
        conv_template.append_message(conv_template.roles[1], None)
        example["message"] = conv_template.get_prompt()

        return example


class LLAMAModel(LLMModel):
    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"],
            use_fast=False,
            token=self.config.get("auth_token", None),
            trust_remote_code=self.config.get("trust_remote_code", False),
        )

        if self.tokenizer.pad_token is None:
            # LLAMA doesnot have pad token (https://github.com/huggingface/transformers/issues/22312)
            self.tokenizer.pad_token = "<unk>"
            self.tokenizer.pad_token_id = (
                0  # unk. we want this to be different from the eos token
            )

        self.tokenizer.padding_side = "left"  # Allow batched inference

        return self.tokenizer


class LLAMA2Chat(LLAMAModel):
    def get_conv_template(self):
        conv_template = super().get_conv_template()
        conv_template.set_system_message(
            "You are LLAMA 2-Chat, a large language model trained by Meta."
        )
        return conv_template


class Baichuan2Chat(LLAMAModel):
    def get_conv_template(self):
        conv_template = super().get_conv_template()
        conv_template.set_system_message(
            "You are Baichuan2-Chat, a large language model trained by Baichuan Intelligence."
        )
        return conv_template


class Phi15Chat(LLAMAModel):
    def get_conv_template(self):
        conv_template = super().get_conv_template()
        conv_template.set_system_message(
            "You are Phi-1.5, a large language model trained by Microsoft."
        )
        return conv_template


class LLAMA2ChatWoModel(LLAMA2Chat):
    def __init__(self, *, config_file: str = None, **kwargs):
        self.config = self.load_config(config_file)

        self.tokenizer = self.load_tokenizer()
        self.generation_config = self.load_generation_config()

        self.conv_template = self.get_conv_template()


class Baichuan2ChatWoModel(Baichuan2Chat):
    def __init__(self, *, config_file: str = None, **kwargs):
        self.config = self.load_config(config_file)

        self.tokenizer = self.load_tokenizer()
        self.generation_config = self.load_generation_config()

        self.conv_template = self.get_conv_template()


class Phi15ChatWoModel(Phi15Chat):
    def __init__(self, *, config_file: str = None, **kwargs):
        self.config = self.load_config(config_file)

        self.tokenizer = self.load_tokenizer()
        self.generation_config = self.load_generation_config()

        self.conv_template = self.get_conv_template()
