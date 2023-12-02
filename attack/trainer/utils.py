import random
import copy
import torch
import torch.distributed as dist
import numpy as np

from .opt_utils import (
    longest_common_prefix,
    modify_past_key_values_len,
)


def find_list_pos(input_ids, start):
    for i in range(len(input_ids)):
        if input_ids[i : i + len(start)] == start:
            return i
    return -1


def get_nonascii_toks(tokenizer, device="cpu"):
    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)

    return torch.tensor(ascii_toks, device=device)


def get_input_slices(
    prompt_template,
    target,
    user_prompt,
    adv_suffix,
    tokenizer,
):
    instruction = f"{user_prompt} {adv_suffix}"
    try:
        prompt = prompt_template.format(instruction=instruction)
    except:
        prompt = prompt_template.replace("{instruction}", instruction)
    prompt_end = f"[/INST]  {target}"

    input_ids = tokenizer.encode(prompt)
    end_input_ids = tokenizer.encode(prompt_end, add_special_tokens=False)
    assistant_input_ids = tokenizer.encode("[/INST] ", add_special_tokens=False)
    adv_suffix_ids = tokenizer.encode(adv_suffix, add_special_tokens=False)
    instruction_ids = tokenizer.encode("\n<</SYS>>\n\n", add_special_tokens=False)[1:]

    start1 = find_list_pos(input_ids, adv_suffix_ids)
    end = find_list_pos(input_ids, end_input_ids)
    start_instruct = find_list_pos(input_ids, instruction_ids)

    instruction_slice = slice(
        start_instruct + len(instruction_ids), start1 + len(adv_suffix_ids)
    )
    control_slice = slice(start1, start1 + len(adv_suffix_ids))
    target_slice = slice(end + len(assistant_input_ids), len(input_ids))
    loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)
    assistant_role_slice = slice(end, end + len(assistant_input_ids))
    assert tokenizer.decode(input_ids[control_slice]) == adv_suffix, tokenizer.decode(
        input_ids[control_slice]
    )
    assert tokenizer.decode(input_ids[target_slice]) == target, tokenizer.decode(
        input_ids[target_slice]
    )

    return (
        input_ids,
        target_slice,
        loss_slice,
        control_slice,
        assistant_role_slice,
        instruction_slice,
    )


def get_input_ids(
    prompt_template,
    user_prompt,
    adv_suffix,
    tokenizer,
    target_slice,
):
    multi_prompt_template, multi_user_prompt = prompt_template, user_prompt
    multi_prompts = []
    for prompt_template, user_prompt in zip(multi_prompt_template, multi_user_prompt):
        instruction = f"{user_prompt} {adv_suffix}"
        try:
            prompt = prompt_template.format(instruction=instruction)
        except:
            prompt = prompt_template.replace("{instruction}", instruction)
        multi_prompts.append(prompt)

    tokenizer.padding_side = "right"
    input_ids = tokenizer(multi_prompts, padding=True, return_tensors="pt").input_ids

    return input_ids


def get_input_ids_multi_suffix(
    prompt_template: list,
    user_prompt: list,
    adv_suffixes: list,
    tokenizer,
    target_slice,
):
    assert len(prompt_template) == 0 and len(
        user_prompt
    ), f"Number of prompt_template is {len(prompt_template)}. Number of user_prompt is {len(user_prompt)}. Should be 0."

    multi_prompts = []
    input_ids = []
    for adv_suffix in adv_suffixes:
        instruction = f"{user_prompt} {adv_suffix}"
        try:
            prompt = prompt_template.format(instruction=instruction)
        except:
            prompt = prompt_template.replace("{instruction}", instruction)
        multi_prompts.append(prompt)

    tokenizer.padding_side = "right"
    input_ids = tokenizer(multi_prompts, padding=True, return_tensors="pt").input_ids

    return input_ids


def get_input_ids_wo_target(
    prompt_template,
    user_prompt,
    adv_suffix,
    target,
    tokenizer,
):
    multi_prompt_template, multi_user_prompt, multi_target = (
        prompt_template,
        user_prompt,
        target,
    )
    multi_prompts = []
    for prompt_template, user_prompt, target in zip(
        multi_prompt_template, multi_user_prompt, multi_target
    ):
        instruction = f"{user_prompt} {adv_suffix}"
        try:
            prompt = prompt_template.format(instruction=instruction)
        except:
            prompt = prompt_template.replace("{instruction}", instruction)

        end = prompt.find(f" {target}")
        multi_prompts.append(prompt[:end])

    tokenizer.padding_side = "left"
    toks = tokenizer(multi_prompts, padding=True, return_tensors="pt")
    input_ids = toks.input_ids
    attn_mask = toks.attention_mask
    return input_ids, attn_mask


@torch.no_grad()
def prepare_past_key_value(model, input_ids, control_slice):
    past_input_ids = torch.tensor(input_ids[: control_slice.start])
    past_input_ids = past_input_ids.to(model.device).unsqueeze(0)
    output = model(past_input_ids, use_cache=True)
    past_key_values = output.past_key_values
    return past_input_ids.squeeze(0), past_key_values


def gather_suffixes(new_adv_suffix, world_size, batch_size):
    if world_size >= 1:
        # different workers may get different results, gather to keep them same
        gathered_new_adv_suffix = [None] * world_size
        dist.all_gather_object(gathered_new_adv_suffix, new_adv_suffix)
        union_adv_suffix = set()
        for sublist in gathered_new_adv_suffix:
            union_adv_suffix = union_adv_suffix.union(set(sublist))
        union_adv_suffix = list(union_adv_suffix)
        union_adv_suffix.sort()
        return union_adv_suffix
    return new_adv_suffix[:batch_size]
