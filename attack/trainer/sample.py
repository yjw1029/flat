import random
import copy
import torch
import torch.distributed as dist
import numpy as np


def find_diff_indices(list1, list2):
    diff_indices = []

    for i in range(len(list1)):
        if list1[i] != list2[i]:
            diff_indices.append(i)
    return diff_indices


def sample_control(
    control_toks,
    grad,
    batch_size,
    temp=1,
    not_allowed_tokens=None,
    not_allowed_tokens_pos=None,
    topk=None,
    generator=None,
    specific_position=None,
    **kwargs,
):
    # top_indices = (-grad).topk(topk, dim=1).indices
    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.infty

    # each postion have not allowed tokens for diversity optimization
    if not_allowed_tokens_pos is not None:
        for p in not_allowed_tokens_pos:
            grad[p, not_allowed_tokens_pos[p].to(grad.device)] = np.infty

    new_token_pos = torch.arange(
        0, len(control_toks), len(control_toks) / batch_size, device=grad.device
    ).type(torch.int64)

    control_toks = control_toks.to(grad.device)
    original_control_toks = control_toks.repeat(batch_size, 1)
    grad_dist = torch.nn.functional.softmax(-grad / temp, dim=1).to(grad.device)

    new_token_val = torch.multinomial(
        grad_dist[new_token_pos], 1, replacement=False, generator=generator
    )

    new_token_pos = new_token_pos.unsqueeze(-1)
    new_control_toks = original_control_toks.scatter_(1, new_token_pos, new_token_val)

    return new_control_toks


def sample_control2(losses, new_adv_suffixes, current_adv_suffix, tokenizer):
    current_adv_toks = tokenizer(current_adv_suffix, add_special_tokens=False).input_ids
    new_advs_toks = [
        tokenizer(new_adv_suffix, add_special_tokens=False).input_ids
        for new_adv_suffix in new_adv_suffixes
    ]
    sort_losses, sort_indexes = losses.sort()

    lowest_losses = {p: (None, None, 10000) for p in range(len(current_adv_toks))}
    for loss, i in zip(sort_losses, sort_indexes):
        new_adv_toks = new_advs_toks[i]
        if len(new_adv_toks) != len(current_adv_toks):
            continue

        diff_indices = find_diff_indices(new_adv_toks, current_adv_toks)
        if len(diff_indices) != 1:
            continue
        position = diff_indices[0]

        if lowest_losses[position][0] is None:
            lowest_losses[diff_indices[0]] = (i, new_adv_toks[position], loss)

    lowest_losses = sorted(lowest_losses.items(), key=lambda x: x[1][-1])

    updated_advs_toks = []
    updated_adv_toks = copy.deepcopy(current_adv_toks)
    for position, (_, tok, loss) in lowest_losses:
        if tok is not None:
            updated_adv_toks[position] = tok
            updated_advs_toks.append(copy.deepcopy(updated_adv_toks))

    return updated_advs_toks


def sample_control3(losses, new_adv_suffixes, current_adv_suffix, tokenizer):
    current_adv_toks = tokenizer(current_adv_suffix, add_special_tokens=False).input_ids
    new_advs_toks = [
        tokenizer(new_adv_suffix, add_special_tokens=False).input_ids
        for new_adv_suffix in new_adv_suffixes
    ]
    sort_losses, sort_indexes = losses.sort()

    lowest_losses = {p: (None, None, 10000) for p in range(len(current_adv_toks))}
    for loss, i in zip(sort_losses, sort_indexes):
        new_adv_toks = new_advs_toks[i]
        if len(new_adv_toks) != len(current_adv_toks):
            continue

        diff_indices = find_diff_indices(new_adv_toks, current_adv_toks)
        if len(diff_indices) != 1:
            continue
        position = diff_indices[0]

        if lowest_losses[position][0] is None:
            lowest_losses[diff_indices[0]] = (i, new_adv_toks[position], loss)

    lowest_losses = sorted(lowest_losses.items(), key=lambda x: x[1][-1])

    updated_advs_toks = []
    updated_adv_toks = copy.deepcopy(current_adv_toks)
    for position, (_, tok, loss) in lowest_losses:
        if tok is not None:
            updated_adv_toks[position] = tok
            updated_advs_toks.append(copy.deepcopy(updated_adv_toks))

    return updated_advs_toks
