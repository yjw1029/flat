import json
import torch
import numpy as np


def sample_control(
    control_toks, grad, batch_size, temp=1, not_allowed_tokens=None, **kwargs
):
    # top_indices = (-grad).topk(topk, dim=1).indices
    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.infty

    control_toks = control_toks.to(grad.device)
    original_control_toks = control_toks.repeat(batch_size, 1)
    grad_dist = torch.nn.functional.softmax(-grad / temp, dim=1).to(grad.device)

    new_token_pos = torch.arange(
        0, len(control_toks), len(control_toks) / batch_size, device=grad.device
    ).type(torch.int64)
    new_token_val = torch.multinomial(grad_dist[new_token_pos], 1, replacement=False)

    new_token_pos = new_token_pos.unsqueeze(-1)
    new_control_toks = original_control_toks.scatter_(1, new_token_pos, new_token_val)

    return new_control_toks


def get_suffix_from_cases(case_file, behavior):
    with open(case_file, "r") as f:
        prompts = json.load(f)[behavior]
    suffixes = [p[len(behavior) + 1 :] for p in prompts]
    return suffixes
