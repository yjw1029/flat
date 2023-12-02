import gc

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPTJForCausalLM,
    GPTNeoXForCausalLM,
    LlamaForCausalLM,
)


def get_embedding_matrix(model):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte.weight
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens.weight
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in.weight
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def get_embeddings(model, input_ids):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte(input_ids).half()
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens(input_ids)
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in(input_ids).half()
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def longest_common_prefix(list1, list2):
    i = 0
    while i < len(list1) and i < len(list2) and list1[i] == list2[i]:
        i += 1
    return list1[:i]


def detach_past_key_values(past_key_values):
    detached_past_key_values = [(i[0].detach(), i[1].detach()) for i in past_key_values]
    return detached_past_key_values


def modify_past_key_values_len(past_key_values, length):
    modified_past_key_values = [
        (i[0][:, :, :length, :], i[1][:, :, :length, :]) for i in past_key_values
    ]
    return modified_past_key_values


def expand_batch(tensor: torch.Tensor, batch_size: int):
    batch_tensor = tensor.expand(batch_size, -1, -1, -1)
    return batch_tensor


def expand_past_key_value(past_key_values, batch_size):
    past_key_values = [
        (expand_batch(i[0], batch_size), expand_batch(i[1], batch_size))
        for i in past_key_values
    ]
    return past_key_values


def token_gradients(
    model,
    input_ids,
    input_slice,
    target_slice,
    loss_slice,
    past_input_ids=None,
    past_key_values=None,
):
    """
    Computes gradients of the loss with respect to the coordinates.

    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.
    past_input_ids: torch.Tensor
        The input sequence before control string (adversarail suffix).
    past_key_value: torch.Tensot
        The pre-computed hidden state of past_input_ids

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """

    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype,
    )
    one_hot.scatter_(
        1,
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype),
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)

    # find the common ids to avoid duplicate computation
    if past_key_values is not None:
        common_input_ids = longest_common_prefix(input_ids, past_input_ids)
        past_key_values = modify_past_key_values_len(
            past_key_values, len(common_input_ids)
        )
    else:
        common_input_ids = []

    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:, len(common_input_ids) : input_slice.start, :],
            input_embeds,
            embeds[:, input_slice.stop :, :],
        ],
        dim=1,
    )

    logits = model(inputs_embeds=full_embeds, past_key_values=past_key_values).logits

    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(
        logits[
            0,
            loss_slice.start
            - len(common_input_ids) : loss_slice.stop
            - len(common_input_ids),
            :,
        ],
        targets,
    )

    loss.backward()

    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)
    # past_key_values = detach_past_key_values(past_key_values)
    return grad


def sample_control(
    control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None
):
    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.infty

    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(batch_size, 1)
    new_token_pos = torch.arange(
        0, len(control_toks), len(control_toks) / batch_size, device=grad.device
    ).type(torch.int64)
    new_token_val = torch.gather(
        top_indices[new_token_pos],
        1,
        torch.randint(0, topk, (batch_size, 1), device=grad.device),
    )
    new_control_toks = original_control_toks.scatter_(
        1, new_token_pos.unsqueeze(-1), new_token_val
    )

    return new_control_toks


def get_filtered_cands(
    tokenizer, control_cand, filter_cand=True, curr_control=None, dedup=True
):
    cands, count = [], 0
    for i in range(len(control_cand)):
        decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
        if filter_cand:
            if decoded_str != curr_control and len(
                tokenizer(decoded_str, add_special_tokens=False).input_ids
            ) == len(control_cand[i]):
                cands.append(decoded_str)
            else:
                count += 1
        else:
            cands.append(decoded_str)

    if filter_cand:
        cands = cands + [cands[-1]] * (len(control_cand) - len(cands))

    if dedup:
        cands = sorted(list(set(cands)))

    return cands


def get_losses(
    *,
    model,
    tokenizer,
    input_ids,
    control_slice,
    target_slice,
    device,
    test_controls=None,
    return_ids=False,
    batch_size=512,
    past_input_ids=None,
    past_key_values=None,
):
    # find the common ids to avoid duplicate computation
    if past_key_values is not None:
        common_input_ids = longest_common_prefix(input_ids, past_input_ids)
        past_key_values = modify_past_key_values_len(
            past_key_values, len(common_input_ids)
        )
    else:
        past_key_values = None
        common_input_ids = []

    if isinstance(test_controls[0], str):
        max_len = control_slice.stop - control_slice.start
        test_ids = [
            torch.tensor(
                tokenizer(control, add_special_tokens=False).input_ids[:max_len],
                device=device,
            )
            for control in test_controls
        ]
        pad_tok = 0
        while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1
        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(
            nested_ids, pad_tok, (len(test_ids), max_len)
        )
    else:
        raise ValueError(
            f"test_controls must be a list of strings, got {type(test_controls)}"
        )

    if not (test_ids[0].shape[0] == control_slice.stop - control_slice.start):
        raise ValueError(
            (
                f"test_controls must have shape "
                f"(n, {control_slice.stop - control_slice.start}), "
                f"got {test_ids.shape}"
            )
        )

    locs = (
        torch.arange(control_slice.start, control_slice.stop)
        .repeat(test_ids.shape[0], 1)
        .to(device)
    )
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(device),
        1,
        locs,
        test_ids,
    )

    if pad_tok >= 0:
        attn_mask = (ids != pad_tok).type(ids.dtype)
    else:
        attn_mask = None

    ids = ids[:, len(common_input_ids) :]

    if past_key_values is not None:
        past_key_values = expand_past_key_value(past_key_values, ids.shape[0])

    if return_ids:
        del locs, test_ids
        gc.collect()
        return (
            forward(
                model=model,
                input_ids=ids,
                attention_mask=attn_mask,
                target_slice=target_slice,
                batch_size=batch_size,
                past_key_values=past_key_values,
                common_ids_len=len(common_input_ids),
            ),
            ids,
        )
    else:
        del locs, test_ids
        losses = forward(
            model=model,
            input_ids=ids,
            attention_mask=attn_mask,
            target_slice=target_slice,
            batch_size=batch_size,
            past_key_values=past_key_values,
            common_ids_len=len(common_input_ids),
        )
        del ids
        gc.collect()
        return losses


@torch.no_grad()
def forward(
    *,
    model,
    input_ids,
    attention_mask,
    target_slice,
    batch_size=512,
    past_key_values=None,
    common_ids_len=0,
):
    losses = []
    crit = nn.CrossEntropyLoss(reduction="none")

    loss_slice = slice(
        target_slice.start - 1 - common_ids_len,
        target_slice.stop - 1 - common_ids_len,
    )
    target_slice = slice(
        target_slice.start - common_ids_len, target_slice.stop - common_ids_len
    )

    for i in range(0, input_ids.shape[0], batch_size):
        batch_input_ids = input_ids[i : i + batch_size]
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i : i + batch_size]
        else:
            batch_attention_mask = None

        if past_key_values is not None:
            batch_past_key_values = [
                (j[0][i : i + batch_size], j[1][i : i + batch_size])
                for j in past_key_values
            ]
        else:
            batch_past_key_values = None

        batch_logits = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            past_key_values=batch_past_key_values,
        ).logits

        batch_losses = crit(
            batch_logits[:, loss_slice, :].transpose(1, 2),
            batch_input_ids[:, target_slice],
        ).mean(dim=-1)
        losses.append(batch_losses.detach())

        del batch_input_ids, batch_attention_mask, batch_logits, batch_losses
        gc.collect()
        torch.cuda.empty_cache()

    return torch.cat(losses, dim=0)


def target_loss(logits, ids, target_slice, common_ids_len=0):
    crit = nn.CrossEntropyLoss(reduction="none")
    loss_slice = slice(
        target_slice.start - 1 - common_ids_len, target_slice.stop - 1 - common_ids_len
    )
    target_slice = slice(
        target_slice.start - common_ids_len, target_slice.stop - common_ids_len
    )
    loss = crit(logits[:, loss_slice, :].transpose(1, 2), ids[:, target_slice])
    return loss.mean(dim=-1)


def load_tokenizer(model_path, tokenizer_path=None):
    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, use_fast=False
    )

    if "oasst-sft-6-llama-30b" in tokenizer_path:
        tokenizer.bos_token_id = 1
        tokenizer.unk_token_id = 0
    if "guanaco" in tokenizer_path:
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0
    if "llama-2" in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "left"
    if "falcon" in tokenizer_path:
        tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model_and_tokenizer(model_path, tokenizer_path=None, **kwargs):
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, trust_remote_code=True, **kwargs
        )
    ).eval()

    tokenizer = load_tokenizer(model_path, tokenizer_path)

    return model, tokenizer


def load_model_and_tokenizer_tp(model_path, tokenizer_path=None, **kwargs):
    from tensor_parallel import TensorParallelPreTrainedModel

    model = TensorParallelPreTrainedModel(
        AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, trust_remote_code=True, **kwargs
        ),
        ["cuda:0", "cuda:1"],
    ).eval()

    tokenizer = load_tokenizer(model_path, tokenizer_path)

    return model, tokenizer


def multi_token_gradients(
    model,
    batch_input_ids,
    batch_input_slices,
    batch_target_slices,
    batch_loss_slices,
    past_input_ids=None,
    past_key_values=None,
):
    """
    Computes gradients of the loss with respect to a batch of coordinates.

    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    batch_input_ids : torch.Tensor
        The batch input sequence in the form of token ids.
    batch_input_slices : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.
    batch_size: int
        The batch size to compute gradients.
    past_input_ids: torch.Tensor
        The input sequence before control string (adversarail suffix).
    past_key_value: torch.Tensot
        The pre-computed hidden state of past_input_ids

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """

    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        batch_input_ids[0][batch_input_slices[0]].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype,
    )
    one_hot.scatter_(
        1,
        batch_input_ids[0][batch_input_slices[0]].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype),
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)

    # find the common ids to avoid duplicate computation
    if past_key_values is not None:
        common_input_ids = longest_common_prefix(batch_input_ids[0], past_input_ids)
        past_key_values = modify_past_key_values_len(
            past_key_values, len(common_input_ids)
        )
        past_key_values = expand_past_key_value(
            past_key_values, batch_input_ids.shape[0]
        )
    else:
        common_input_ids = []

    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, batch_input_ids).detach()

    batch_full_embeds = []
    for i, input_slice in enumerate(batch_input_slices):
        full_embeds = torch.cat(
            [
                embeds[i : i + 1, len(common_input_ids) : input_slice.start, :],
                input_embeds,
                embeds[i : i + 1, input_slice.stop :, :],
            ],
            dim=1,
        )
        batch_full_embeds.append(full_embeds)

    batch_full_embeds = torch.cat(batch_full_embeds, dim=0)
    logits = model(
        inputs_embeds=batch_full_embeds, past_key_values=past_key_values
    ).logits

    all_loss = None
    for input_ids, target_slice, loss_slice in zip(
        batch_input_ids, batch_target_slices, batch_loss_slices
    ):
        targets = input_ids[target_slice]
        loss = nn.CrossEntropyLoss()(
            logits[
                0,
                loss_slice.start
                - len(common_input_ids) : loss_slice.stop
                - len(common_input_ids),
                :,
            ],
            targets,
        )
        if all_loss:
            all_loss += loss
        else:
            all_loss = loss

    all_loss.backward()

    grad = one_hot.grad.clone().detach()
    # grad = grad / grad.norm(dim=-1, keepdim=True)
    # past_key_values = detach_past_key_values(past_key_values)
    return grad


def mutli_token_gradients_bs(
    model,
    input_ids,
    input_slices,
    target_slices,
    loss_slices,
    past_input_ids=None,
    past_key_values=None,
    batch_size=None,
):
    all_grad = None
    for i in range(0, input_ids.shape[0], batch_size):
        batch_input_ids = input_ids[i : i + batch_size]
        batch_input_slices = input_slices[i : i + batch_size]
        batch_target_slices = target_slices[i : i + batch_size]
        batch_loss_slices = loss_slices[i : i + batch_size]

        batch_grad = multi_token_gradients(
            model,
            batch_input_ids,
            batch_input_slices,
            batch_target_slices,
            batch_loss_slices,
            past_input_ids,
            past_key_values,
        )

        if all_grad is None:
            all_grad = batch_grad
        else:
            all_grad += batch_grad
    return all_grad


def gather_grad_and_norm(grad, world_size):
    if world_size > 1:
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)

    grad = grad / grad.norm(dim=-1, keepdim=True)
    return grad


@torch.no_grad()
def multi_get_losses(
    *,
    model,
    tokenizer,
    multi_input_ids,
    control_slices,
    target_slices,
    device,
    test_controls=None,
    return_indivisual_loss=False,
    batch_size=512,
    past_input_ids=None,
    past_key_values=None,
):
    # find the common ids to avoid duplicate computation
    if past_key_values is not None:
        common_input_ids = longest_common_prefix(multi_input_ids[0], past_input_ids)
        past_key_values = modify_past_key_values_len(
            past_key_values, len(common_input_ids)
        )
    else:
        past_key_values = None
        common_input_ids = []

    if isinstance(test_controls[0], str):
        max_len = control_slices[0].stop - control_slices[0].start
        test_ids = [
            torch.tensor(
                tokenizer(control, add_special_tokens=False).input_ids[:max_len],
                device=device,
            )
            for control in test_controls
        ]
        pad_tok = 0
        while pad_tok in multi_input_ids or any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1
        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(
            nested_ids, pad_tok, (len(test_ids), max_len)
        )
    else:
        raise ValueError(
            f"test_controls must be a list of strings, got {type(test_controls)}"
        )

    if not (test_ids[0].shape[0] == control_slices[0].stop - control_slices[0].start):
        raise ValueError(
            (
                f"test_controls must have shape "
                f"(n, {control_slices[0].stop - control_slices[0].start}), "
                f"got {test_ids.shape}"
            )
        )

    multi_losses = []
    for i, target_slice, control_slice in zip(
        range(len(target_slices)), target_slices, control_slices
    ):
        locs = (
            torch.arange(control_slice.start, control_slice.stop)
            .repeat(test_ids.shape[0], 1)
            .to(device)
        )

        ids = torch.scatter(
            multi_input_ids[i].unsqueeze(0).repeat(test_ids.shape[0], 1).to(device),
            1,
            locs,
            test_ids,
        )

        if pad_tok >= 0:
            attn_mask = (ids != pad_tok).type(ids.dtype)
        else:
            attn_mask = None

        ids = ids[:, len(common_input_ids) :]

        if past_key_values is not None:
            past_key_values = expand_past_key_value(past_key_values, ids.shape[0])

        del locs
        losses = forward(
            model=model,
            input_ids=ids,
            attention_mask=attn_mask,
            target_slice=target_slice,
            batch_size=batch_size,
            past_key_values=past_key_values,
            common_ids_len=len(common_input_ids),
        ).detach()
        del attn_mask, ids
        gc.collect()
        torch.cuda.empty_cache()
        multi_losses.append(losses)

    multi_losses = torch.stack(multi_losses, dim=0)
    if return_indivisual_loss:
        mean_losses = torch.mean(multi_losses, dim=0)
        del test_ids
        gc.collect()
        torch.cuda.empty_cache()
        return mean_losses, multi_losses
    else:
        mean_losses = torch.mean(multi_losses, dim=0)
        del test_ids, multi_losses
        gc.collect()
        torch.cuda.empty_cache()
        return mean_losses


@torch.no_grad()
def generate(
    model,
    tokenizer,
    input_ids,
    attention_mask=None,
    gen_config=None,
    past_input_ids=None,
    past_key_values=None,
):
    if past_input_ids is not None:
        common_input_ids = longest_common_prefix(input_ids[0], past_input_ids)
        past_key_values = modify_past_key_values_len(
            past_key_values, len(common_input_ids)
        )
        past_key_values = expand_past_key_value(past_key_values, input_ids.shape[0])
    else:
        common_input_ids = []

    input_ids = input_ids[:, len(common_input_ids) :]
    input_ids = input_ids.to(model.device)

    if attention_mask is None:
        attention_mask = torch.ones(
            (input_ids.shape[0], len(common_input_ids) + input_ids.shape[-1])
        ).to(model.device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
    )
    next_tokens = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
    past_key_values = outputs.past_key_values
    attention_mask = torch.cat(
        [attention_mask, torch.ones((input_ids.shape[0], 1), device=model.device)],
        dim=-1,
    )

    all_ret = [next_tokens]

    for _ in range(gen_config.max_new_tokens - 1):
        next_model_inputs = model.prepare_inputs_for_generation(
            input_ids=next_tokens,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=True,
        )

        outputs = model(
            **next_model_inputs,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_tokens = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
            dim=-1,
        )
        all_ret.append(next_tokens)

    next_tokens = torch.stack(all_ret, dim=1)
    return tokenizer.batch_decode(next_tokens.squeeze(2))
