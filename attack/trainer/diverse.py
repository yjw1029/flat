import torch
import gc
import math
import json
import numpy as np
from collections import Counter

from .utils import (
    get_input_slices,
    get_input_ids,
    get_input_ids_wo_target,
    gather_suffixes,
)
from .opt_utils import (
    get_filtered_cands,
    multi_token_gradients,
    gather_grad_and_norm,
    multi_get_losses,
    generate,
)
from .sample import sample_control2, sample_control, sample_control3
from .base import Trainer
from .diverse_utils import (
    diversify_optimized_suffixes,
    compute_diversity,
    get_invalid_suffix_indexes,
    check_compeletion,
    get_not_allowed_tokens_pos,
)


class DiverseTrainer(Trainer):
    def __init__(
        self,
        args,
        init_adv_suffix,
        user_prompts,
        prompt_templates,
        targets,
        bs_scheduler,
        bs_scheduler_loss,
        min_loss=10000,
        adv_last_step=0,
        logger=None,
        adv_suffix_sets=[],
        resume=False,
    ):
        super().__init__(
            args,
            init_adv_suffix,
            user_prompts,
            prompt_templates,
            targets,
            bs_scheduler,
            min_loss,
            adv_last_step,
            logger,
        )

        self.enable_match_judge = True

        if len(adv_suffix_sets) > self.args.num_adv_suffixes:
            raise ValueError(
                f"adv_suffix_sets should be less than {self.args.num_adv_suffixes}, but get {len(adv_suffix_sets)}"
            )
        else:
            # Pad adv_suffix_sets to num_adv_suffixes
            adv_suffix_sets = adv_suffix_sets + [init_adv_suffix] * (
                self.args.num_adv_suffixes - len(adv_suffix_sets)
            )

        # Filter invalid suffixes
        self.init_adv_suffix_sets = adv_suffix_sets
        self.init_adv_suffix_losses = self._init_adv_suffix_losses(adv_suffix_sets)
        self.position_rng = np.random.default_rng(self.args.sample_seed)

        # Init embedding layer for computing cosine diversity
        self.embedding_layer = self._init_embedding_layer()

        self._init_scheduler_ls(bs_scheduler_loss)

    def _init_scheduler_ls(self, bs_scheduler_loss):
        self.bs_scheduler_loss = bs_scheduler_loss

    def _init_embedding_layer(self):
        embedding_weights = torch.load(self.args.embedding_file)
        embedding_layer = torch.nn.Embedding.from_pretrained(embedding_weights).to(
            self.model.device
        )
        return embedding_layer

    def _init_adv_suffix_losses(self, adv_suffix_sets):
        input_ids = self.get_input_ids(
            adv_suffix=adv_suffix_sets[0], target_slices=self.target_slices
        )
        adv_suffix_losses = multi_get_losses(
            model=self.module,
            device=self.module.module.device,
            tokenizer=self.tokenizer,
            multi_input_ids=input_ids,
            control_slices=self.control_slices,
            target_slices=self.target_slices,
            test_controls=adv_suffix_sets,
            return_indivisual_loss=False,
            batch_size=512,
            past_input_ids=self.past_input_ids,
            past_key_values=self.past_key_values,
        )

        return adv_suffix_losses

    def select_position(self, adv_suffix_sets, adv_suffix_losses):
        adv_suffix_toks = [
            self.tokenizer(adv_suffix, add_special_tokens=False).input_ids
            for adv_suffix in adv_suffix_sets
        ]
        adv_suffix_lens = list(map(len, adv_suffix_toks))

        position_prio = {}
        pos_toks_cnts = {}
        for p in range(max(adv_suffix_lens)):
            # select suffixes with length > p
            valid_index = list(
                filter(
                    lambda x: adv_suffix_lens[x] > p, range(self.args.num_adv_suffixes)
                )
            )
            none_cnt = self.args.num_adv_suffixes - len(valid_index)
            pos_toks_cnt = Counter([adv_suffix_toks[i][p] for i in valid_index])
            position_prio[p] = len(pos_toks_cnt) + none_cnt
            pos_toks_cnts[p] = pos_toks_cnt

        # Locations with fewer unique tokens will be sampled with greater priority
        position_prio = sorted(position_prio.items(), key=lambda x: x[1])
        # Locations where all suffix tokens are different will no longer be sampled.
        position_prio = [
            (i[0], self.args.num_adv_suffixes - i[1])
            for i in position_prio
            if i[1] != self.args.num_adv_suffixes
        ]
        positions, prios = list(zip(*position_prio))

        weights = np.array(prios) / np.sum(prios)
        optimize_position = self.position_rng.choice(positions, p=weights)

        optimize_suffixes, wo_optimize_suffixes = [], []
        optimize_losses, wo_optimize_losses = [], []
        pos_cnt = pos_toks_cnts[optimize_position]

        exists_toks = set()
        for toks, suffix, loss in zip(
            adv_suffix_toks, adv_suffix_sets, adv_suffix_losses
        ):
            if len(toks) <= optimize_position:
                wo_optimize_suffixes.append(suffix)
                continue
            tok = toks[optimize_position]
            if pos_cnt[tok] > 1 and tok in exists_toks:
                optimize_suffixes.append(suffix)
                optimize_losses.append(loss)
            else:
                exists_toks.add(tok)
                wo_optimize_suffixes.append(suffix)
                wo_optimize_losses.append(loss)

        return (
            optimize_position,
            optimize_suffixes,
            wo_optimize_suffixes,
            optimize_losses,
            wo_optimize_losses,
        )

    def get_input_ids(self, adv_suffix, target_slices=None, return_slice=False):
        if return_slice:
            (
                input_ids,
                target_slice,
                loss_slice,
                control_slice,
                assistant_role_slice,
                instruction_slice,
            ) = get_input_slices(
                prompt_template=self.prompt_templates[0],
                target=self.targets[0],
                user_prompt=self.user_prompts[0],
                adv_suffix=adv_suffix,
                tokenizer=self.tokenizer,
            )
            return (
                [input_ids],
                [target_slice],
                [loss_slice],
                [control_slice],
                [assistant_role_slice],
                [instruction_slice],
            )
        else:
            multi_input_ids = get_input_ids(
                prompt_template=self.prompt_templates,
                user_prompt=self.user_prompts,
                adv_suffix=adv_suffix,
                tokenizer=self.tokenizer,
                target_slice=self.target_slices,
            )
            return multi_input_ids

    def step_gradient_sample_control(self):
        pass

    def log_suffix_to_jsonl(
        self, step, diversity, loss, min_loss, suffixes, losses, flag="DIVERSE"
    ):
        if self.args.suffix_jsonl_file and self.args.local_rank == 0:
            with open(self.args.suffix_jsonl_file, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "step": step,
                            "diversity": diversity.detach().cpu().numpy().tolist()
                            if torch.is_tensor(diversity)
                            else diversity,
                            "loss": loss.detach().cpu().numpy().tolist()
                            if torch.is_tensor(loss)
                            else loss,
                            "min_loss": min_loss.detach().cpu().numpy().tolist()
                            if torch.is_tensor(min_loss)
                            else min_loss,
                            "suffixes": suffixes,
                            "losses": losses.detach().cpu().numpy().tolist()
                            if torch.is_tensor(losses)
                            else losses,
                            "flag": flag,
                        }
                    )
                    + "\n"
                )

    def generate(self, adv_suffix_sets, step=None, iter_num=None, max_new_tokens=32):
        completions = []
        for i in range(len(adv_suffix_sets)):
            input_ids, attention_mask = get_input_ids_wo_target(
                prompt_template=[self.prompt_templates[0]],
                user_prompt=[self.user_prompts[0]],
                adv_suffix=adv_suffix_sets[i],
                target=[self.targets[0]],
                tokenizer=self.tokenizer,
            )
            input_ids = input_ids.to(self.model.device)
            attention_mask = attention_mask.to(self.model.device)
            gen_config = self.model.generation_config
            gen_config.max_new_tokens = max_new_tokens

            completion = generate(
                self.model,
                self.tokenizer,
                input_ids,
                attention_mask,
                gen_config=gen_config,
                past_input_ids=self.past_input_ids,
                past_key_values=self.past_key_values,
            )[0].strip()

            if step is not None:
                self.logger.info(
                    f"Step: {step} | Suffix: {i} | Completion: {completion}"
                )
            elif iter_num is not None:
                self.logger.info(
                    f"Iter: {iter_num} | Suffix: {i} | Completion: {completion}"
                )
            else:
                self.logger.info(f"Final | Suffix: {i} | Completion: {completion}")

            completions.append(completion)
        return completions

    def step(self, step: int, adv_suffix_sets: list, adv_suffix_losses: list):
        batch_size, mini_batch_size = self.bs_scheduler.step(self.min_loss)

        (
            optimize_position,
            optimize_suffixes,
            wo_optimize_suffixes,
            optimize_losses,
            wo_optimize_losses,
        ) = self.select_position(adv_suffix_sets, adv_suffix_losses)

        multi_input_ids = []
        multi_new_adv_suffix = []
        multi_suffix_slices = []
        for i, adv_suffix in enumerate(optimize_suffixes):
            batch_size, mini_batch_size = self.bs_scheduler.step(self.min_loss)

            input_ids = self.get_input_ids(
                adv_suffix=adv_suffix, target_slices=self.target_slices
            )

            input_ids = input_ids.to(self.model.device)

            # Step 2. Compute Coordinate Gradient
            # batch size = 1: not need to compute grad among all workers
            coordinate_grad = multi_token_gradients(
                self.model,
                input_ids,
                self.control_slices,
                self.target_slices,
                self.loss_slices,
                past_input_ids=self.past_input_ids,
                past_key_values=self.past_key_values,
            )
            coordinate_grad = gather_grad_and_norm(
                coordinate_grad, self.args.world_size
            )

            # Step 3. Sample a batch of new tokens based on the coordinate gradient.
            # Notice that we only need the one that minimizes the loss.
            with torch.no_grad():
                # Step 3.1 Slice the input to locate the adversarial suffix.

                adv_suffix_tokens = input_ids[0][self.control_slices[0]].to(
                    self.model.device
                )

                # Step 3.2 Randomly sample a batch of replacements.
                new_adv_suffix_toks = sample_control(
                    adv_suffix_tokens,
                    coordinate_grad,
                    batch_size,
                    topk=self.args.topk,
                    temp=self.args.temperature,
                    generator=self.sample_gen,
                    not_allowed_tokens=self.not_allowed_tokens,
                    specific_position=optimize_position,
                )

                # Step 3.3 This step ensures all adversarial candidates have the same number of tokens.
                # This step is necessary because tokenizers are not invertible
                # so Encode(Decode(tokens)) may produce a different tokenization.
                # We ensure the number of token remains to prevent the memory keeps growing and run into OOM.
                new_adv_suffix = get_filtered_cands(
                    self.tokenizer,
                    new_adv_suffix_toks,
                    filter_cand=True,
                    curr_control=adv_suffix,
                )

                del coordinate_grad, adv_suffix_tokens, new_adv_suffix_toks
                gc.collect()
                torch.cuda.empty_cache()

                # Gather to keep same results among different workers
                new_adv_suffix = gather_suffixes(
                    new_adv_suffix, self.args.world_size, batch_size
                )

                multi_input_ids.append(input_ids)
                multi_suffix_slices.append(
                    slice(
                        len(multi_new_adv_suffix),
                        len(multi_new_adv_suffix) + len(new_adv_suffix),
                    )
                )
                multi_new_adv_suffix.extend(new_adv_suffix)

        with torch.no_grad():
            # Step 3.4 Compute loss on these candidates
            losses = multi_get_losses(
                model=self.module,
                device=self.module.module.device,
                tokenizer=self.tokenizer,
                multi_input_ids=multi_input_ids[0],
                control_slices=self.control_slices,
                target_slices=self.target_slices,
                test_controls=multi_new_adv_suffix,
                return_indivisual_loss=False,
                batch_size=mini_batch_size,
                past_input_ids=self.past_input_ids,
                past_key_values=self.past_key_values,
            )

            # Step 3.5 Select diverse tokens on the optimize position
            optimized_suffixes, optimized_losses = diversify_optimized_suffixes(
                optimize_position,
                optimize_suffixes,
                optimize_losses,
                wo_optimize_suffixes,
                multi_new_adv_suffix,
                losses,
                multi_suffix_slices,
                self.tokenizer,
                self.args.loss_upper_bound,
            )
            adv_suffix_sets = wo_optimize_suffixes + optimized_suffixes

            adv_suffix_losses = torch.stack(wo_optimize_losses + optimized_losses)

            current_loss = torch.mean(adv_suffix_losses)

            # Update the running adv_suffix with the best candidate
            if current_loss < self.min_loss:
                self.min_loss = current_loss

            diversity, cos_diversity, bleu_diversity = compute_diversity(
                self.user_prompts[0],
                adv_suffix_sets,
                self.tokenizer,
                self.embedding_layer,
                self.model.device,
            )

        self.logger.info(
            f"Step: {step} | Loss: {current_loss:.4f} | Diversity: {diversity:.4f} | Cosine Diverse: {cos_diversity:.4f} | BLEU Diverse: {bleu_diversity:.4f} | Min Loss: {self.min_loss:.4f} | Positon: {optimize_position} | Num: {len(optimize_suffixes)}"
        )
        self.log_suffix_to_jsonl(
            step=step,
            diversity=diversity,
            loss=current_loss,
            min_loss=self.min_loss,
            suffixes=adv_suffix_sets,
            losses=adv_suffix_losses,
        )

        if step % 10 == 0 and self.args.debug:
            self.generate(adv_suffix_sets=adv_suffix_sets, step=step)

        return (
            adv_suffix_sets,
            adv_suffix_losses,
            diversity > self.args.max_diverse_theshold,
        )

    def step_loss(
        self, step: int, adv_suffix=str, other_adv_suffix_sets=None, loss=None
    ):
        not_allowed_tokens_pos = get_not_allowed_tokens_pos(
            other_adv_suffix_sets, tokenizer=self.tokenizer
        )
        batch_size, mini_batch_size = self.bs_scheduler_loss.step(loss)

        input_ids = self.get_input_ids(
            adv_suffix=adv_suffix, target_slices=self.target_slices
        )

        input_ids = input_ids.to(self.model.device)

        # Step 2. Compute Coordinate Gradient
        # batch size = 1: not need to compute grad among all workers
        coordinate_grad = multi_token_gradients(
            self.model,
            input_ids,
            self.control_slices,
            self.target_slices,
            self.loss_slices,
            past_input_ids=self.past_input_ids,
            past_key_values=self.past_key_values,
        )
        coordinate_grad = gather_grad_and_norm(coordinate_grad, self.args.world_size)

        # Step 3. Sample a batch of new tokens based on the coordinate gradient.
        # Notice that we only need the one that minimizes the loss.
        with torch.no_grad():
            # Step 3.1 Slice the input to locate the adversarial suffix.

            adv_suffix_tokens = input_ids[0][self.control_slices[0]].to(
                self.model.device
            )

            # Step 3.2 Randomly sample a batch of replacements.
            new_adv_suffix_toks = sample_control(
                adv_suffix_tokens,
                coordinate_grad,
                batch_size,
                topk=self.args.topk,
                temp=self.args.temperature,
                generator=self.sample_gen,
                not_allowed_tokens=self.not_allowed_tokens,
                not_allowed_tokens_pos=not_allowed_tokens_pos,
            )

            # Step 3.3 This step ensures all adversarial candidates have the same number of tokens.
            # This step is necessary because tokenizers are not invertible
            # so Encode(Decode(tokens)) may produce a different tokenization.
            # We ensure the number of token remains to prevent the memory keeps growing and run into OOM.
            new_adv_suffix = get_filtered_cands(
                self.tokenizer,
                new_adv_suffix_toks,
                filter_cand=True,
                curr_control=adv_suffix,
            )

            # Gather to keep same results among different workers
            new_adv_suffix = gather_suffixes(
                new_adv_suffix, self.args.world_size, batch_size
            )

            del coordinate_grad, adv_suffix_tokens, new_adv_suffix_toks
            gc.collect()
            torch.cuda.empty_cache()

            # Step 3.4 Compute loss on these candidates and take the argmin.
            losses = multi_get_losses(
                model=self.module,
                device=self.module.module.device,
                tokenizer=self.tokenizer,
                multi_input_ids=input_ids,
                control_slices=self.control_slices,
                target_slices=self.target_slices,
                test_controls=new_adv_suffix,
                return_indivisual_loss=False,
                batch_size=mini_batch_size,
                past_input_ids=self.past_input_ids,
                past_key_values=self.past_key_values,
            )  # decrease this number if you run into OOM.

            best_new_adv_suffix_id = losses.argmin()
            best_new_adv_suffix = new_adv_suffix[best_new_adv_suffix_id]

            current_loss = losses[best_new_adv_suffix_id]

            # Find best suffix for all positions, add then iteratively, select the final one with lowest loss
            if current_loss < self.last_loss and self.args.enable_stacking:
                updated_adv_suffix_toks = sample_control2(
                    losses, new_adv_suffix, adv_suffix, self.tokenizer
                )
                updated_adv_suffix_toks = torch.LongTensor(updated_adv_suffix_toks).to(
                    self.model.device
                )

                updated_adv_suffix = get_filtered_cands(
                    self.tokenizer,
                    updated_adv_suffix_toks,
                    filter_cand=True,
                    curr_control=adv_suffix,
                )

                updated_adv_suffix = gather_suffixes(
                    updated_adv_suffix, self.args.world_size, batch_size
                )

                updated_losses = multi_get_losses(
                    model=self.module,
                    device=self.module.module.device,
                    tokenizer=self.tokenizer,
                    multi_input_ids=input_ids,
                    control_slices=self.control_slices,
                    target_slices=self.target_slices,
                    test_controls=updated_adv_suffix,
                    return_indivisual_loss=False,
                    batch_size=mini_batch_size,
                    past_input_ids=self.past_input_ids,
                    past_key_values=self.past_key_values,
                )

                best_updated_adv_suffix_id = updated_losses.argmin()
                best_updated_adv_suffix = updated_adv_suffix[best_updated_adv_suffix_id]
                min_updated_losses = updated_losses[best_updated_adv_suffix_id]

                if min_updated_losses < current_loss:
                    current_loss = min_updated_losses
                    best_new_adv_suffix = best_updated_adv_suffix

                del updated_adv_suffix_toks, updated_losses

            self.last_loss = current_loss.detach().cpu()
            adv_suffix = best_new_adv_suffix

            del losses
            gc.collect()
            torch.cuda.empty_cache()

        self.logger.info(
            f"Step: {step} | Current Loss: {current_loss:.4f} | Min Loss: {self.min_loss:.4f}"
        )

        completion = self.generate(adv_suffix_sets=[adv_suffix], step=step)

        return adv_suffix, current_loss, completion

    def run(self):
        adv_suffix_sets, adv_suffix_losses = (
            self.init_adv_suffix_sets,
            self.init_adv_suffix_losses,
        )
        iter_num = 0
        global_step = self.adv_last_step

        diversity, cos_diversity, bleu_diversity = compute_diversity(
            self.user_prompts[0],
            adv_suffix_sets,
            self.tokenizer,
            self.embedding_layer,
            self.model.device,
        )

        current_loss = torch.mean(adv_suffix_losses)

        self.logger.info(
            f"Initial | Loss: {current_loss:.4f} | Diversity: {diversity:.4f} | Cosine Diverse: {cos_diversity:.4f} | BLEU Diverse: {bleu_diversity:.4f} | Min Loss: {self.min_loss:.4f}"
        )

        while True:
            iter_num += 1
            # if diversity < self.args.max_diverse_theshold:
            #     # TODO: (delete) Maximum iter 2 steps to get higher diversity
            #     for _ in range(0):
            #         adv_suffix_sets, adv_suffix_losses, is_stop = self.step(
            #             global_step, adv_suffix_sets, adv_suffix_losses
            #         )
            #         global_step += 1

            #         if is_stop:
            #             break

            completions = self.generate(
                adv_suffix_sets=adv_suffix_sets, iter_num=iter_num
            )

            invalid_suffix_indexes, invalid_suffixes = get_invalid_suffix_indexes(
                completions,
                adv_suffix_sets,
                self.targets[0],
                enable_match=self.enable_match_judge,
            )

            for adv_suffix_index, adv_suffix in zip(
                invalid_suffix_indexes, invalid_suffixes
            ):
                self.logger.info(f"Optimize completion for suffix {adv_suffix_index}")
                other_adv_suffix_sets = (
                    adv_suffix_sets[:adv_suffix_index]
                    + adv_suffix_sets[adv_suffix_index + 1 :]
                )

                # Maximum iter 10 step to get lower loss
                for _ in range(10):
                    current_loss = None
                    adv_suffix, current_loss, completion = self.step_loss(
                        step=global_step,
                        adv_suffix=adv_suffix,
                        loss=current_loss,
                        other_adv_suffix_sets=other_adv_suffix_sets,
                    )
                    adv_suffix_sets[adv_suffix_index] = adv_suffix
                    adv_suffix_losses[adv_suffix_index] = current_loss

                    diversity, cos_diversity, bleu_diversity = compute_diversity(
                        self.user_prompts[0],
                        adv_suffix_sets,
                        self.tokenizer,
                        self.embedding_layer,
                        self.model.device,
                    )
                    self.log_suffix_to_jsonl(
                        step=global_step,
                        diversity=diversity,
                        loss=current_loss,
                        min_loss=self.min_loss,
                        suffixes=adv_suffix_sets,
                        losses=adv_suffix_losses,
                        flag="LOSS",
                    )
                    global_step += 1
                    success = check_compeletion(
                        completion=completion,
                        target=self.targets[0],
                        enable_match=self.enable_match_judge,
                    )
                    if success:
                        break

            diversity, cos_diversity, bleu_diversity = compute_diversity(
                self.user_prompts[0],
                adv_suffix_sets,
                self.tokenizer,
                self.embedding_layer,
                self.model.device,
            )
            self.logger.info(
                f"Iter {iter_num} | Loss: {current_loss:.4f} | Diversity: {diversity:.4f} | Cosine Diverse: {cos_diversity:.4f} | BLEU Diverse: {bleu_diversity:.4f} | Min Loss: {self.min_loss:.4f}"
            )

            # if diversity > self.args.min_diverse_theshold:
            break

        # self.generate(adv_suffix_sets=adv_suffix_sets, max_new_tokens=256)

    """
    def initial_step(self, step: int, adv_suffix=str):
        # TODO: This function might not needed if diversity is continues optimized after ASR, since there are multiple successful prompts.
        # TODO: The logic of initial step is to conduct on step in base trainer, but in sample_control2 explore diversity.
        pass
    """
