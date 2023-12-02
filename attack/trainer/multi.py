import torch
import gc
import math

from .base import Trainer
from .utils import (
    get_input_ids,
    prepare_past_key_value,
    gather_suffixes,
)
from .opt_utils import (
    get_filtered_cands,
    mutli_token_gradients_bs,
    gather_grad_and_norm,
    multi_get_losses,
)
from .sample import sample_control2, sample_control


class MultiBehaviorTrainer(Trainer):
    @torch.no_grad()
    def _init_past_key_value(self, init_adv_suffix):
        # keep the past key value of system prompts
        input_ids = get_input_ids(
            prompt_template=[self.prompt_templates[0]],
            user_prompt=[self.user_prompts[0]],
            adv_suffix=init_adv_suffix,
            tokenizer=self.tokenizer,
            target_slice=[self.target_slices[0]],
        ).squeeze(0)

        self.past_input_ids, self.past_key_values = prepare_past_key_value(
            self.model, input_ids, self.instruction_slices[0]
        )

    def step(self, step: int, adv_suffix=str):
        batch_size, mini_batch_size = self.bs_scheduler.step(self.min_loss)

        input_ids = self.get_input_ids(
            adv_suffix=adv_suffix, target_slices=self.target_slices
        )

        input_ids = input_ids.to(self.model.device)

        # Step 2. Compute Coordinate Gradient
        local_grad_size = math.ceil(input_ids.shape[0] // self.args.world_size)
        local_grad_slice = slice(
            self.args.local_rank * local_grad_size,
            (self.args.local_rank + 1) * local_grad_size,
        )
        coordinate_grad = mutli_token_gradients_bs(
            self.model,
            input_ids[local_grad_slice],
            self.control_slices[local_grad_slice],
            self.target_slices[local_grad_slice],
            self.loss_slices[local_grad_slice],
            past_input_ids=self.past_input_ids,
            past_key_values=self.past_key_values,
            batch_size=self.args.grad_batch_size,
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

            del adv_suffix_tokens, new_adv_suffix_toks, coordinate_grad
            gc.collect()
            torch.cuda.empty_cache()

            # Step 3.4 Compute loss on these candidates and take the argmin.
            losses, individual_losses = multi_get_losses(
                model=self.module,
                device=self.module.module.device,
                tokenizer=self.tokenizer,
                multi_input_ids=input_ids,
                control_slices=self.control_slices,
                target_slices=self.target_slices,
                test_controls=new_adv_suffix,
                return_indivisual_loss=True,
                batch_size=mini_batch_size,
                past_input_ids=self.past_input_ids,
                past_key_values=self.past_key_values,
            )  # decrease this number if you run into OOM.

            best_new_adv_suffix_id = losses.argmin()
            best_new_adv_suffix = new_adv_suffix[best_new_adv_suffix_id]

            current_loss = losses[best_new_adv_suffix_id]
            current_individual_loss = individual_losses[:, best_new_adv_suffix_id]

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

                updated_losses, updated_individual_losses = multi_get_losses(
                    model=self.module,
                    device=self.module.module.device,
                    tokenizer=self.tokenizer,
                    multi_input_ids=input_ids,
                    control_slices=self.control_slices,
                    target_slices=self.target_slices,
                    test_controls=updated_adv_suffix,
                    return_indivisual_loss=True,
                    batch_size=mini_batch_size,
                    past_input_ids=self.past_input_ids,
                    past_key_values=self.past_key_values,
                )

                best_updated_adv_suffix_id = updated_losses.argmin()
                best_updated_adv_suffix = updated_adv_suffix[best_updated_adv_suffix_id]
                min_updated_losses = updated_losses[best_updated_adv_suffix_id]

                if min_updated_losses < current_loss:
                    current_loss = min_updated_losses
                    current_individual_loss = updated_individual_losses[
                        :, best_updated_adv_suffix_id
                    ]
                    best_new_adv_suffix = best_updated_adv_suffix

                del updated_adv_suffix_toks, updated_losses

            self.last_loss = current_loss.detach().cpu()
            # Update the running adv_suffix with the best candidate
            if current_loss < self.min_loss:
                self.min_loss = current_loss.detach().cpu()

            adv_suffix = best_new_adv_suffix

            del losses
            gc.collect()
            torch.cuda.empty_cache()

        self.logger.info(
            f"Step: {step} | Current Loss: {current_loss:.4f} | Current Suffix: {best_new_adv_suffix} | Min Loss: {self.min_loss:.4f}"
        )
        self.log_suffix_to_jsonl(
            step=step,
            suffix=best_new_adv_suffix,
            loss=current_loss,
            min_loss=self.min_loss,
        )

        self.logger.info(
            f"Step: {step} |"
            + "|".join(
                [
                    f" Behavior {i}: {current_individual_loss[i]:.4f} "
                    for i in range(len(current_individual_loss))
                ]
            )
        )

        if step % 10 == 0 and self.args.debug:
            self.generate(adv_suffix=adv_suffix, step=step)

        return adv_suffix


class ProgressiveTrainer(MultiBehaviorTrainer):
    """Progressive add multiple optimization target accordding to current average loss."""

    def __init__(
        self,
        args,
        init_adv_suffix,
        user_prompts,
        prompt_templates,
        targets,
        bs_scheduler,
        min_loss=10000,
        adv_last_step=0,
        logger=None,
    ):
        self.set_all_prompt(
            all_user_prompts=user_prompts,
            all_prompt_templates=prompt_templates,
            all_targets=targets,
        )
        self.progress_cnt = 0
        self.update_prompt(update_slice=False)

        super().__init__(
            args,
            init_adv_suffix,
            self.user_prompts,
            self.prompt_templates,
            self.targets,
            bs_scheduler,
            min_loss,
            adv_last_step,
            logger,
        )

    def set_all_prompt(
        self, all_user_prompts=[], all_prompt_templates=[], all_targets=[]
    ):
        self.all_user_prompts = all_user_prompts
        self.all_prompt_templates = all_prompt_templates
        self.all_targets = all_targets

    def update_prompt(self, update_slice=True, adv_suffix=None):
        self.progress_cnt += 1
        self.user_prompts = self.all_user_prompts[: self.progress_cnt]
        self.prompt_templates = self.all_prompt_templates[: self.progress_cnt]
        self.targets = self.all_targets[: self.progress_cnt]

        if update_slice:
            (
                _,
                self.target_slices,
                self.loss_slices,
                self.control_slices,
                self.assistant_role_slices,
                self.instruction_slices,
            ) = self.get_input_ids(adv_suffix, return_slice=True)

    def clear_state(self):
        self.bs_scheduler.clear_state()
        self.min_loss = 10000

    def update_min_loss(self, adv_suffix):
        """Get loss of multiple prompt on current adv_suffix"""
        input_ids = self.get_input_ids(
            adv_suffix=adv_suffix, target_slices=self.target_slices
        )

        input_ids = input_ids.to(self.model.device)

        losses, individual_losses = multi_get_losses(
            model=self.module,
            device=self.module.module.device,
            tokenizer=self.tokenizer,
            multi_input_ids=input_ids,
            control_slices=self.control_slices,
            target_slices=self.target_slices,
            test_controls=[adv_suffix],
            return_indivisual_loss=True,
            batch_size=512,
            past_input_ids=self.past_input_ids,
            past_key_values=self.past_key_values,
        )
        del input_ids
        gc.collect()
        torch.cuda.empty_cache()

        return losses[0], individual_losses[:, 0]

    def run(self):
        adv_suffix = self.init_adv_suffix
        self.min_loss, individual_losses = self.update_min_loss(adv_suffix)

        self.logger.info(
            f"Starting | Min loss: {self.min_loss} | "
            + "|".join(
                [
                    f" Behavior {i}: {individual_losses[i]:.4f} "
                    for i in range(len(individual_losses))
                ]
            )
        )

        for i in range(self.adv_last_step, self.args.num_steps):
            # Progressive insert goals if loss meets requirements
            with torch.no_grad():
                while (
                    self.min_loss < self.args.progress_update_loss
                    and self.progress_cnt < len(self.all_prompt_templates)
                ):
                    self.update_prompt(update_slice=True, adv_suffix=adv_suffix)
                    self.logger.info(
                        f"Progressively add {self.progress_cnt} behaviors."
                    )
                    self.clear_state()
                    self.min_loss, individual_losses = self.update_min_loss(adv_suffix)
                    self.logger.info(
                        f"Progressive Add: {self.progress_cnt} | Min loss: {self.min_loss} | "
                        + "|".join(
                            [
                                f" Behavior {i}: {individual_losses[i]:.4f} "
                                for i in range(len(individual_losses))
                            ]
                        )
                    )
                    del individual_losses
                    gc.collect()
                    torch.cuda.empty_cache()

            adv_suffix = self.step(i, adv_suffix)

        self.generate(adv_suffix=adv_suffix, max_new_tokens=256)
