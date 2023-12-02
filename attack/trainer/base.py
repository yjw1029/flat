import torch
import gc
import json
import deepspeed

from .utils import (
    get_input_ids,
    prepare_past_key_value,
    get_input_slices,
    get_input_ids_wo_target,
    gather_suffixes,
    get_nonascii_toks,
)
from .opt_utils import (
    get_filtered_cands,
    load_model_and_tokenizer,
    multi_token_gradients,
    multi_get_losses,
    generate,
    gather_grad_and_norm,
)
from .sample import sample_control2, sample_control


class Trainer:
    """Optimize adversarial suffix for a single behavior"""

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
        self.args = args
        self.logger = logger
        self.min_loss = min_loss
        self.last_loss = 10000
        self.adv_last_step = adv_last_step

        self.init_adv_suffix = init_adv_suffix
        self.set_prompt(
            user_prompts=user_prompts,
            prompt_templates=prompt_templates,
            targets=targets,
        )

        self._init_model_and_tokenizer()
        (
            _,
            self.target_slices,
            self.loss_slices,
            self.control_slices,
            self.assistant_role_slices,
            self.instruction_slices,
        ) = self.get_input_ids(init_adv_suffix, return_slice=True)

        self._init_scheduler(bs_scheduler)
        self._init_past_key_value(init_adv_suffix)
        self._init_not_allowed_tokens()

        self.sample_gen = torch.Generator(device=self.model.device)
        self.sample_gen.manual_seed(self.args.sample_seed)

    def log_suffix_to_jsonl(self, step, suffix, loss, min_loss):
        if self.args.suffix_jsonl_file and self.args.local_rank == 0:
            with open(self.args.suffix_jsonl_file, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "step": step,
                            "suffix": suffix,
                            "loss": loss.detach().cpu().numpy().tolist()
                            if torch.is_tensor(loss)
                            else loss,
                            "min_loss": min_loss.detach().cpu().numpy().tolist()
                            if torch.is_tensor(min_loss)
                            else min_loss,
                        }
                    )
                    + "\n"
                )

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

    def _init_model_and_tokenizer(self):
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.args.model_name_or_path,
            low_cpu_mem_usage=True,
            use_cache=False,
        )
        self.module = deepspeed.init_inference(
            self.model,
            mp_size=self.args.world_size,
            dtype=torch.float16,
            replace_with_kernel_inject=False,
            max_tokens=512,
        )

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
            self.model, input_ids, self.control_slices[0]
        )

    def _init_scheduler(self, bs_scheduler):
        self.bs_scheduler = bs_scheduler

    def _init_not_allowed_tokens(self):
        self.not_allowed_tokens = (
            None if self.args.allow_non_ascii else get_nonascii_toks(self.tokenizer)
        )

    def set_prompt(self, user_prompts=[], prompt_templates=[], targets=[]):
        self.user_prompts = user_prompts
        self.prompt_templates = prompt_templates
        self.targets = targets

    def get_input_ids(self, adv_suffix, target_slices=None, return_slice=False):
        if return_slice:
            (
                multi_input_ids,
                multi_target_slice,
                multi_loss_slice,
                multi_control_slice,
                multi_assistant_role_slice,
                multi_instruction_slice,
            ) = (
                [],
                [],
                [],
                [],
                [],
                [],
            )
            for prompt_template, target, user_prompt in zip(
                self.prompt_templates, self.targets, self.user_prompts
            ):
                (
                    input_ids,
                    target_slice,
                    loss_slice,
                    control_slice,
                    assistant_role_slice,
                    instruction_slice,
                ) = get_input_slices(
                    prompt_template=prompt_template,
                    target=target,
                    user_prompt=user_prompt,
                    adv_suffix=adv_suffix,
                    tokenizer=self.tokenizer,
                )
                multi_input_ids.append(input_ids)
                multi_target_slice.append(target_slice)
                multi_loss_slice.append(loss_slice)
                multi_control_slice.append(control_slice)
                multi_assistant_role_slice.append(assistant_role_slice)
                multi_instruction_slice.append(instruction_slice)
            return (
                multi_input_ids,
                multi_target_slice,
                multi_loss_slice,
                multi_control_slice,
                multi_assistant_role_slice,
                multi_instruction_slice,
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

    def generate(self, adv_suffix, step=None, max_new_tokens=32):
        for i in range(len(self.prompt_templates)):
            input_ids, attention_mask = get_input_ids_wo_target(
                prompt_template=[self.prompt_templates[i]],
                user_prompt=[self.user_prompts[i]],
                adv_suffix=adv_suffix,
                target=[self.targets[i]],
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
            if step:
                self.logger.info(
                    f"Step: {step} | Behavior: {i} | Completion: {completion}"
                )
            else:
                self.logger.info(f"Final | Behavior: {i} | Completion: {completion}")

    def step(self, step: int, adv_suffix=str):
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
            # Update the running adv_suffix with the best candidate
            if current_loss < self.min_loss:
                self.min_loss = current_loss.detach().cpu()

            adv_suffix = best_new_adv_suffix

            del losses
            gc.collect()
            torch.cuda.empty_cache()

        self.logger.info(
            f"Step: {step} | Current Loss: {current_loss:.4f} | Current Suffix: {json.dumps(best_new_adv_suffix)} | Min Loss: {self.min_loss:.4f}"
        )
        self.log_suffix_to_jsonl(
            step=step,
            suffix=best_new_adv_suffix,
            loss=current_loss,
            min_loss=self.min_loss,
        )

        if step % 10 == 0 and self.args.debug:
            self.generate(adv_suffix=adv_suffix, step=step)

        return adv_suffix

    def run(self):
        adv_suffix = self.init_adv_suffix
        for i in range(self.adv_last_step, self.args.num_steps):
            adv_suffix = self.step(i, adv_suffix)

        self.generate(adv_suffix=adv_suffix, max_new_tokens=256)
