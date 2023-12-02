from dataclasses import dataclass
import logging


@dataclass
class BaseScheduler:
    batch_size: int
    mini_batch_size: int

    def step(self, step, **kwargs):
        """Return current batch size and mini batch size according to args."""
        raise NotImplementedError


class MultiStepScheduler(BaseScheduler):
    def __init__(
        self, multi_batch_sizes, multi_mini_batch_sizes, multi_losses, logger=None
    ):
        self.multi_losses = multi_losses
        self.multi_batch_sizes = multi_batch_sizes
        self.multi_mini_batch_sizes = multi_mini_batch_sizes

        self.index = 0
        self.step_num = 0

        if logger:
            self.verbose = True
            self.logger = logger
        else:
            self.verbose = False

        if len(multi_batch_sizes) != len(multi_losses) + 1:
            raise ValueError(
                f"The number of multi_batch_sizes should be the number of multi_losses + 1. Current is {len(multi_batch_sizes)} and {len(multi_losses)}."
            )

    def step(self, loss=None):
        self.step_num += 1
        if loss is None:
            if self.verbose:
                self.logger.info(
                    f"Current batch_size: {self.multi_batch_sizes[self.index]}; mini_batch_size: {self.multi_mini_batch_sizes[self.index]}"
                )
            return (
                self.multi_batch_sizes[self.index],
                self.multi_mini_batch_sizes[self.index],
            )

        if self.index == len(self.multi_batch_sizes) - 1:
            return (
                self.multi_batch_sizes[self.index],
                self.multi_mini_batch_sizes[self.index],
            )

        if loss < self.multi_losses[self.index]:
            self.index += 1

            if self.verbose:
                self.logger.info(
                    f"Current batch_size: {self.multi_batch_sizes[self.index]}; mini_batch_size: {self.multi_mini_batch_sizes[self.index]}"
                )

        return (
            self.multi_batch_sizes[self.index],
            self.multi_mini_batch_sizes[self.index],
        )

    def clear_state(self):
        # start from the initial batch size setting
        self.index = 0
