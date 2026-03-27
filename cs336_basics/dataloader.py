import os
from typing import IO, BinaryIO

import numpy.typing as npt
import torch


class DataLoader:
    def __init__(self, device: str) -> None:
        self.device = device

    def load(self, dataset: npt.NDArray, batch_size: int, context_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        start_idx = torch.randint(len(dataset) - context_length - 1, (batch_size,))

        x_batch = torch.stack([torch.from_numpy(dataset[i : i + context_length]) for i in start_idx])

        y_batch = torch.stack([torch.from_numpy(dataset[i + 1 : i + context_length + 1]) for i in start_idx])

        return (x_batch.to(self.device), y_batch.to(self.device))

    def iter_load(self, dataset: npt.NDArray, batch_size: int, context_length: int):
        while True:
            yield self.load(dataset, batch_size, context_length)

    def __call__(self, dataset: npt.NDArray, batch_size: int, context_length: int):
        return self.load(dataset, batch_size, context_length)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    torch.save({"model_params": model.state_dict(), "optim_params": optimizer.state_dict(), "it": iteration}, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_params"])
    optimizer.load_state_dict(checkpoint["optim_params"])
    return checkpoint.get("it", 0)
