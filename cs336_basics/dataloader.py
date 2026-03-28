import os
import queue
import threading
from typing import IO, BinaryIO

import numpy.typing as npt
import torch
import einx


class DataLoader:
    def __init__(self, device: str) -> None:
        self.device = device

    def load(self, dataset: npt.NDArray, batch_size: int, context_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        start_idx = torch.randint(len(dataset) - context_length, (batch_size,))
        length_idx = torch.arange(context_length)

        # Fetch all x, y tensors once, avoiding fragment reading.
        x_idx = einx.add("b, l -> b l", start_idx, length_idx).numpy()
        y_idx = x_idx + 1

        # Using pin memory to allow GPU using DMA to read from memory
        x_batch = torch.from_numpy(dataset[x_idx]).pin_memory()
        y_batch = torch.from_numpy(dataset[y_idx]).pin_memory()
        # And not to block cpu for next step
        return (x_batch.to(self.device, non_blocking=True), y_batch.to(self.device, non_blocking=True))

    def iter_load(self, dataset: npt.NDArray, batch_size: int, context_length: int):
        while True:
            yield self.load(dataset, batch_size, context_length)

    def __call__(self, dataset: npt.NDArray, batch_size: int, context_length: int):
        return self.load(dataset, batch_size, context_length)


class Prefetcher:
    def __init__(self, generator, buffer_size: int = 3) -> None:
        self.queue = queue.Queue(buffer_size)
        self.generator = generator
        self.thread = threading.Thread(target=self._fill_queue, daemon=True)
        self.thread.start()

    def _fill_queue(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        print(f"DEBUG: Buffer size is {self.queue.qsize()}")
        item = self.queue.get()
        if item is None:
            raise StopIteration
        return item


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
