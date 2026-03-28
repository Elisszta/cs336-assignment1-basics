import json

import torch
import numpy as np
import torch.nn as nn
from torch import Tensor, dtype
from typing import cast
from pathlib import Path
import einx
from tqdm import tqdm

from cs336_basics.dataloader import DataLoader, Prefetcher, save_checkpoint
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import Transformer_LM
from cs336_basics.tools import CrossEntropyLoss, AdamW, CosineAnnealingLR, GradientClipping


def train(config_path: str):
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device is {device}.")

    # Load configs
    with open(config_path, encoding="utf-8") as f:
        configs = json.load(f)
    dataset_path, vocab_filepath, merges_filepath, special_tokens, text_path, batch_size, context_len = (
        configs["dataset_path"],
        configs["vocab_filepath"],
        configs["merges_filepath"],
        configs["special_tokens"],
        configs["text_path"],
        configs["batch_size"],
        configs["context_len"],
    )
    print("Settings loaded.")

    # Load data
    dataloader = DataLoader(device)
    if not Path(dataset_path).is_file():
        t = Tokenizer().from_files(vocab_filepath, merges_filepath, special_tokens)
        with open(dataset_path, "wb") as f_out:
            with open(text_path, encoding="utf-8") as f_in:
                for line in tqdm(f_in, desc="Processing Chunks"):
                    tokens = t.encode(line)
                    if tokens:
                        np.array(tokens, dtype=np.int32).tofile(f_out)
    dataset = np.memmap(dataset_path, np.int32, "r")
    print("Dataset loaded.")

    # Load LM
    vocab_size, d_model, num_layers, num_heads, d_ff, rope_theta, data_type = (
        configs["vocab_size"],
        configs["d_model"],
        configs["num_layers"],
        configs["num_heads"],
        configs["d_ff"],
        configs["rope_theta"],
        configs["dtype"],
    )
    if data_type == "bfloat16":
        data_type = torch.bfloat16
    elif data_type == "float16":
        data_type = torch.float16
    else:
        data_type = torch.float32

    model = Transformer_LM(
        vocab_size,
        context_len,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta,
        device=torch.device(device),
        dtype=data_type,
    )
    model = cast(nn.Module, torch.compile(model))

    # Define tools
    max_lr, min_lr, warmup_it, cosine_it, max_l2_norm, eps = (
        configs["max_lr"],
        configs["min_lr"],
        configs["warmup_it"],
        configs["cosine_it"],
        configs["max_l2_norm"],
        configs["eps"],
    )
    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters())
    scheduler = CosineAnnealingLR(max_lr, min_lr, warmup_it, cosine_it)
    clipper = GradientClipping(max_l2_norm, eps)

    # Training
    steps = 0
    epochs, buffer_size, steps_per_epoch, save_path = (
        configs["epochs"],
        configs["buffer_size"],
        configs["steps_per_epoch"],
        configs["save_path"],
    )
    # Using prefetcher to multi-thread-ly build a queue for dataload, enhance data load efficiency
    prefetcher = Prefetcher(dataloader.iter_load(dataset, batch_size, context_len), buffer_size)

    for epoch in range(epochs):
        # tqdm progress bar
        progress = tqdm(enumerate(prefetcher), total=steps_per_epoch, desc=f"Epoch {epoch}: ")
        # Lazy load inputs and labels in dataloader
        for batch_it, (inputs, labels) in progress:
            if batch_it >= steps_per_epoch:
                break
            # Update lr using annealing
            cur_lr = scheduler(steps)
            for group in optimizer.param_groups:
                group["lr"] = cur_lr

            # Clearing out grads
            optimizer.zero_grad()

            # Forwards Propagation
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = einx.rearrange("b s v -> (b s) v", model(inputs))
                    labels = einx.rearrange("b s -> (b s)", labels)
                    loss = criterion(logits, labels)
            else:
                logits = einx.rearrange("b s v -> (b s) v", model(inputs))
                labels = einx.rearrange("b s -> (b s)", labels)
                loss = criterion(logits, labels)

            if batch_it % 100 == 0:
                progress.set_postfix({"loss": f"{loss.item():.5f}"})

            # if steps % 1000 == 0:
            #     save_checkpoint(model, optimizer, steps, save_path + f"_{steps}.pt")

            # Backward Propagation
            loss.backward()

            # Clipping Grads
            clipper(model.parameters())

            # Optimizer updates
            optimizer.step()

            # Schedular it updates
            steps += 1


if __name__ == "__main__":
    train("configs/configs.json")
