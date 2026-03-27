import json

import torch
import numpy as np
import torch.nn as nn
from torch import Tensor, dtype
from typing import cast
from pathlib import Path
import einx

from cs336_basics.dataloader import DataLoader
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import Transformer_LM
from cs336_basics.tools import CrossEntropyLoss, AdamW, CosineAnnealingLR, GradientClipping


def train(config_path: str):
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    # Load data
    dataloader = DataLoader(device)
    if not Path(dataset_path).is_file():
        t = Tokenizer().from_files(vocab_filepath, merges_filepath, special_tokens)
        with open(text_path, encoding="utf-8") as f:
            token_stream = t.encode_iterable(f)
            token_np = np.fromiter(token_stream, dtype=np.int32)
        token_np.tofile(dataset_path)
    dataset = np.memmap(dataset_path, np.int32, "r")

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
    epochs = configs["epochs"]
    for epoch in range(epochs):
        # Lazy load inputs and labels in dataloader
        for batch_it, (inputs, labels) in dataloader.iter_load(dataset, batch_size, context_len):
            if batch_it >= configs["steps_per_epoch"]:
                break
            # Update lr using annealing
            cur_lr = scheduler(steps)
            for group in optimizer.param_groups:
                group["lr"] = cur_lr

            # Load datasets
            inputs, labels = inputs.to(device), labels.to(device)

            # Clearing out grads
            optimizer.zero_grad()

            # Forwards Propagation
            logits = einx.rearrange("b s v -> (b s) v", model(inputs))
            labels = einx.rearrange("b s -> (b s)", labels)
            loss = criterion(logits, labels)

            if batch_it % 100 == 0:
                print(f"Step: {it}, Current loss: {loss}")

            # Backward Propagation
            loss.backward()

            # Clipping Grads
            clipper(model.parameters())

            # Optimizer updates
            optimizer.step()

            # Schedular it updates
            steps += 1
