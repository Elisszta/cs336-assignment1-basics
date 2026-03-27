import json

import torch
import numpy as np
import torch.nn as nn
from torch import Tensor, dtype
from pathlib import Path

from cs336_basics.dataloader import DataLoader
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import Transformer_LM
from cs336_basics.tools import CrossEntropyLoss, AdamW, CosineAnnealingLR, GradientClipping
from jaxtyping import Int


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
        configs["special_tokens"],
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
    vocab_size, d_model, num_layers, num_heads, d_ff, rope_theta, dtype = (
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
    )

    # Define tools
    max_lr, min_lr, warmup_it, cosine_it, max_l2_norm = {
        configs["max_lr"],
        configs["min_lr"],
        configs["warmup_it"],
        configs["cosine_it"],
        configs["max_l2_norm"],
    }
    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters())
    scheduler = CosineAnnealingLR(max_lr, min_lr, warmup_it, cosine_it)

    # Training
    it = 0
    epochs = configs["epochs"]
    for epoch in range(epochs):
        for inputs, labels in dataloader(dataset, batch_size, context_len):
            cur_lr = scheduler(it)
            for group in optimizer.param_groups:
                group["lr"] = cur_lr
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
