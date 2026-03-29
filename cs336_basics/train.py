import json
import math

import torch
import numpy as np
import torch.nn as nn
from typing import cast
from pathlib import Path
import einx
from tqdm import tqdm
import wandb

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
    (
        dataset_path,
        valid_path,
        vocab_filepath,
        merges_filepath,
        special_tokens,
        text_path,
        valid_text_path,
        batch_size,
        context_len,
        wandb_entity,
        proj_name,
    ) = (
        configs["dataset_path"],
        configs["valid_path"],
        configs["vocab_filepath"],
        configs["merges_filepath"],
        configs["special_tokens"],
        configs["text_path"],
        configs["valid_text_path"],
        configs["batch_size"],
        configs["context_len"],
        configs["wandb_entity"],
        configs["proj_name"],
    )
    run = wandb.init(entity=wandb_entity, project=proj_name, config=configs)
    print("Settings loaded.")

    # Load data
    dataloader = DataLoader(device)
    if not Path(dataset_path).is_file():
        t = Tokenizer().from_files(vocab_filepath, merges_filepath, special_tokens)
        with open(dataset_path, "wb") as f_out:
            with open(text_path, encoding="utf-8") as f_in:
                for line in tqdm(f_in, desc="Processing Train Dataloader"):
                    tokens = t.encode(line)
                    if tokens:
                        np.array(tokens, dtype=np.int32).tofile(f_out)
    if not Path(valid_path).is_file():
        t = Tokenizer().from_files(vocab_filepath, merges_filepath, special_tokens)
        with open(valid_path, "wb") as f_out:
            with open(valid_text_path, encoding="utf-8") as f_in:
                for line in tqdm(f_in, desc="Processing Valid Dataloader"):
                    tokens = t.encode(line)
                    if tokens:
                        np.array(tokens, dtype=np.int32).tofile(f_out)
    dataset = np.memmap(dataset_path, np.int32, "r")
    validset = np.memmap(valid_path, np.int32, "r")
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
    epochs, buffer_size, steps_per_epoch, val_size, save_path = (
        configs["epochs"],
        configs["buffer_size"],
        configs["steps_per_epoch"],
        configs["val_size"],
        configs["save_path"],
    )
    # Using prefetcher to multi-thread-ly build a queue for dataload, enhance data load efficiency
    prefetcher_train = Prefetcher(dataloader.iter_load(dataset, batch_size, context_len), buffer_size)
    prefetcher_valid = Prefetcher(dataloader.iter_load(validset, batch_size, context_len), buffer_size)
    best_loss = float("inf")
    for epoch in range(epochs):
        # tqdm progress bar
        progress = tqdm(enumerate(prefetcher_train), total=steps_per_epoch, desc=f"Epoch {epoch}: ")
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

            # Forwards Propagation, using mixed precision
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = einx.rearrange("b s v -> (b s) v", model(inputs))
                    labels = einx.rearrange("b s -> (b s)", labels)
                    loss = criterion(logits, labels)
            else:
                logits = einx.rearrange("b s v -> (b s) v", model(inputs))
                labels = einx.rearrange("b s -> (b s)", labels)
                loss = criterion(logits, labels)

            # 10 steps to update loss, 2000 steps to save latest ckpt, after 5000 steps to save best ckpt
            if batch_it % 10 == 0:
                cur_loss = loss.item()
                if cur_loss < best_loss:
                    best_loss = cur_loss
                    if steps > 5000:
                        save_checkpoint(model, optimizer, steps, save_path + "_best.pt")
                run.log({"iter": steps, "loss": cur_loss, "ppl": math.exp(cur_loss), "lr": cur_lr})
                progress.set_postfix({"loss": f"{cur_loss:.5f}"})
                if batch_it % 2000 == 0:
                    valid_loss = valid(model, prefetcher_valid, val_size, criterion, device)
                    run.log({"iter": steps, "val_loss": valid_loss, "val_ppl": math.exp(valid_loss)})
                    save_checkpoint(model, optimizer, steps, save_path + "_latest.pt")

            # Backward Propagation
            loss.backward()

            # Clipping Grads
            clipper(model.parameters())

            # Optimizer updates
            optimizer.step()

            # Schedular it updates
            steps += 1


def valid(model, val_loader, val_size, criterion, device) -> float:
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for it, (inputs, labels) in enumerate(val_loader):
            if it >= val_size:
                break
            with torch.autocast(device, torch.bfloat16 if device == "cuda" else torch.float32):
                logits = model(inputs)
                logits = einx.rearrange("b s v -> (b s) v", logits)
                labels = einx.rearrange("b s -> (b s)", labels)
                total_loss += criterion(logits, labels).item()
    avg_loss = total_loss / val_size
    model.train()
    return avg_loss


if __name__ == "__main__":
    train("configs/configs.json")
