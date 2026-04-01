import json

import torch

from cs336_basics.dataloader import load_checkpoint
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.tools import NucleusSampling
from cs336_basics.transformer import Transformer_LM


def generate(inputs: str, checkpoint_path: str, config_path: str):
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device is {device}.")

    # Load LM
    with open(config_path, encoding="utf-8") as f:
        configs = json.load(f)
        (
            vocab_filepath,
            merges_filepath,
            special_tokens,
            vocab_size,
            d_model,
            num_layers,
            num_heads,
            d_ff,
            rope_theta,
            data_type,
            context_len,
        ) = (
            configs["vocab_filepath"],
            configs["merges_filepath"],
            configs["special_tokens"],
            configs["vocab_size"],
            configs["d_model"],
            configs["num_layers"],
            configs["num_heads"],
            configs["d_ff"],
            configs["rope_theta"],
            configs["dtype"],
            configs["context_len"],
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

    # Load checkpoint
    load_checkpoint(checkpoint_path, model)

    t = Tokenizer().from_files(vocab_filepath, merges_filepath, special_tokens)
    input_token_ids = torch.tensor([t.encode(inputs)], device=device, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        for _ in range(context_len):
            input_token_ids = input_token_ids[:, -context_len:]
            logits = model(input_token_ids)[:, -1, :]
            next_token_id = NucleusSampling(logits, 0.6, 0.9)
            input_token_ids = torch.cat([input_token_ids, next_token_id], -1)
            if next_token_id.item() == t.encode_vocab[b"<|endoftext|>"]:
                break
    print(t.decode(input_token_ids[0].tolist()))


if __name__ == "__main__":
    inputs = "Across the river and sea, there"
    generate(inputs, "ckpt/task_tinystories_params_latest.pt", "configs/configs.json")
