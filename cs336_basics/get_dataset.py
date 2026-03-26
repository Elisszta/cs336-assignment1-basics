# import os

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# from datasets import load_dataset
# from typing import Any, cast


# def save_as_txt():
#     dataset_dict = load_dataset("Skylion007/openwebtext", num_proc=8)
#     target_dir = "../datasets/openwebtext"
#     os.makedirs(target_dir, exist_ok=True)

#     END_OF_TEXT = "<|endoftext|>"

#     for split_name, dataset in dataset_dict.items():
#         save_path = os.path.join(target_dir, f"{split_name}.txt")
#         print(f"Transforming [{split_name}] to {save_path}...")

#         with open(save_path, "w", encoding="utf-8") as f:
#             for row in dataset:
#                 text_content = row["text"]  # type: ignore
#                 f.write(str(text_content) + END_OF_TEXT)

#     print("Finished")


# save_as_txt()
