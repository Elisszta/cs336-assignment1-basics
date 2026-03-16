import os
from os import PathLike
from collections import defaultdict
from typing import BinaryIO

from multiprocessing import Pool
import regex as re


class Tokenizer:
    def __init__(self) -> None:
        self.vocabulary: dict[int, bytes] = {}

    def initialize_vocabulary(self, special_tokens: list[str]):
        self.vocabulary = {i: bytes([i]) for i in range(256)}
        for token in special_tokens:
            self.vocabulary[len(self.vocabulary)] = token.encode("utf-8")

    def find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(
            split_special_token, bytes
        ), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    @staticmethod
    def pretokenizer(raw_data: str) -> dict:
        """
        Used to pre process the string by using regex
        Returns:
        Key: Word
        Value: Freq
        """
        text_words = defaultdict(int)
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        regex_tokenized_it = re.finditer(PAT, raw_data)
        for word in regex_tokenized_it:
            encoded_word = word.group().encode("utf-8")
            text_words[
                tuple(encoded_word[i : i + 1] for i in range(len(encoded_word)))
            ] += 1
        return text_words

    @staticmethod
    def process_chunk(args) -> dict:
        file_path, start, end, special_tokens = args
        split_pattern = "|".join(re.escape(t) for t in special_tokens)
        with open(file_path, "rb") as f:
            f.seek(start)
            chunk_data = (
                f.read(end - start)
                .decode("utf-8", errors="ignore")
                .replace("\r\n", "\n")
                .replace("\r", "\n")
            )
            stories = re.split(split_pattern, chunk_data)
            final_dict = defaultdict(int)
            for story in stories:
                if not story or story in special_tokens:
                    continue
                for word, freq in Tokenizer.pretokenizer(story).items():
                    final_dict[word] += freq
            return final_dict

    def parallel_process(
        self, task_num: int, file_path: str | PathLike, special_tokens: list[str]
    ) -> dict:
        with open(file_path, "rb") as f:
            sentence_split = special_tokens[0].encode("utf-8")
            boundaries = self.find_chunk_boundaries(f, task_num, sentence_split)
        tasks = [
            [file_path, start, end, special_tokens]
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

        with Pool(processes=task_num) as pool:
            results = pool.map(self.process_chunk, tasks)
        global_words = defaultdict(int)
        for result in results:
            for word, freq in result.items():
                global_words[word] += freq

        return global_words

    def tokenizer_trainer(
        self,
        dataset_path: str | PathLike = "../datasets/TinyStories/validation.txt",
        vocab: int = 1000,
        special_tokens: list[str] = ["<|endoftext|>"],
        task_num: int = 4,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        words = self.parallel_process(task_num, dataset_path, special_tokens)
        self.initialize_vocabulary(special_tokens)

        top_pair = []
        concat_pair = ""
        pair_indices = defaultdict(set)
        freq_count = defaultdict(int)
        merges = []

        # Preprocess the word to maintain a valid pair indices and freq count
        for word, freq in words.items():
            for i in range(len(word) - 1):
                pair_indices[(word[i], word[i + 1])].add(word)
                freq_count[(word[i], word[i + 1])] += freq

        merge_times = vocab - len(self.vocabulary)
        for _ in range(merge_times):
            top_pair = max(freq_count, key=lambda p: (freq_count[p], p))
            concat_pair = top_pair[0] + top_pair[1]
            self.vocabulary[len(self.vocabulary)] = concat_pair
            merges.append(top_pair)

            # Only need to update words that are in the top pair indices
            affected_words = list(pair_indices[top_pair])
            for word in affected_words:
                prev_word = word
                freq = words.pop(word)
                # Delete all pair indices and frequency counts from previous word
                for i in range(len(prev_word) - 1):
                    freq_count[(prev_word[i], prev_word[i + 1])] -= freq
                    pair_indices[(prev_word[i], prev_word[i + 1])].discard(prev_word)
                    if freq_count[(prev_word[i], prev_word[i + 1])] == 0:
                        del freq_count[(prev_word[i], prev_word[i + 1])]
                # Performing merging operation
                new_word = []
                i = 0
                while i < len(prev_word):
                    if (
                        i < len(prev_word) - 1
                        and prev_word[i] == top_pair[0]
                        and prev_word[i + 1] == top_pair[1]
                    ):
                        new_word.append(concat_pair)
                        i += 2
                    else:
                        new_word.append(prev_word[i])
                        i += 1
                t_new_word = tuple(new_word)
                words[t_new_word] = words.get(t_new_word, 0) + freq
                # Adding pair indices and frequency counts based on new word
                for i in range(len(new_word) - 1):
                    freq_count[(new_word[i], new_word[i + 1])] += freq
                    pair_indices[(new_word[i], new_word[i + 1])].add(t_new_word)
            # Delete useless pairs
            if top_pair in pair_indices:
                del pair_indices[top_pair]

        return self.vocabulary, merges


if __name__ == "__main__":
    t = Tokenizer()
    vocab, _ = t.tokenizer_trainer(vocab=10000)
    with open("vocab.txt", "w", encoding="utf-8") as f:
        for idx, token_bytes in vocab.items():
            token_str = token_bytes.decode("utf-8", errors="replace")
            f.write(f"{idx}: {token_str}\n")
