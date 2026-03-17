import os
from os import PathLike
from collections import defaultdict
import pickle
from typing import BinaryIO, Iterable, Iterator, Optional

from multiprocessing import Pool
import regex as re


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes] | None = None,
        merges: list[tuple[bytes, bytes]] | None = None,
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocabulary = vocab if vocab is not None else {}
        self.merges = merges if merges is not None else []
        self.special_tokens = (
            special_tokens if special_tokens is not None else ["<|endoftext|>"]
        )
        self.special_tokens_set = {t.encode("utf-8") for t in self.special_tokens}

        self.encode_vocab: dict[bytes, int] = {}
        self.merges_dict: dict[tuple[bytes, bytes], int] = {}
        if self.vocabulary and self.merges:
            self._update_internal_states()

    def _update_internal_states(self):
        self.encode_vocab = {v: k for k, v in self.vocabulary.items()}
        self.merges_dict = {pair: i for i, pair in enumerate(self.merges)}

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

    def find_string_boundaries(
        self, raw_data: str, desired_num_string_chunks: int
    ) -> list[int]:
        # assert isinstance(
        #     split_special_token, bytes
        # ), "Must represent special token as a bytestring"

        # Get total file size in bytes
        str_len = len(raw_data)

        chunk_size = str_len // desired_num_string_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [
            i * chunk_size for i in range(desired_num_string_chunks + 1)
        ]
        chunk_boundaries[-1] = str_len

        mini_chunk_size = 1024  # Must larger than any special token

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            while True:
                if initial_position + mini_chunk_size >= str_len:
                    chunk_boundaries[bi] = str_len
                    break

                min_found_at = str_len
                for split_special_token in self.special_tokens:
                    found_at = raw_data.find(
                        split_special_token,
                        initial_position,
                        initial_position + mini_chunk_size,
                    )
                    min_found_at = (
                        min(min_found_at, found_at) if found_at != -1 else min_found_at
                    )

                if min_found_at != str_len:
                    chunk_boundaries[bi] = min_found_at + len(split_special_token)
                    break
                initial_position += (
                    mini_chunk_size - 100
                )  # Avoid cutting down the str at the special token

        return sorted(set(chunk_boundaries))

    @staticmethod
    def train_pretokenizer(raw_data: str) -> dict:
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
    def encode_pretokenizer(raw_data: str) -> list[bytes]:
        text_words = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        regex_tokenized_it = re.finditer(PAT, raw_data)
        for word in regex_tokenized_it:
            encoded_word = word.group().encode("utf-8")
            text_words.append(encoded_word)
        return text_words

    @staticmethod
    def train_process_chunk(args) -> dict:
        file_path, start, end, special_tokens = args
        special_tokens = sorted(special_tokens, key=len, reverse=True)
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
                for word, freq in Tokenizer.train_pretokenizer(story).items():
                    final_dict[word] += freq
            return final_dict

    @staticmethod
    def encode_stream_chunk(args) -> Iterator[bytes]:
        chunk_str, start, end, special_tokens = args
        special_tokens = sorted(special_tokens, key=len, reverse=True)
        split_pattern = "|".join(re.escape(t) for t in special_tokens)
        compiled_pattern = re.compile(split_pattern)
        last_pos = start
        for match in compiled_pattern.finditer(chunk_str, start, end):
            text_before = chunk_str[last_pos : match.start()]
            if text_before:
                for word_bytes in Tokenizer.encode_pretokenizer(text_before):
                    yield word_bytes
            yield match.group().encode("utf-8")
            last_pos = match.end()
        text_remaining = chunk_str[last_pos : len(chunk_str)]
        if text_remaining:
            for word_bytes in Tokenizer.encode_pretokenizer(text_remaining):
                yield word_bytes

    @staticmethod
    def encode_process_chunk(args) -> list[bytes]:
        chunk_str, start, end, special_tokens = args
        sorted_tokens = sorted(special_tokens, key=len, reverse=True)
        split_pattern = "(" + "|".join(re.escape(t) for t in sorted_tokens) + ")"
        raw_parts = re.split(split_pattern, chunk_str[start:end])
        final_list = []
        for part in raw_parts:
            if not part:
                continue
            if part in special_tokens:
                final_list.append(part.encode("utf-8"))
            else:
                final_list.extend(Tokenizer.encode_pretokenizer(part))
        return final_list

    def parallel_process_train(self, task_num: int, file_path: str | PathLike) -> dict:
        with open(file_path, "rb") as f:
            sentence_split = self.special_tokens[0].encode("utf-8")
            boundaries = self.find_chunk_boundaries(f, task_num, sentence_split)
        tasks = [
            [file_path, start, end, self.special_tokens]
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

        with Pool(processes=task_num) as pool:
            results = pool.map(self.train_process_chunk, tasks)
        global_words = defaultdict(int)
        for result in results:
            for word, freq in result.items():
                global_words[word] += freq

        return global_words

    def parallel_process_encode(self, task_num: int, input_str: str) -> list[bytes]:
        boundaries = self.find_string_boundaries(input_str, task_num)
        tasks = [
            [input_str, start, end, self.special_tokens]
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

        with Pool(processes=task_num) as pool:
            results = pool.map(self.encode_process_chunk, tasks)
        global_bytes = []
        for result in results:
            global_bytes += result

        return global_bytes

    def tokenizer_trainer(
        self,
        dataset_path: str | PathLike = "../datasets/TinyStories/validation.txt",
        vocab: int = 1000,
        task_num: int = 4,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        words = self.parallel_process_train(task_num, dataset_path)
        self.initialize_vocabulary(self.special_tokens)

        top_pair = []
        concat_pair = ""
        pair_indices = defaultdict(set)
        freq_count = defaultdict(int)

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
            self.merges.append(top_pair)

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

            self._update_internal_states()

        return self.vocabulary, self.merges

    def _merge_bytes_to_ids(self, word: bytes) -> list[int]:
        bytes_ids = [word[i : i + 1] for i in range(len(word))]
        while len(bytes_ids) > 1:
            # iter-ly find best pair and prepare to merge
            best_pair = None
            best_pair_rank = float("inf")
            for i in range(len(bytes_ids) - 1):
                curr_pair = (bytes_ids[i], bytes_ids[i + 1])
                if (
                    curr_pair in self.merges_dict
                    and self.merges_dict[curr_pair] < best_pair_rank
                ):
                    best_pair, best_pair_rank = curr_pair, self.merges_dict[curr_pair]
            if best_pair == None:
                break
            i = 0
            new_bytes_ids = []
            while i < len(bytes_ids):
                if (
                    i < len(bytes_ids) - 1
                    and (bytes_ids[i], bytes_ids[i + 1]) == best_pair
                ):
                    new_bytes_ids.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_bytes_ids.append(bytes_ids[i])
                    i += 1
            bytes_ids = new_bytes_ids
        token_ids = []
        for byte in bytes_ids:
            token_ids.append(self.encode_vocab[byte])
        return token_ids

    def encode(self, text: str, multiprocess: bool = False) -> list[int]:
        encoded_list = []
        if multiprocess:
            words = self.parallel_process_encode(8, text)
            for word in words:
                if word in self.special_tokens_set:
                    encoded_list.append(self.encode_vocab[word])
                else:
                    encoded_list.extend(self._merge_bytes_to_ids(word))
        else:
            token_stream = self.encode_stream_chunk(
                [text, 0, len(text), self.special_tokens]
            )
            for word_bytes in token_stream:
                if word_bytes in self.special_tokens_set:
                    encoded_list.append(self.encode_vocab[word_bytes])
                else:
                    encoded_list.extend(self._merge_bytes_to_ids(word_bytes))
        return encoded_list

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text_chunk in iterable:
            token_ids = self.encode(text_chunk)
            yield from token_ids

    def decode(self, ids: list[int]) -> str:
        token_bytes = [self.vocabulary[token_id] for token_id in ids]
        return b"".join(token_bytes).decode("utf-8", errors="ignore")

    def save(
        self,
        vocab_path: str = "vocab.pkl",
        merges_path: str = "merges.pkl",
    ):
        vocab_data, merges_data = (
            self.vocabulary,
            self.merges,
        )
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(merges_path, "wb") as f:
            pickle.dump(merges_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Model data saved.")

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[list[str]] = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)


import time

if __name__ == "__main__":
    # start_time = time.perf_counter()
    # t = Tokenizer()
    # vocab, _ = t.tokenizer_trainer(
    #     dataset_path="../datasets/TinyStories/train.txt", vocab=10000, task_num=10
    # )
    # end_time = time.perf_counter()
    # t.save("TinyStories10K_vocab.pkl", "TinyStories10K_merges.pkl")
    # print(f"Training finished. Tokenizer training time is: {end_time - start_time}.")

    test_str = "This is a test,<|endoftext|> <|endoftext|> if the Tokenizer functioning well. <|endoftext|>"
    newT = Tokenizer.from_files("TinyStories10K_vocab.pkl", "TinyStories10K_merges.pkl")
    encoded = newT.encode(test_str)
    decoded = newT.decode(encoded)
    print(
        f"Raw text are: {test_str}, Encoded tokens are: {encoded}, Decoded text are: {decoded}"
    )
    assert test_str == decoded
    print("Encode & decode test finished!")
