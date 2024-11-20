"""Tools for tokenizing and manipulating text datasets."""

import math
from multiprocessing import cpu_count
from typing import Iterable, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizerBase

T = TypeVar("T", bound=Union[Dataset, DatasetDict])


def chunk_and_tokenize(
    data: T,
    tokenizer: PreTrainedTokenizerBase,
    *,
    format: str = "torch",
    num_proc: int = cpu_count() // 2,
    text_key: str = "text",
    max_seq_len: int = 2048,
    return_final_batch: bool = False,
    load_from_cache_file: bool = True,
) -> T:
    """Perform GPT-style chunking and tokenization on a dataset.

    The resulting dataset will consist entirely of chunks exactly `max_seq_len` tokens
    long. Long sequences will be split into multiple chunks, and short sequences will
    be merged with their neighbors, using `eos_token` as a separator. The fist token
    will also always be an `eos_token`.

    Args:
        data: The dataset to chunk and tokenize.
        tokenizer: The tokenizer to use.
        format: The format to return the dataset in, passed to `Dataset.with_format`.
        num_proc: The number of processes to use for tokenization.
        text_key: The key in the dataset to use as the text to tokenize.
        max_seq_len: The maximum length of a batch of input ids.
        return_final_batch: Whether to return the final batch, which may be smaller
            than the others.
        load_from_cache_file: Whether to load from the cache file.

    Returns:
        The chunked and tokenized dataset.
    """

    def _tokenize_fn(x: dict[str, list]):
        chunk_size = min(tokenizer.model_max_length, max_seq_len)
        sep = tokenizer.eos_token or "<|endoftext|>"
        joined_text = sep.join([""] + x[text_key])
        output = tokenizer(
            # Concatenate all the samples together, separated by the EOS token.
            joined_text,  # start with an eos token
            max_length=chunk_size,
            return_attention_mask=False,
            return_overflowing_tokens=True,
            truncation=True,
        )

        if overflow := output.pop("overflowing_tokens", None):
            # Slow Tokenizers return unnested lists of ints
            assert isinstance(output.input_ids[0], int)

            # Chunk the overflow into batches of size `chunk_size`
            chunks = [output["input_ids"]] + [
                overflow[i * chunk_size : (i + 1) * chunk_size]
                for i in range(math.ceil(len(overflow) / chunk_size))
            ]
            output = {"input_ids": chunks}

        if not return_final_batch:
            # We know that the last sample will almost always be less than the max
            # number of tokens, and we don't want to pad, so we just drop it.
            output = {k: v[:-1] for k, v in output.items()}

        output_batch_size = len(output["input_ids"])

        if output_batch_size == 0:
            raise ValueError(
                "Not enough data to create a single complete batch."
                " Either allow the final batch to be returned,"
                " or supply more data."
            )

        return output

    data = data.map(
        _tokenize_fn,
        # Batching is important for ensuring that we don't waste tokens
        # since we always throw away the last element of the batch we
        # want to keep the batch size as large as possible
        batched=True,
        batch_size=2048,
        num_proc=num_proc,
        remove_columns=get_columns_all_equal(data),
        load_from_cache_file=load_from_cache_file,
    )
    return data.with_format(format, columns=["input_ids"])


def chunk_and_tokenize_streaming(
    data: Tuple[DatasetDict | Dataset | IterableDatasetDict | IterableDataset],
    tokenizer: PreTrainedTokenizerBase,
    *,
    text_key: str = "text",
    max_seq_len: int = 2048,
    eos_token_id: Optional[int] = None,
) -> IterableDataset:
    """Perform GPT-style chunking and tokenization on a streaming dataset.

    The function yields individual chunks of tokenized data, each of length `max_seq_len`.
    Long sequences are split into multiple chunks, while short sequences are concatenated
    with their neighbors. No padding is added, and all tokens are fully utilized.

    Args:
        data: The streaming dataset to chunk and tokenize.
        tokenizer: The tokenizer to use.
        text_key: The key in the dataset to use as the text to tokenize.
        max_seq_len: The maximum length of a chunk of input ids.
        eos_token_id: The id of the eos token. If None, will use tokenizer.eos_token_id.

    Returns:
        An IterableDataset over the tokenized chunks.
    """
    # Identify columns to remove
    columns_to_remove = get_columns_all_equal(data)
    if text_key in columns_to_remove:
        columns_to_remove.remove(text_key)

    def generator():
        eos_token = tokenizer.eos_token or "<|endoftext|>"
        eos_token_id_local = eos_token_id or tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids(eos_token)
        assert eos_token_id_local is not None, "The tokenizer must have an eos token id."

        buffer = []
        for sample in data:
            # Remove unwanted columns from the sample
            sample = {key: sample[key] for key in sample if key not in columns_to_remove}

            text = sample[text_key]
            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens.append(eos_token_id_local)
            buffer.extend(tokens)

            # Slice the buffer into chunks of max_seq_len
            while len(buffer) >= max_seq_len:
                chunk = buffer[:max_seq_len]
                buffer = buffer[max_seq_len:]
                yield {'input_ids': torch.tensor(chunk)}

        # Process any remaining tokens in the buffer
        while len(buffer) >= max_seq_len:
            chunk = buffer[:max_seq_len]
            buffer = buffer[max_seq_len:]
            yield {'input_ids': torch.tensor(chunk)}

        # Yield the final chunk if any tokens are left
        if buffer:
            yield {'input_ids': torch.tensor(buffer)}

    return IterableDataset.from_generator(generator)


def get_columns_all_equal(dataset: Union[Dataset, DatasetDict]) -> list[str]:
    """Get a single list of columns in a `Dataset` or `DatasetDict`.

    We assert the columms are the same across splits if it's a `DatasetDict`.

    Args:
        dataset: The dataset to get the columns from.

    Returns:
        A list of columns.
    """
    if isinstance(dataset, DatasetDict):
        cols_by_split = dataset.column_names.values()
        columns = next(iter(cols_by_split))
        if not all(cols == columns for cols in cols_by_split):
            raise ValueError("All splits must have the same columns")

        return columns

    return dataset.column_names


class MemmapDataset(TorchDataset):
    """Torch Dataset backed by a memory-mapped numpy array."""
    def __init__(
        self,
        data_path: str,
        ctx_len: int,
        max_examples: int | None = None,
        dtype = np.uint16,
    ):
        mmap = np.memmap(data_path, dtype=dtype, mode="r").reshape(-1, ctx_len)
        self.mmap = mmap[:max_examples]

    def __len__(self):
        return len(self.mmap)

    def __getitem__(self, idx):
        return dict(
            input_ids=torch.from_numpy(self.mmap[idx].astype(np.int64))
        )
    
    def select(self, rng: range) -> "MemmapDataset":
        """Select a subset of the dataset."""
        mmap = MemmapDataset.__new__(MemmapDataset)
        mmap.mmap = self.mmap[rng.start:rng.stop]
        return mmap

    def shard(self, num_shards: int, shard_id: int) -> "MemmapDataset":
        """Split the dataset into `num_shards` and return the `shard_id`-th shard."""
        mmap = MemmapDataset.__new__(MemmapDataset)

        # Split the mmap array into `num_shards` and return the `shard_id`-th shard
        shards = np.array_split(self.mmap, num_shards)
        mmap.mmap = shards[shard_id]
        return mmap
