from typing import List, Tuple
import math

import torch
import random

class DataLoader:
    def __init__(self, filepath: str, tokenizer, batch_size: int, ctx_size: int):
        with open(filepath, "r") as f:
            raw_docs = f.read().split("<|endoftext|>")

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.ctx_size = ctx_size
        self.block_i = 0

        # Tokenize once; pack into a token stream each epoch (after shuffling doc order).
        self.eos_tokens: List[int] = self.tokenizer.encode("<|endoftext|>", allowed_special="all")
        if len(self.eos_tokens) < 1:
            raise ValueError("Tokenizer returned empty encoding for <|endoftext|>.")
        self.docs_tokens: List[List[int]] = []
        for doc in raw_docs:
            doc = doc.strip()
            if not doc:
                continue
            self.docs_tokens.append(self.tokenizer.encode(doc))

        self.stream: torch.Tensor = torch.empty(0, dtype=torch.long)
        self.num_blocks: int = 0
        self._rebuild_stream()

    def _rebuild_stream(self) -> None:
        stream_tokens: List[int] = []
        eos = self.eos_tokens[0]
        for doc_tokens in self.docs_tokens:
            if len(doc_tokens) == 0:
                continue
            stream_tokens.extend(doc_tokens)
            stream_tokens.append(eos)

        self.stream = torch.tensor(stream_tokens, dtype=torch.long)
        block_len = self.ctx_size + 1
        self.num_blocks = int(self.stream.numel() // block_len)

    def __len__(self) -> int:
        if self.batch_size <= 0:
            return 0
        return int(math.ceil(self.num_blocks / self.batch_size))

    def __iter__(self):
        self.block_i = 0
        random.shuffle(self.docs_tokens)
        self._rebuild_stream()
        return self
            
    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.block_i >= self.num_blocks:
            raise StopIteration

        block_len = self.ctx_size + 1
        remaining = self.num_blocks - self.block_i
        cur_bs = min(self.batch_size, remaining)

        tokens = torch.empty((cur_bs, block_len), dtype=torch.long)
        for b in range(cur_bs):
            start = (self.block_i + b) * block_len
            tokens[b] = self.stream[start : start + block_len]

        x, y = tokens[:, :-1], tokens[:, 1:]
        self.block_i += cur_bs
        return x, y
