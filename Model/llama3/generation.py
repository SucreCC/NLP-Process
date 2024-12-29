import glob
from typing import List, Tuple, Optional, TypedDict

import fire
import numpy as np
import torch
import os
import time
from pathlib import Path
import json

from Model.llama3.model import Transformer, KVCache
from Model.llama3.tokenizer import Tokenizer
from config import ModelArgs


class Llama:

    @staticmethod
    def build(
            ckpt_dir: str,
            tokenizer_path: str,
            max_seq_len: int,
            max_batch_size: int,
            flash: bool = False,
            model_parallel_size: Optional[int] = 1,
            seed: int = 1,
    ) -> "Llama":
        assert 1 <= max_seq_len <= 8192, f"max_seq_len must be between 1 and 8192, got {max_seq_len}."
        assert os.path.isdir(ckpt_dir), f"Checkpoint directory '{ckpt_dir}' does not exist."
        assert os.path.isfile(tokenizer_path), f"Tokenizer file '{tokenizer_path}' does not exist."


        torch.manual_seed(seed)  # seed must be the same in all processes

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(checkpoints)
        ckpt_path = checkpoints[0]
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            flash=flash,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        assert model_args.vocab_size == tokenizer.n_words

        if torch.cuda.is_available():
            local_rank = 0
            torch.cuda.set_device(local_rank)
            if torch.cuda.is_bf16_supported():
                torch.set_default_dtype(torch.bfloat16)
                torch.set_default_device("cuda")
                print("Using GPU with BFloat16 precision")
            else:
                torch.set_default_dtype(torch.float16)
                torch.set_default_device("cuda")
                print("Using GPU with Half precision (float16)")
        else:
            torch.set_default_dtype(torch.float32)
            torch.set_default_device("cpu")
            print("Using CPU with default Float precision")

        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")
        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
            self,
            prompt_tokens: List[List[int]],
            sample_rng: torch.Generator,
            max_gen_len: int,
            temperature: float = 0.6,
            top_p: float = 0.9,
            echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
        max_gen_len (int): Maximum length of the generated text sequence.
        temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
        top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
        logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
        echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        # install KV cache in all the Attention layers
        for block in self.model.layers:
            layer_dtype = block.attention.wq.weight.dtype
            layer_device = block.attention.wq.weight.device
            block.attention.cache = KVCache(
                batch_size=bsz,
                seq_length=total_len,
                n_kv_heads=params.n_kv_heads,
                head_dim=params.dim // params.n_heads,
                dtype=layer_dtype,
                device=layer_device,
            )

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=device)
        input_text_mask = tokens != pad_id

        if min_prompt_len == total_len:
            logits = self.model.forward_inference(tokens, prev_pos)

        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens))

        for cur_pos in range(min_prompt_len, total_len):
            # get the logits for the next token in all the batch rows
            logits = self.model.forward_inference(tokens[:, prev_pos:cur_pos], prev_pos)
            # sample the next token
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p, sample_rng)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                torch.isin(next_token, stop_tokens)
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        out_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start: len(prompt_tokens[i]) + max_gen_len]
            # cut to after eos tok if any
            for stop_token in self.tokenizer.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                except ValueError:
                    pass
            out_tokens.append(toks)

        # clean up the KV cache in all the layers
        for block in self.model.layers:
            block.attention.cache = None

        return out_tokens

    def text_completion(
            self,
            prompts: List[str],
            sample_rng: torch.Generator,
            temperature: float = 0.6,
            top_p: float = 0.9,
            max_gen_len: Optional[int] = None,
            echo: bool = False,
    ):
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        # encode the (string) prompts to tokens
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        # generate the completions in tokens space
        generation_tokens = self.generate(
            prompt_tokens=prompt_tokens,
            sample_rng=sample_rng,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            echo=echo,
        )
        # decode the completions back to strings
        completions = [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]
        return completions


def sample_top_p(probs, p, generator):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1, generator=generator)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


# -----------------------------------------------------------------------------
# distributed and sharded data loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240801:
        print("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    assert header[1] == 7, "unsupported version"
    ntok = header[2]  # number of tokens (claimed)
    return ntok  # for now just return the number of tokens


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240801, "magic number mismatch in the data .bin file"
        assert header[1] == 7, "unsupported version"
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint32)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


class DistributedShardedDataLoader:
    """
    This DataLoader is both:
    - distributed (works correctly in case of multiple processes in DDP)
    - sharded (supports datasets that are broken up into multiple data shards)
    It is not *permuted*, meaning that it itearates over the data in the order
    of the dataset on disk, so the user should make sure to shuffle their examples
    during the creation of their data shards for best performance.
    """

    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        # kick things off
        self.current_shard = None
        self.reset()

    def reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        buf = torch.tensor(buf, dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the start pointer in current shard
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds advance the shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x, y


# -----------------------------------------------------------------------------
# int main

def main(
        ckpt_dir: str = "../../others/pretrain-model/llama/Llama3.1-8B",
        tokenizer_path: str = "../../others/pretrain-model/llama/Llama3.1-8B/tokenizer.model",
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_seq_len: int = 256,
        max_gen_len: int = 256,
        max_batch_size: int = 8,
        flash: bool = True,
):
    # load the val data shard
    data_loader = DistributedShardedDataLoader(
        filename_pattern="tinystories/*_val.bin",
        B=max_batch_size,
        T=max_seq_len,
        process_rank=0,
        num_processes=1,
    )

    llama = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        flash=flash,
    )

    total_batch_size = max_batch_size * max_seq_len
    print(f"total_batch_size: {total_batch_size}")

    # super simple training loop to start
    model = llama.model
    model.train()
    optimizer = model.configure_optimizers(learning_rate=1e-5, weight_decay=0.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for step in range(20):
        optimizer.zero_grad()
        x, y = data_loader.next_batch()
        x, y = x.to(device), y.to(device)
        loss = model.forward_loss(x, y)
        loss.backward()
        optimizer.step()
        print(f"step {step}, loss: {loss.item()}")

    # and now generate
    model.eval()
    prompts: List[str] = [
        "Once upon a time",
        "One day",
        "Lily and George were best friends",
        "On a dark and stormy night",
    ]

    sample_rng = torch.Generator(device='cuda')
    sample_rng.manual_seed(1337)
    t0 = time.time()
    results = llama.text_completion(
        prompts,
        sample_rng=sample_rng,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    t1 = time.time()
    print(f"Generated in {t1 - t0:.2f} seconds")
    for prompt, result in zip(prompts, results):
        print(prompt, end="")  # AK: change end="\n" to end=""
        print(f"{result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
