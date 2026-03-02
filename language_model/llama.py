from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Union
import json
import math
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def llama3format(prompt: str) -> str:
    """
    Llama-3 instruct-style chat wrapper.
    Use ONLY with Llama-3 instruct checkpoints that support these special tokens.
    """
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


@dataclass
class LlamaHyperparameters:
    # Model / tokenizer
    base: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer_reponame: Optional[str] = None
    format: Optional[Callable[[str], str]] = None

    # Training
    epochs: int = 0
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0

    # Sequence / generation
    max_sequence_length: int = 512
    gen_batch_size: int = 1
    num_beams: int = 1
    repetition_penalty: float = 1.0
    max_output_length: int = 32

    # Optional flags referenced by configs (kept for compatibility)
    dynamic_tokenization: bool = False
    protected_input_length: int = 0
    warmup_steps: int = 0
    optimizer: str = "adamw"
    param_magnitude: Optional[str] = None

    # Quantization/LoRA knobs (not implemented in this minimal backend)
    quantize: Optional[str] = None
    lora: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: float = 0.0


class Llama:
    """
    Thin HF backend for dextrous/tracker.py.

    Required methods:
      - generate(prompts) -> List[str]
      - training(prompts, values, yield_every_x_epochs) -> Iterator[float]
      - perplexity(prompts, values) -> float
      - save(path) -> None
    """

    def __init__(self, **kwargs):
        hp = LlamaHyperparameters(**kwargs)
        self.hp = hp

        model_name = hp.base
        tok_name = hp.tokenizer_reponame or model_name

        self.tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
        # Decoder-only generation is easiest with left padding
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            # Many causal LMs do not define a pad token
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.model.eval()

        self.formatter: Callable[[str], str] = hp.format if hp.format is not None else (lambda x: x)

        self._debug = os.environ.get("D0T_DEBUG_DUMPS", "0") == "1"

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "hf_model").mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path / "hf_model")
        self.tokenizer.save_pretrained(path / "hf_model")

    @torch.no_grad()
    def generate(self, prompts: List[str]) -> List[str]:
        """
        Deterministic generation used by tracker.predict().
        We decode only the newly generated portion.
        """
        self.model.eval()
        results: List[str] = []

        # Safe (but slower) per-sample generation avoids tricky per-row slicing with left padding
        for p in prompts:
            p_fmt = self.formatter(p)
            enc = self.tokenizer(
                [p_fmt],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.hp.max_sequence_length,
            )
            enc = {k: v.to(self.model.device) for k, v in enc.items()}

            out = self.model.generate(
                **enc,
                max_new_tokens=self.hp.max_output_length,
                num_beams=self.hp.num_beams,
                repetition_penalty=self.hp.repetition_penalty,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            # Slice off the prompt tokens (single-example, so prompt length is enc["input_ids"].shape[1])
            prompt_len = int(enc["input_ids"].shape[1])
            gen_ids = out[0][prompt_len:]
            txt = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            results.append(txt)

        if self._debug:
            self._dump_debug(prompts, results)

        return results

    def perplexity(self, prompts: List[str], values: List[str]) -> float:
        """
        Approx perplexity with *prompt-masked* loss, matching instruction tuning semantics:
          - loss computed only on the value tokens, not on the prompt tokens
        """
        self.model.eval()
        total_nll = 0.0
        total_tokens = 0

        for p, v in zip(prompts, values):
            p_fmt = self.formatter(p)
            v = v or ""

            # Build full text and separately compute prompt length
            full_text = p_fmt + "\n" + v

            enc_prompt = self.tokenizer(
                p_fmt,
                return_tensors="pt",
                truncation=True,
                max_length=self.hp.max_sequence_length,
            )
            enc_full = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.hp.max_sequence_length,
            )

            input_ids = enc_full["input_ids"].to(self.model.device)
            attn = enc_full["attention_mask"].to(self.model.device)

            labels = input_ids.clone()
            prompt_len = int(enc_prompt["input_ids"].shape[1])
            labels[:, :prompt_len] = -100

            with torch.no_grad():
                out = self.model(input_ids=input_ids, attention_mask=attn, labels=labels)

            loss = float(out.loss.detach().float().item())

            # Count only non-masked tokens for weighting
            n_tokens = int((labels != -100).sum().item())
            if n_tokens > 0:
                total_nll += loss * n_tokens
                total_tokens += n_tokens

        if total_tokens == 0:
            return float("inf")

        avg_nll = total_nll / total_tokens
        return float(math.exp(avg_nll))

    def training(
        self,
        prompts: List[str],
        values: List[str],
        yield_every_x_epochs: float = 1.0,
    ) -> Iterator[float]:
        """
        Minimal trainer:
          - instruction-style loss masking (prompt masked)
          - AdamW
        For smoke tests, set epochs=0 and it will just yield a quick ppl estimate.
        """
        if self.hp.epochs <= 0:
            # quick check to keep pipeline moving
            yield self.perplexity(prompts[:10], values[:10])
            return

        self.model.train()

        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.hp.learning_rate),
            weight_decay=float(self.hp.weight_decay),
        )

        bs = max(1, int(self.hp.train_batch_size))
        ga = max(1, int(self.hp.gradient_accumulation_steps))

        def iter_batches():
            for i in range(0, len(prompts), bs):
                yield prompts[i : i + bs], values[i : i + bs]

        steps = 0
        last_yield = 0.0

        for epoch in range(int(self.hp.epochs)):
            opt.zero_grad(set_to_none=True)

            for bp, bv in iter_batches():
                # Build batch by computing prompt+value full sequences and masking prompt tokens
                input_ids_list = []
                attn_list = []
                labels_list = []

                for p, v in zip(bp, bv):
                    p_fmt = self.formatter(p)
                    v = v or ""
                    full_text = p_fmt + "\n" + v

                    enc_prompt = self.tokenizer(
                        p_fmt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.hp.max_sequence_length,
                    )
                    enc_full = self.tokenizer(
                        full_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.hp.max_sequence_length,
                    )

                    input_ids = enc_full["input_ids"][0]
                    attn = enc_full["attention_mask"][0]
                    labels = input_ids.clone()

                    prompt_len = int(enc_prompt["input_ids"].shape[1])
                    labels[:prompt_len] = -100

                    input_ids_list.append(input_ids)
                    attn_list.append(attn)
                    labels_list.append(labels)

                # Pad batch (left padding for decoder-only)
                # We pad with pad_token_id and set attention_mask accordingly.
                pad_id = int(self.tokenizer.pad_token_id)

                max_len = max(x.size(0) for x in input_ids_list)
                def left_pad(x: torch.Tensor, pad_value: int) -> torch.Tensor:
                    if x.size(0) == max_len:
                        return x
                    pad = torch.full((max_len - x.size(0),), pad_value, dtype=x.dtype)
                    return torch.cat([pad, x], dim=0)

                input_ids = torch.stack([left_pad(x, pad_id) for x in input_ids_list], dim=0).to(self.model.device)
                attn = torch.stack([left_pad(x, 0) for x in attn_list], dim=0).to(self.model.device)
                labels = torch.stack([left_pad(x, -100) for x in labels_list], dim=0).to(self.model.device)

                out = self.model(input_ids=input_ids, attention_mask=attn, labels=labels)
                loss = out.loss / ga
                loss.backward()

                steps += 1
                if steps % ga == 0:
                    opt.step()
                    opt.zero_grad(set_to_none=True)

            # Yield perplexity at epoch boundaries (good enough for this repo’s loop)
            self.model.eval()
            yield self.perplexity(prompts[:10], values[:10])
            self.model.train()

        self.model.eval()

    def _dump_debug(self, prompts: List[str], outputs: List[str]) -> None:
        """
        Writes prompts and outputs to a timestamped JSONL for quick diffing.
        """
        ts = int(time.time())
        out_dir = Path("debug_dumps")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"llama_generate_{ts}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for p, o in list(zip(prompts, outputs))[:50]:
                f.write(json.dumps({"prompt": p, "formatted": self.formatter(p), "output": o}, ensure_ascii=False) + "\n")