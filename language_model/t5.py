from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Union
import json
import math
import os
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@dataclass
class T5Hyperparameters:
    base: str = "t5-small"
    tokenizer_reponame: Optional[str] = None

    epochs: int = 0
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0

    max_sequence_length: int = 512
    gen_batch_size: int = 1
    num_beams: int = 1
    max_output_length: int = 32


class T5:
    """
    Thin HF backend for dextrous/tracker.py (seq2seq).
    """

    def __init__(self, **kwargs):
        hp = T5Hyperparameters(**kwargs)
        self.hp = hp

        model_name = hp.base
        tok_name = hp.tokenizer_reponame or model_name

        self.tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.model.eval()

        self._debug = os.environ.get("D0T_DEBUG_DUMPS", "0") == "1"

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "hf_model").mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path / "hf_model")
        self.tokenizer.save_pretrained(path / "hf_model")

    @torch.no_grad()
    def generate(self, prompts: List[str]) -> List[str]:
        self.model.eval()

        # Batch generation is fine for T5
        enc = self.tokenizer(
            prompts,
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
            do_sample=False,
        )

        texts = [self.tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in out]

        if self._debug:
            self._dump_debug(prompts, texts)

        return texts

    def perplexity(self, prompts: List[str], values: List[str]) -> float:
        """
        Approx perplexity for seq2seq: compute NLL on target tokens.
        IMPORTANT: mask pad tokens in labels to -100.
        """
        self.model.eval()
        total_nll = 0.0
        total_tokens = 0

        for p, v in zip(prompts, values):
            v = v or ""

            enc_in = self.tokenizer(
                p,
                return_tensors="pt",
                truncation=True,
                max_length=self.hp.max_sequence_length,
            )
            enc_in = {k: t.to(self.model.device) for k, t in enc_in.items()}

            enc_out = self.tokenizer(
                v,
                return_tensors="pt",
                truncation=True,
                max_length=self.hp.max_output_length,
            )
            labels = enc_out["input_ids"].to(self.model.device)
            labels[labels == self.tokenizer.pad_token_id] = -100

            with torch.no_grad():
                out = self.model(**enc_in, labels=labels)

            loss = float(out.loss.detach().float().item())
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
        Minimal trainer with correct label masking.
        For smoke tests, set epochs=0 (yields quick ppl only).
        """
        if self.hp.epochs <= 0:
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

        steps = 0
        for epoch in range(int(self.hp.epochs)):
            opt.zero_grad(set_to_none=True)

            for i in range(0, len(prompts), bs):
                bp = prompts[i : i + bs]
                bv = [x or "" for x in values[i : i + bs]]

                enc_in = self.tokenizer(
                    bp,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.hp.max_sequence_length,
                )
                enc_in = {k: t.to(self.model.device) for k, t in enc_in.items()}

                enc_out = self.tokenizer(
                    bv,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.hp.max_output_length,
                )
                labels = enc_out["input_ids"].to(self.model.device)
                labels[labels == self.tokenizer.pad_token_id] = -100

                out = self.model(**enc_in, labels=labels)
                loss = out.loss / ga
                loss.backward()

                steps += 1
                if steps % ga == 0:
                    opt.step()
                    opt.zero_grad(set_to_none=True)

            self.model.eval()
            yield self.perplexity(prompts[:10], values[:10])
            self.model.train()

        self.model.eval()

    def _dump_debug(self, prompts: List[str], outputs: List[str]) -> None:
        ts = int(time.time())
        out_dir = Path("debug_dumps")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"t5_generate_{ts}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for p, o in list(zip(prompts, outputs))[:50]:
                f.write(json.dumps({"prompt": p, "output": o}, ensure_ascii=False) + "\n")