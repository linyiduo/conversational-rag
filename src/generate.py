from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.schema import Passage


def format_context(passages: list[Passage], max_chars: int = 3500) -> str:
    parts: list[str] = []
    used = 0
    for i, p in enumerate(passages, 1):
        block = f"[{i}] {p.title}\n{p.text}".strip() if p.title else f"[{i}] {p.text}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def build_prompt(question: str, history: str, context: str) -> str:
    hist_block = f"Dialogue history:\n{history}\n\n" if history else ""
    return (
        "You are a helpful assistant. Answer the user's question using only the given context. "
        "If the context is insufficient, say you do not know.\n\n"
        f"{hist_block}"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


class Generator:
    def __init__(self, model_path: str, device: str = "cuda", dtype: str = "float16"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device != "cuda":
            self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.inference_mode()
    def generate(
        self,
        question: str,
        history: str,
        passages: list[Passage],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> str:
        context = format_context(passages)
        prompt = build_prompt(question, history, context)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        output = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = output[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # Keep first paragraph for base models that continue rambling.
        return text.split("\n\n")[0].strip()
