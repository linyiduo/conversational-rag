from __future__ import annotations

from src.schema import Example

QUERY_MODES = ("question", "history", "rewrite")


def build_query(example: Example, mode: str = "question", max_history_turns: int = 6) -> str:
    mode = mode.lower()
    if mode == "question":
        return example.question
    if mode == "history":
        hist = example.history_text(max_turns=max_history_turns)
        if hist:
            return f"{hist}\nUser: {example.question}"
        return example.question
    if mode == "rewrite":
        # Baseline uses gold rewrite when available; otherwise falls back to history concat.
        if example.rewrite:
            return example.rewrite
        return build_query(example, mode="history", max_history_turns=max_history_turns)
    raise ValueError(f"Unknown query mode: {mode}. Choose from {QUERY_MODES}")
