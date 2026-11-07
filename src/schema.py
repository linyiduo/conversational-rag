from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Passage:
    pid: str
    text: str
    title: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Example:
    """One conversational turn to answer."""

    example_id: str
    question: str
    answer: str
    history: list[tuple[str, str]]  # (role, text), role in {user, agent}
    gold_passage_ids: list[str] = field(default_factory=list)
    rewrite: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def history_text(self, max_turns: int | None = None) -> str:
        turns = self.history if max_turns is None else self.history[-max_turns:]
        lines = []
        for role, text in turns:
            prefix = "User" if role == "user" else "Agent"
            lines.append(f"{prefix}: {text}")
        return "\n".join(lines)
