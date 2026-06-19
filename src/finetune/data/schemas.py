from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrainingExample:
    id: str
    prompt: str = ""
    completion: str = ""
    text: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    source_lang: str | None = None
    target_lang: str | None = None
    group_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def training_text(self) -> str:
        if self.text:
            return self.text
        if self.messages:
            return "\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in self.messages)
        if self.prompt or self.completion:
            return f"{self.prompt}\n{self.completion}".strip()
        return ""

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "completion": self.completion,
            "text": self.text,
            "messages": self.messages,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "group_id": self.group_id,
            "metadata": self.metadata,
        }
