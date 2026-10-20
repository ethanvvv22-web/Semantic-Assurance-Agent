"""Optional LLM adapter (kept thin and off the default path).

The Semantic Assurance Agent can use an LLM to (a) generate richer paraphrase
variants for the forward chain and (b) back-translate an action into a
reconstructed intent for the backward chain, instead of the template
verbalizers. It is OPTIONAL: enabled only if `anthropic` is installed and
ANTHROPIC_API_KEY is set; otherwise the deterministic template path is used,
so everything runs offline.
"""
from __future__ import annotations
import os

MODEL = "claude-opus-4-8"


class LLMAdapter:
    def __init__(self) -> None:
        self.client = None
        try:
            import anthropic  # noqa
            if os.environ.get("ANTHROPIC_API_KEY"):
                self.client = anthropic.Anthropic()
        except Exception:
            self.client = None

    @property
    def available(self) -> bool:
        return self.client is not None

    def back_translate(self, action_nl: str) -> str | None:
        """Reconstruct the operator intent an action implies. None if offline."""
        if not self.client:
            return None
        msg = self.client.messages.create(
            model=MODEL, max_tokens=256,
            messages=[{"role": "user", "content":
                       "In one sentence, state the operator intent this network "
                       f"action implements:\n{action_nl}"}],
        )
        return "".join(b.text for b in msg.content if b.type == "text").strip()
