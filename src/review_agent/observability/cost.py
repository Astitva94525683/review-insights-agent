"""Token usage and cost per request, from the usage numbers each provider returns."""
from dataclasses import dataclass


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, usage_metadata: dict | None) -> None:
        if usage_metadata:
            self.input_tokens += int(usage_metadata.get("input_tokens") or 0)
            self.output_tokens += int(usage_metadata.get("output_tokens") or 0)

    def cost_usd(self, input_per_mtok: float, output_per_mtok: float) -> float:
        return round((self.input_tokens * input_per_mtok + self.output_tokens * output_per_mtok) / 1e6, 6)
