"""Per-attribute + full-card accuracy trackers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AccMeter:
    """Running count of correct per-attribute + full-card predictions."""

    attrs: tuple[str, ...] = ("number", "color", "shape", "shading")
    correct: dict[str, int] = field(default_factory=lambda: {})
    full_card: int = 0
    total: int = 0

    def __post_init__(self) -> None:
        for a in self.attrs:
            self.correct.setdefault(a, 0)

    def update(self, preds: dict[str, list[int]], labels: dict[str, list[int]]) -> None:
        n = len(next(iter(preds.values())))
        self.total += n
        all_right = [True] * n
        for a in self.attrs:
            for i, (p, y) in enumerate(zip(preds[a], labels[a])):
                if p == y:
                    self.correct[a] += 1
                else:
                    all_right[i] = False
        self.full_card += sum(all_right)

    def summary(self) -> dict[str, float]:
        if self.total == 0:
            return {f"acc/{a}": 0.0 for a in self.attrs} | {"acc/full_card": 0.0, "acc/mean_attr": 0.0}
        per = {a: self.correct[a] / self.total for a in self.attrs}
        out = {f"acc/{a}": v for a, v in per.items()}
        out["acc/full_card"] = self.full_card / self.total
        out["acc/mean_attr"] = sum(per.values()) / len(self.attrs)
        return out
