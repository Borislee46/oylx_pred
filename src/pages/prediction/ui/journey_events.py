from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Act = Literal["read", "match", "judge", "deliver"]
Verdict = Literal["neutral", "watch", "shortfall", "strength"]

ACTS: tuple[Act, ...] = ("read", "match", "judge", "deliver")
ACT_LABELS: dict[Act, str] = {
    "read": "读懂你",
    "match": "对照历史",
    "judge": "校准判断",
    "deliver": "给出区间",
}


@dataclass(frozen=True)
class JourneyEvent:
    act: Act
    title: str
    lines: tuple[str, ...] = ()
    verdict: Verdict | None = None
    metrics: tuple[tuple[str, str], ...] = ()
    key: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.lines, list):
            object.__setattr__(self, "lines", tuple(self.lines))
        if isinstance(self.metrics, list):
            object.__setattr__(self, "metrics", tuple(tuple(x) for x in self.metrics))
        if not self.key:
            object.__setattr__(self, "key", f"{self.act}:{self.title}")


def append_event(events: list[JourneyEvent], event: JourneyEvent) -> list[JourneyEvent]:
    out = [e for e in events if e.key != event.key]
    out.append(event)
    return out


def events_to_props(events: list[JourneyEvent], *, active_act: Act) -> dict[str, Any]:
    return {
        "active_act": active_act,
        "acts": [{"id": a, "label": ACT_LABELS[a]} for a in ACTS],
        "events": [
            {
                "act": e.act,
                "title": e.title,
                "lines": list(e.lines),
                "verdict": e.verdict,
                "metrics": [{"label": k, "value": v} for k, v in e.metrics],
                "key": e.key,
            }
            for e in events
        ],
    }
