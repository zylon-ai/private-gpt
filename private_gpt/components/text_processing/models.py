from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class Action(Enum):
    PASS = auto()
    DROP = auto()
    UNWRAP = auto()
    REPLACE = auto()
    HOLD = auto()


class ProbeStatus(Enum):
    NO_MATCH = auto()
    MATCH = auto()
    NEED_MORE = auto()


@dataclass(frozen=True)
class ProbeResult:
    status: ProbeStatus
    consumed: int = 0
    action: Action = Action.PASS
    replacement: str | None = None
    metadata: tuple[Any, ...] = ()
    state_updates: dict[str, Any] = field(default_factory=dict)
    state_deletes: tuple[str, ...] = ()

    @classmethod
    def no_match(cls) -> ProbeResult:
        return cls(status=ProbeStatus.NO_MATCH)

    @classmethod
    def need_more(cls) -> ProbeResult:
        return cls(status=ProbeStatus.NEED_MORE, action=Action.HOLD)


@dataclass
class ProcessingContext:
    final: bool = False
    state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProcessResult:
    text: str
    metadata: tuple[Any, ...]
    pending: str
    consumed: int


@dataclass(frozen=True)
class ProcessDelta:
    text: str
    metadata: tuple[Any, ...]
    pending: str
