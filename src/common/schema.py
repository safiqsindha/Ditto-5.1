"""
Normalized event schema shared across all five v5 cells.

The GameEvent dataclass is the canonical input format for the translation
layer (T). All per-domain pipelines must produce streams of GameEvents.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

VALID_CELLS = frozenset(["fortnite", "nba", "csgo", "rocket_league", "poker"])


@dataclass
class GameEvent:
    """
    Normalized game event — the atomic unit consumed by T.

    Fields
    ------
    timestamp      : seconds elapsed from game/match start (float, monotonic)
    event_type     : normalized event type string; phase_ prefix already stripped
    actor          : player or team identifier (string, domain-consistent)
    location_context: domain-specific spatial or situational context
    raw_data_blob  : full unmodified source record for traceability
    cell           : domain identifier (one of VALID_CELLS)
    game_id        : globally unique match/game/replay identifier
    sequence_idx   : 0-based ordinal position within the game's event stream
    """
    timestamp: float
    event_type: str
    actor: str
    location_context: dict
    raw_data_blob: dict
    cell: str
    game_id: str
    sequence_idx: int
    # Optional enrichment fields; pipelines may populate these
    actor_team: str | None = None
    phase: str | None = None   # e.g. "regular_season", "playoffs", "round_1"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.cell not in VALID_CELLS:
            raise ValueError(f"Unknown cell '{self.cell}'; expected one of {VALID_CELLS}")
        if self.sequence_idx < 0:
            raise ValueError("sequence_idx must be >= 0")
        # Strip phase_ prefix from event_type (v1.1 amendment)
        if self.event_type.startswith("phase_"):
            self.event_type = self.event_type[len("phase_"):]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> GameEvent:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, s: str) -> GameEvent:
        return cls.from_dict(json.loads(s))


@dataclass
class EventStream:
    """
    Ordered sequence of GameEvents for a single game/match.
    """
    game_id: str
    cell: str
    events: list[GameEvent] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.cell not in VALID_CELLS:
            raise ValueError(f"Unknown cell '{self.cell}'; expected one of {VALID_CELLS}")

    def append(self, event: GameEvent) -> None:
        self.events.append(event)

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[GameEvent]:
        return iter(self.events)

    def to_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps({"game_id": self.game_id, "cell": self.cell,
                                "metadata": self.metadata, "_type": "header"}) + "\n")
            for ev in self.events:
                f.write(ev.to_json() + "\n")

    @classmethod
    def from_jsonl(cls, path: Path) -> EventStream:
        """
        Load EventStream from JSONL. F6 fix: validate header marker.
        First line must have '_type': 'header' for forward-compatibility.
        """
        with open(path) as f:
            lines = f.readlines()
        if not lines:
            raise ValueError(f"Empty JSONL file: {path}")
        try:
            header = json.loads(lines[0])
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed header in {path}: {e}") from e
        if header.get("_type") != "header":
            raise ValueError(
                f"Missing or invalid header marker in {path}; expected '_type': 'header'. "
                "File may be from an older version or corrupted."
            )
        stream = cls(game_id=header["game_id"], cell=header["cell"],
                     metadata=header.get("metadata", {}))
        for line in lines[1:]:
            stream.append(GameEvent.from_json(line.strip()))
        return stream


@dataclass
class ChainCandidate:
    """
    Output of T: a candidate constraint chain extracted from an EventStream.
    T is not implemented today — this type is defined for interface completeness.
    """
    chain_id: str
    game_id: str
    cell: str
    events: list[GameEvent]
    chain_metadata: dict = field(default_factory=dict)
    # Populated by harness after evaluation
    is_actionable: bool | None = None
    model_response: str | None = None
    scored_correct: bool | None = None

    def __len__(self) -> int:
        return len(self.events)
