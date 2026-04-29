"""
Evaluation prompt template + response parser for v5.

Per Q1 sign-off (D-16): two separate API calls per chain — one baseline (no
constraint context), one intervention (with constraint context). This module
defines the prompt template that turns a ChainCandidate into the user-message
string sent to the subject model.

PER-CELL PROMPT CONTENT IS DELIBERATELY GENERIC HERE.
The cell-specific phrasing (constraint description, prediction question) requires
T-design context and lands in Phase B (joint authoring). Per-cell PromptBuilder
subclasses can override `format_constraint_context()` and `format_question()`
once T is designed.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from ..common.schema import ChainCandidate, GameEvent

logger = logging.getLogger(__name__)


@dataclass
class PromptPair:
    """The two prompts (baseline + intervention) for one chain."""
    chain_id: str
    cell: str
    baseline_prompt: str
    intervention_prompt: str
    metadata: dict


class PromptBuilder:
    """
    Builds baseline + intervention prompts for one cell.

    Subclasses override per-cell hooks:
      - format_constraint_context(chain) → str
      - format_question(chain) → str
      - format_event(event, idx) → str (optional)
    """

    def __init__(self, cell: str):
        self.cell = cell

    # --- Public entry point -------------------------------------------------
    def build(self, chain: ChainCandidate) -> PromptPair:
        if chain.cell != self.cell:
            raise ValueError(
                f"PromptBuilder for cell '{self.cell}' got chain for cell '{chain.cell}'"
            )

        chain_block = self.format_chain(chain)
        question = self.format_question(chain)
        constraint_block = self.format_constraint_context(chain)

        baseline_prompt = self._compose(
            chain_block=chain_block,
            constraint_block=None,
            question=question,
        )
        intervention_prompt = self._compose(
            chain_block=chain_block,
            constraint_block=constraint_block,
            question=question,
        )

        return PromptPair(
            chain_id=chain.chain_id,
            cell=self.cell,
            baseline_prompt=baseline_prompt,
            intervention_prompt=intervention_prompt,
            metadata={
                "n_events": len(chain.events),
                "chain_length": len(chain.events),
                "actor_count": len({e.actor for e in chain.events}),
            },
        )

    # --- Composition --------------------------------------------------------
    @staticmethod
    def _compose(chain_block: str, constraint_block: str | None, question: str) -> str:
        parts: list[str] = []
        if constraint_block:
            parts.append("## Constraint Context\n" + constraint_block.strip())
        parts.append("## Event Chain\n" + chain_block.strip())
        parts.append("## Question\n" + question.strip())
        return "\n\n".join(parts)

    # --- Per-cell hooks (override in subclasses) ----------------------------
    def format_chain(self, chain: ChainCandidate) -> str:
        """Default: render each event on its own line. Subclasses can prettify."""
        actor_map = _build_actor_map(chain.events)
        lines = []
        for i, ev in enumerate(chain.events):
            lines.append(self.format_event(ev, i, actor_map=actor_map))
        return "\n".join(lines)

    def format_event(self, event: GameEvent, idx: int, actor_map: dict | None = None) -> str:
        """
        Default: 'idx | t=TIME | TYPE by Player_N (context_summary)'.

        CF-4=B: actor is anonymised to 'Player_N' using the chain-local actor_map
        built in format_chain(). This prevents player/team identity leaking into
        prompts and ensures the model reasons from constraint rules alone.
        """
        anon = actor_map.get(event.actor, "Player_?") if actor_map else "Player_?"
        ctx_summary = self._summarize_context(event.location_context)
        return (
            f"{idx:>3}. t={event.timestamp:>7.1f}s | "
            f"{event.event_type:<22} | actor={anon:<12} "
            f"| {ctx_summary}"
        )

    def format_constraint_context(self, chain: ChainCandidate) -> str:
        """
        TBD per cell — pre-registered at T-design time (Phase B).

        Default placeholder explains why this is unset.
        """
        return (
            f"[CONSTRAINT CONTEXT FOR {self.cell.upper()} — TO BE DEFINED AT T-DESIGN]\n"
            "Per SPEC §6 (T design), the constraint context language is locked "
            "during the T-design joint authoring session. This default is a "
            "placeholder so prompt construction can be tested infrastructure-side."
        )

    def format_question(self, chain: ChainCandidate) -> str:
        """
        TBD per cell — pre-registered at T-design time (Phase B).
        """
        return (
            "[QUESTION TBD AT T-DESIGN]\n"
            "Predict the actor or event-type that follows. "
            "Reply with a single token only."
        )

    @staticmethod
    def _summarize_context(ctx: dict) -> str:
        if not ctx:
            return ""
        # Cap at 6 keys, 80 chars total to keep prompt compact
        items = list(ctx.items())[:6]
        s = ", ".join(f"{k}={_short(v)}" for k, v in items)
        return s[:120]


def _short(v) -> str:
    if isinstance(v, float):
        return f"{v:.2f}"
    s = str(v)
    return s if len(s) <= 32 else s[:29] + "..."


def _build_actor_map(events: list) -> dict[str, str]:
    """
    CF-4=B anonymisation: map each unique actor ID to a stable 'Player_N' slot.

    Order is first-appearance within the chain, ensuring the same actor always
    gets the same slot across all events in the chain. The mapping is chain-local
    so different chains do not share slot assignments.
    """
    seen: dict[str, str] = {}
    for ev in events:
        if ev.actor not in seen:
            seen[ev.actor] = f"Player_{len(seen)}"
    return seen


# ---------------------------------------------------------------------------
# Per-cell PromptBuilder stubs.
# These hold the per-cell hooks; cell-specific prompt content is filled in
# at T-design time (Phase B).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Per-cell PromptBuilders — constraint context locked in Phase B 2026-04-27.
# CF-1=A (natural language), CF-2=D (binary classify), CF-4=B (domain only).
# ---------------------------------------------------------------------------

_CLASSIFY_QUESTION = (
    "Is the sequence of events above consistent with the rules of {domain}?\n"
    "Reply with exactly one word: YES or NO."
)


# Constraint-context wording locked in T-design review (2026-04-28, both
# authors). Per that review, constraint blocks are stripped of domain-name
# anchors so the intervention condition tests "follow these rules" rather
# than "recall this game's Wikipedia page". The question template still
# names the domain — that decision was deferred (see review §1).


class FortnitePromptBuilder(PromptBuilder):
    """⚠ LEGACY — superseded by PUBGPromptBuilder per A2 (D-35). Kept as
    a test fixture; not in active PER_CELL_PROMPT_BUILDERS for evaluation."""

    def __init__(self):
        super().__init__(cell="fortnite")

    def format_constraint_context(self, chain: ChainCandidate) -> str:
        return (
            "In Fortnite, players must remain within the safe storm zone or take "
            "damage over time. An eliminated player cannot generate further actions. "
            "Building structures consumes exactly one material per piece."
        )

    def format_question(self, chain: ChainCandidate) -> str:
        return _CLASSIFY_QUESTION.format(domain="Fortnite")


class PUBGPromptBuilder(PromptBuilder):
    """Per A2 (D-35), replaces FortnitePromptBuilder in active evaluation.
    Wording locked in T-design review (2026-04-28)."""

    def __init__(self):
        super().__init__(cell="pubg")

    def format_constraint_context(self, chain: ChainCandidate) -> str:
        return (
            "Players must stay inside the shrinking safe zone; remaining outside "
            "deals damage that increases per phase. An eliminated player cannot "
            "act. Squads consist of up to 4 players who can revive downed teammates."
        )

    def format_question(self, chain: ChainCandidate) -> str:
        return _CLASSIFY_QUESTION.format(domain="PUBG")


class NBAPromptBuilder(PromptBuilder):
    """N-5 constraint context. Wording locked in T-design review (2026-04-28).

    A4 (2026-04-29): format_event overridden to surface terminal_action,
    actor_foul_count_after, and possession_elapsed_s — the variables the
    locked constraint context (24-sec shot clock, 6-foul ejection,
    possession-change rule) refers to.
    """

    def __init__(self):
        super().__init__(cell="nba")

    def format_constraint_context(self, chain: ChainCandidate) -> str:
        return (
            "Offensive team must shoot within 24 seconds of gaining possession. "
            "A player with 6 fouls is ejected. Possession changes on made shots, "
            "turnovers, defensive rebounds."
        )

    def format_question(self, chain: ChainCandidate) -> str:
        return _CLASSIFY_QUESTION.format(domain="NBA basketball")

    def format_event(self, event: GameEvent, idx: int, actor_map: dict | None = None) -> str:
        anon = actor_map.get(event.actor, "Player_?") if actor_map else "Player_?"
        ctx = event.location_context or {}
        # A4 fields: surface terminal action label, foul count, elapsed time.
        # Fall back to base behavior if the extractor hasn't populated them
        # (e.g. mock data path).
        terminal = ctx.get("terminal_action") or "?"
        foul = ctx.get("actor_foul_count_after")
        elapsed = ctx.get("possession_elapsed_s")
        period = ctx.get("period", "?")
        clock = ctx.get("clock_end") or ctx.get("clock_start") or ""
        score = f"{ctx.get('score_home','?')}-{ctx.get('score_away','?')}"

        parts = [f"action={terminal}"]
        if elapsed is not None:
            parts.append(f"possession_elapsed_s={elapsed}")
        if foul is not None:
            parts.append(f"actor_total_fouls={foul}")
        parts.append(f"period={period}")
        if clock:
            parts.append(f"clock={clock}")
        parts.append(f"score={score}")
        ctx_str = ", ".join(parts)

        return (
            f"{idx:>3}. t={event.timestamp:>7.1f}s | "
            f"{event.event_type:<22} | actor={anon:<12} "
            f"| {ctx_str}"
        )


class CSGOPromptBuilder(PromptBuilder):
    """C-5 constraint context. Wording locked in T-design review (2026-04-28).

    A5 (2026-04-29): format_event overridden to surface real per-event fields
    from FACEIT raw stats (action_label, team_id, round, map, score). Best-
    effort within FACEIT aggregate-stat data ceiling. If A5 fails to move
    signal, CS:GO graduates to v5.1 awpy demo extraction.
    """

    def __init__(self):
        super().__init__(cell="csgo")

    def format_constraint_context(self, chain: ChainCandidate) -> str:
        return (
            "T wins by detonating bomb or eliminating CTs. CT wins by defusing "
            "bomb, eliminating Ts, or time expiring. Eliminated players don't "
            "respawn until next round. Bomb plants only at sites A or B."
        )

    def format_question(self, chain: ChainCandidate) -> str:
        return _CLASSIFY_QUESTION.format(domain="Counter-Strike")

    def format_event(self, event: GameEvent, idx: int, actor_map: dict | None = None) -> str:
        anon = actor_map.get(event.actor, "Player_?") if actor_map else "Player_?"
        ctx = event.location_context or {}
        action_label = ctx.get("action_label") or "?"
        team_id = ctx.get("team_id") or ""
        # CF-4=B: anonymise team identity to "team_A"/"team_B". Use a stable
        # deterministic hash (md5 first byte) so the same team_id always
        # maps to the same slot across runs (Python's built-in hash() is
        # PYTHONHASHSEED-randomized per process).
        team_slot = self._team_slot(team_id, ctx)
        winner_id = ctx.get("winner_faction", "")
        # CF-4=B: also anonymise final_winner — raw team UUIDs are PII-equivalent
        # for the experiment. Map to team_A / team_B / "(undecided)".
        winner_slot = self._team_slot(winner_id, ctx) if winner_id else "(undecided)"
        round_num = ctx.get("round", "?")
        map_name = ctx.get("map", "?")
        score = ctx.get("match_score", "?")
        ctx_str = (
            f"action={action_label}, team={team_slot}, round={round_num}, "
            f"map={map_name}, final_score={score}, final_winner={winner_slot}"
        )
        return (
            f"{idx:>3}. t={event.timestamp:>7.1f}s | "
            f"{event.event_type:<22} | actor={anon:<12} "
            f"| {ctx_str}"
        )

    @staticmethod
    def _team_slot(team_id: str, ctx: dict) -> str:
        """Deterministic team slot assignment (CF-4=B). Same team_id always
        maps to the same A/B slot across runs (md5-stable, not Python hash())."""
        if not team_id:
            return "?"
        import hashlib
        h = hashlib.md5(team_id.encode("utf-8")).digest()[0]
        return "team_A" if (h & 1) == 0 else "team_B"


class RocketLeaguePromptBuilder(PromptBuilder):
    """R-5 constraint context. Wording locked in T-design review (2026-04-28).

    A6 (2026-04-29): format_event overridden to surface real per-event fields
    from BallChasing aggregate stats (action_label, team_color, score,
    duration). Best-effort within data ceiling. If A6 fails, RL graduates
    to v5.1 carball/rrrocket replay extraction.
    """

    def __init__(self):
        super().__init__(cell="rocket_league")

    def format_constraint_context(self, chain: ChainCandidate) -> str:
        return (
            "Teams of 3 score by hitting the ball into the opposing goal. Boost "
            "meter caps at 100; collected from pads. A goal resets ball and "
            "player positions for a kickoff."
        )

    def format_question(self, chain: ChainCandidate) -> str:
        return _CLASSIFY_QUESTION.format(domain="Rocket League")

    def format_event(self, event: GameEvent, idx: int, actor_map: dict | None = None) -> str:
        anon = actor_map.get(event.actor, "Player_?") if actor_map else "Player_?"
        ctx = event.location_context or {}
        action_label = ctx.get("action_label") or "?"
        team_color = ctx.get("team_color") or event.actor_team or "?"
        score_b = ctx.get("score_blue", "?")
        score_o = ctx.get("score_orange", "?")
        duration = ctx.get("duration_s")
        ctx_str = (
            f"action={action_label}, team={team_color}, "
            f"final_score=blue:{score_b}–orange:{score_o}"
        )
        if duration is not None:
            try:
                ctx_str += f", match_duration_s={float(duration):.0f}"
            except (TypeError, ValueError):
                pass
        return (
            f"{idx:>3}. t={event.timestamp:>7.1f}s | "
            f"{event.event_type:<22} | actor={anon:<12} "
            f"| {ctx_str}"
        )


class PokerPromptBuilder(PromptBuilder):
    """P-5 constraint context. Wording locked in T-design review (2026-04-28)
    — game-name anchor ('NLHE') stripped to test rule-following rather than
    pretrained domain recall."""

    def __init__(self):
        super().__init__(cell="poker")

    def format_constraint_context(self, chain: ChainCandidate) -> str:
        return (
            "Rules: fold/check/call/bet/raise up to stack. Cannot wager more "
            "than held. Action proceeds clockwise. Folded player can't act "
            "again. Best 5-card hand wins at showdown."
        )

    def format_question(self, chain: ChainCandidate) -> str:
        return _CLASSIFY_QUESTION.format(domain="poker")


PER_CELL_PROMPT_BUILDERS = {
    "pubg": PUBGPromptBuilder,
    "nba": NBAPromptBuilder,
    "csgo": CSGOPromptBuilder,
    "rocket_league": RocketLeaguePromptBuilder,
    "poker": PokerPromptBuilder,
    # FortnitePromptBuilder kept as legacy fixture (per A2/D-35); not in
    # active evaluation.
    "fortnite": FortnitePromptBuilder,
}


# ---------------------------------------------------------------------------
# Response parser — converts model text to a normalized prediction string.
# ---------------------------------------------------------------------------

# Common abstain phrases the model might emit
_ABSTAIN_PATTERNS = re.compile(
    r"^\s*(i\s*don'?t\s*know|abstain|n/?a|unknown|cannot\s*determine|insufficient|"
    r"no\s*answer|skip)\s*\.?\s*$",
    re.IGNORECASE,
)


def parse_model_response(raw: str, allowed_predictions: list[str] | None = None) -> str:
    """
    Parse a model response string into a normalized prediction.

    Returns:
      - "" if the response is empty / abstain (scoring layer treats this as -1)
      - First matched token if allowed_predictions is given and one matches
      - The first stripped non-empty line of the response otherwise

    Per Q1 sign-off (separate calls), parsing is shared between baseline and
    intervention responses.
    """
    if not raw or not raw.strip():
        return ""

    s = raw.strip()
    if _ABSTAIN_PATTERNS.match(s):
        return ""

    # If the model wrapped the answer in JSON, try to extract it
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                for k in ("answer", "prediction", "response", "value"):
                    if k in obj:
                        return str(obj[k]).strip().lower()
        except json.JSONDecodeError:
            pass

    # First line, lowercased, stripped of trailing punctuation
    first_line = s.splitlines()[0].strip().rstrip(".!?,").lower()

    if allowed_predictions:
        # Exact match only — substring matching invites answer-inverting bugs
        # (e.g. "No, the sequence is not consistent" vs "There's no doubt — yes,
        # this is consistent" both contain "no" before "yes"). The Haiku prompt
        # already says "Reply with exactly one word: YES or NO" so exact match
        # is the contract we've declared. Non-conforming responses become
        # abstains (score=-1) and are excluded from McNemar.
        for pred in allowed_predictions:
            if first_line == pred.lower():
                return pred.lower()
        logger.debug(f"Response '{first_line[:60]}' did not exact-match any allowed prediction; treating as abstain")
        return ""

    return first_line
