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
    "Answer with exactly one token:\n"
    "YES\n"
    "NO\n"
    "If you output anything else, the answer is invalid."
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
    Wording locked in T-design review (2026-04-28).

    2026-04-29: format_event override added so elimination markers planted
    by the violation injector survive the default 6-key context cap. The
    override surfaces `eliminates_player` and `already_eliminated_player`
    explicitly when present so violation-detection diagnostics work on
    PUBG chains (the original chain renderer was dropping these markers).
    """

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

    def format_event(self, event: GameEvent, idx: int, actor_map: dict | None = None) -> str:
        anon = actor_map.get(event.actor, "Player_?") if actor_map else "Player_?"
        ctx = event.location_context or {}

        # Always-surface keys for PUBG: zone state + violation markers.
        # If violation markers are present, render them explicitly so the
        # 6-key context cap doesn't drop them.
        prefix_parts: list[str] = []
        eliminates = ctx.get("eliminates_player")
        if eliminates:
            anon_target = (
                actor_map.get(eliminates, eliminates) if actor_map else eliminates
            )
            prefix_parts.append(f"ELIMINATES={anon_target}")
        already_elim = ctx.get("already_eliminated_player")
        if already_elim:
            anon_already = (
                actor_map.get(already_elim, already_elim)
                if actor_map else already_elim
            )
            prefix_parts.append(f"NOTE={anon_already}_already_eliminated")
        zone_radius = ctx.get("safety_zone_radius")
        if zone_radius is not None:
            prefix_parts.append(f"zone_radius={zone_radius}")

        # Drop violation marker keys from the rest of the summary so they
        # don't appear twice + don't crowd out other zone/weapon fields.
        remaining = {
            k: v for k, v in ctx.items()
            if k not in {"eliminates_player", "already_eliminated_player",
                         "eliminated_at_event", "safety_zone_radius"}
        }
        ctx_summary = self._summarize_context(remaining)
        prefix_str = " ".join(prefix_parts)
        if prefix_str and ctx_summary:
            ctx_str = f"{prefix_str}, {ctx_summary}"
        else:
            ctx_str = prefix_str or ctx_summary

        return (
            f"{idx:>3}. t={event.timestamp:>7.1f}s | "
            f"{event.event_type:<22} | actor={anon:<12} "
            f"| {ctx_str}"
        )


class NBAPromptBuilder(PromptBuilder):
    """N-5 constraint context. Wording locked in T-design review (2026-04-28).

    A4 (2026-04-29): format_event overridden to surface terminal_action,
    actor_foul_count_after, and time_in_possession_s — the variables the
    locked constraint context (24-sec shot clock, 6-foul ejection,
    possession-change rule) refers to.

    2026-04-29 (D-44 prep): renamed `possession_elapsed_s` to
    `time_in_possession_s`. The old name was ambiguous and contributed to
    the single NBA FP under strict grounding — CoT showed the model misread
    elapsed=14.8 as a shot-clock violation. The new name unambiguously
    indicates "seconds into the possession when the action occurred."
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
        # Field renamed 2026-04-29 (D-44 prep) — see class docstring.
        elapsed = ctx.get("time_in_possession_s") or ctx.get("possession_elapsed_s")
        period = ctx.get("period", "?")
        clock = ctx.get("clock_end") or ctx.get("clock_start") or ""
        score = f"{ctx.get('score_home','?')}-{ctx.get('score_away','?')}"

        parts = [f"action={terminal}"]
        if elapsed is not None:
            parts.append(f"time_in_possession_s={elapsed}")
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


class _MarkerSurfacing:
    """Mixin-like helper for surfacing diagnostic state markers per-event.

    Used by Poker, CS:GO, and RL prompt builders so the violation injectors'
    derived-state markers (`already_folded`, `already_eliminated_this_round`,
    `pre_goal_state_persisting`) survive the default 6-key context cap.
    Mirrors the PUBG renderer override added 2026-04-29.
    """
    @staticmethod
    def surface_markers(ctx: dict, actor_anon: str, actor_map: dict | None = None) -> str:
        parts: list[str] = []
        if ctx.get("already_folded"):
            parts.append(f"NOTE={actor_anon}_already_folded")
        if ctx.get("already_eliminated_this_round"):
            parts.append(f"NOTE={actor_anon}_eliminated_this_round")
        if ctx.get("pre_goal_state_persisting"):
            parts.append("NOTE=pre_goal_state_persists_after_goal")
        if ctx.get("eliminates_player"):
            target = ctx["eliminates_player"]
            # CF-4=B: anonymise the target of the elimination, not just the
            # acting actor.
            anon_target = (
                actor_map.get(target, target) if actor_map else target
            )
            parts.append(f"ELIMINATES={anon_target}")
        if ctx.get("goal_marker"):
            parts.append("GOAL_SCORED")
        return " ".join(parts)


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
        # 2026-04-29 (D-43): surface diagnostic markers first so they survive.
        marker_str = _MarkerSurfacing.surface_markers(ctx, anon, actor_map)
        action_label = ctx.get("action_label") or "?"
        team_id = ctx.get("team_id") or ""
        team_slot = self._team_slot(team_id, ctx)
        round_num = ctx.get("round", "?")
        map_name = ctx.get("map", "?")
        # 2026-04-29 (D-44 prep): DROPPED `final_score` and `final_winner` —
        # these post-game state fields were appearing on every event line,
        # leaking end-of-match information into per-event prompts. Reviewer
        # synthesis (Gemini, ChatGPT) flagged this as the dominant remaining
        # FP driver after Layer-1 strict grounding. Removing leaked state ≠
        # tuning; it's removing information that doesn't belong on
        # individual event lines.
        std_str = (
            f"action={action_label}, team={team_slot}, round={round_num}, "
            f"map={map_name}"
        )
        ctx_str = f"{marker_str}, {std_str}" if marker_str else std_str
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
        # 2026-04-29 (D-43): surface diagnostic markers first.
        marker_str = _MarkerSurfacing.surface_markers(ctx, anon, actor_map)
        action_label = ctx.get("action_label") or "?"
        team_color = ctx.get("team_color") or event.actor_team or "?"
        score_b = ctx.get("score_blue", "?")
        score_o = ctx.get("score_orange", "?")
        duration = ctx.get("duration_s")
        std_str = (
            f"action={action_label}, team={team_color}, "
            f"final_score=blue:{score_b}–orange:{score_o}"
        )
        if duration is not None:
            try:
                std_str += f", match_duration_s={float(duration):.0f}"
            except (TypeError, ValueError):
                pass
        ctx_str = f"{marker_str}, {std_str}" if marker_str else std_str
        return (
            f"{idx:>3}. t={event.timestamp:>7.1f}s | "
            f"{event.event_type:<22} | actor={anon:<12} "
            f"| {ctx_str}"
        )


class PokerPromptBuilder(PromptBuilder):
    """P-5 constraint context. Wording locked in T-design review (2026-04-28)
    — game-name anchor ('NLHE') stripped to test rule-following rather than
    pretrained domain recall.

    2026-04-29 (D-43): format_event override added to surface diagnostic
    state markers (`already_folded`) per-event, mirroring PUBG's
    elimination-marker rendering. Without the override, the default
    6-key context cap drops these markers. The override applies the
    same logic uniformly across PUBG/Poker/CSGO/RL renderers.
    """

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

    def format_event(self, event: GameEvent, idx: int, actor_map: dict | None = None) -> str:
        anon = actor_map.get(event.actor, "Player_?") if actor_map else "Player_?"
        ctx = event.location_context or {}
        # Surface diagnostic markers first so they're never dropped.
        marker_str = _MarkerSurfacing.surface_markers(ctx, anon, actor_map)
        # Standard poker fields.
        street = ctx.get("street", "?")
        position = ctx.get("position", "?")
        action = ctx.get("action", "?")
        bet = ctx.get("bet_size_bb", "?")
        pot = ctx.get("pot_size_bb", "?")
        stack = ctx.get("stack_bb", "?")
        std_str = (
            f"street={street}, position={position}, action={action}, "
            f"bet_size_bb={bet}, pot_size_bb={pot}, stack_bb={stack}"
        )
        ctx_str = f"{marker_str}, {std_str}" if marker_str else std_str
        return (
            f"{idx:>3}. t={event.timestamp:>7.1f}s | "
            f"{event.event_type:<22} | actor={anon:<12} "
            f"| {ctx_str}"
        )


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
