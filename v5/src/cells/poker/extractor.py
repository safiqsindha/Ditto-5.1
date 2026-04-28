"""
Texas Hold'em poker event extractor for v5.

Converts parsed PHH hand dicts (produced by PokerPipeline.parse()) into
normalized GameEvent streams.

Input format: dict with keys
  game_id, players (already anonymised as actor_N), starting_stacks,
  blinds, big_blind, actions (list of PHH action strings), subset

PHH action string format (from phh.readthedocs.io):
  "p<idx> f"          — fold
  "p<idx> cc"         — check or call
  "p<idx> cbr <amt>"  — complete, bet, or raise to <amt>
  "p<idx> sm [cards]" — show or muck (not a decision event)
  "d dh p<idx> <cards>" — deal hole cards (not a decision event)
  "d db <cards>"        — deal board cards (marks street change)

CF-4=B: player names are anonymised in the parse step (pipeline.py),
so the extractor receives actor_0, actor_1, ... identifiers.
"""

from __future__ import annotations

import logging

from ...common.schema import EventStream, GameEvent

logger = logging.getLogger(__name__)

# PHH action code → normalized event_type
_ACTION_MAP: dict[str, str] = {
    "f": "disengage_decision",   # fold
    "cc": "engage_decision",     # check or call
    "cbr": "engage_decision",    # complete, bet, or raise
}

_STREETS = ("preflop", "flop", "turn", "river")

# Position names by player count and seat index
_POSITIONS: dict[int, dict[int, str]] = {
    2: {0: "SB", 1: "BB"},
    3: {0: "BTN", 1: "SB", 2: "BB"},
    4: {0: "BTN", 1: "SB", 2: "BB", 3: "UTG"},
    5: {0: "BTN", 1: "SB", 2: "BB", 3: "UTG", 4: "CO"},
    6: {0: "SB", 1: "BB", 2: "UTG", 3: "HJ", 4: "CO", 5: "BTN"},
}


class PokerExtractor:
    """
    Convert a parsed PHH hand dict into a normalized GameEvent stream.

    Each player action (fold/check/call/bet/raise) becomes one GameEvent.
    Dealer actions (deal board, deal hole cards) update internal state but
    do not produce events.
    """

    def extract(self, record: dict) -> EventStream:
        game_id = record.get("game_id", "pk_unknown")
        players: list[str] = record.get("players", [])
        starting_stacks: list[int] = list(record.get("starting_stacks", []))
        blinds: list[int] = list(record.get("blinds", []))
        big_blind: int = int(record.get("big_blind", 100)) or 100
        actions: list[str] = record.get("actions", [])
        subset: str = record.get("subset", "")

        n_players = len(players)
        pos_map = _POSITIONS.get(n_players, {})

        # Initialise stacks and pot with blind contributions
        stacks = list(starting_stacks) if starting_stacks else [1000 * big_blind] * n_players
        while len(stacks) < n_players:
            stacks.append(1000 * big_blind)

        pot = 0
        for i, b in enumerate(blinds[:n_players]):
            if b > 0:
                stacks[i] = max(0, stacks[i] - b)
                pot += b

        # Per-street tracking
        board_deal_count = 0    # increments on each "d db ..." action
        contributions = [0] * n_players   # chips put in this street by each player
        current_bet = max(blinds[:n_players]) if blinds else 0
        for i, b in enumerate(blinds[:n_players]):
            if i < len(contributions):
                contributions[i] = b

        stream = EventStream(
            game_id=game_id,
            cell="poker",
            metadata={"subset": subset, "n_players": n_players},
        )
        seq = 0
        ts = 0.0

        for action_str in actions:
            parsed = _parse_action_string(action_str)
            if parsed is None:
                continue

            atype = parsed["type"]

            if atype == "deal_board":
                board_deal_count += 1
                contributions = [0] * n_players
                current_bet = 0
                ts += 2.0
                continue

            if atype != "player_action":
                continue

            action_code: str = parsed["action_code"]
            if action_code not in _ACTION_MAP:
                continue          # sm / uf / unknown — skip

            player_idx: int = parsed["player_idx"]
            if player_idx >= n_players:
                logger.debug(f"[poker] player_idx {player_idx} >= n_players {n_players}")
                continue

            etype = _ACTION_MAP[action_code]
            amount_chips = 0

            if action_code == "cbr":
                raw_amount = int(parsed.get("amount") or 0)
                contrib_so_far = contributions[player_idx] if player_idx < len(contributions) else 0
                add = max(0, raw_amount - contrib_so_far)
                amount_chips = raw_amount
                stacks[player_idx] = max(0, stacks[player_idx] - add)
                if player_idx < len(contributions):
                    contributions[player_idx] = raw_amount
                pot += add
                current_bet = raw_amount

            elif action_code == "cc":
                contrib_so_far = contributions[player_idx] if player_idx < len(contributions) else 0
                to_call = max(0, current_bet - contrib_so_far)
                to_call = min(to_call, stacks[player_idx])
                amount_chips = to_call
                stacks[player_idx] = max(0, stacks[player_idx] - to_call)
                if player_idx < len(contributions):
                    contributions[player_idx] += to_call
                pot += to_call

            street = _STREETS[min(board_deal_count, 3)]
            actor = players[player_idx] if player_idx < len(players) else f"actor_{player_idx}"
            position = pos_map.get(player_idx, f"seat_{player_idx}")
            stack_bb = round(stacks[player_idx] / big_blind, 1)
            pot_bb = round(pot / big_blind, 1)
            bet_bb = round(amount_chips / big_blind, 1)

            ev = GameEvent(
                timestamp=ts,
                event_type=etype,
                actor=actor,
                location_context={
                    "street": street,
                    "position": position,
                    "action": action_code,
                    "bet_size_bb": bet_bb,
                    "pot_size_bb": pot_bb,
                    "stack_bb": stack_bb,
                    "num_players": n_players,
                },
                raw_data_blob={
                    "player_idx": player_idx,
                    "action": action_code,
                    "amount_chips": amount_chips,
                },
                cell="poker",
                game_id=game_id,
                sequence_idx=seq,
                phase=street,
                metadata={},
            )
            stream.append(ev)
            seq += 1
            ts += max(3.0, float(player_idx) * 0.3 + 2.0)

        return stream


# ---------------------------------------------------------------------------
# PHH action string parser
# ---------------------------------------------------------------------------

def _parse_action_string(s: str) -> dict | None:
    """
    Parse one PHH action string into a structured dict.

    Returns None for unrecognised or irrelevant strings.
    """
    parts = s.strip().split()
    if not parts:
        return None

    if parts[0] == "d":
        # Dealer action
        if len(parts) >= 3 and parts[1] == "db":
            return {"type": "deal_board", "cards": parts[2] if len(parts) > 2 else ""}
        if len(parts) >= 4 and parts[1] == "dh":
            return {"type": "deal_hole", "player": parts[2], "cards": parts[3]}
        return None

    actor_tok = parts[0]
    if not (actor_tok.startswith("p") and actor_tok[1:].isdigit()):
        return None

    player_idx = int(actor_tok[1:])
    if len(parts) < 2:
        return None

    action_code = parts[1]
    amount = None
    if len(parts) > 2:
        try:
            amount = float(parts[2])
        except ValueError:
            pass

    return {
        "type": "player_action",
        "player_idx": player_idx,
        "action_code": action_code,
        "amount": amount,
    }
