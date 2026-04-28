# v5 Phase B Prep — T Design Joint Authoring Session

**Audience:** Lead Author + Myriam  
**Purpose:** Pre-think the hard design decisions so the joint session is productive  
**Status:** DRAFT — fill in answers before/during the session  
**Output:** Each cell will have a fully-specified T that can be implemented same-session

---

## Common framework questions (apply to all cells)

Answer once for the whole experiment:

### CF-1 — How is "constraint" expressed in v5?
v3's chess constraint was something like "the piece-movement rules of the game". In v5, what's the constraint structure we're testing detection for?

- **Option A:** A natural-language description of the rule structure (e.g., "in CS:GO, the team without the bomb cannot win by elimination after time expires").
- **Option B:** A formal-ish predicate set (e.g., bullet list of invariants).
- **Option C:** A reference to in-context examples.

**Recommendation:** A — most aligned with v3. Easiest to make consistent across 5 cells.

**Decision:** _________________

---

### CF-2 — What is the prediction target?
What is the model asked to do with the chain?

- **Option A:** Predict the next event type (e.g., "what kind of decision happens next?").
- **Option B:** Predict the actor of the next event (e.g., "which player makes the next move?").
- **Option C:** Predict a domain-relevant outcome (e.g., "does this team win this round?").
- **Option D:** Classify the chain as constraint-respecting or constraint-violating (binary).

**Recommendation:** D — directly tests detection methodology, matches v3's framing best. The chain IS the artifact under test.

**Decision:** _________________

---

### CF-3 — How are constraint-violating "shuffled" controls constructed?
In v3, the experiment compared real chains vs shuffled chains. Do we replicate this?

- **Option A:** Yes — for each cell, generate 1× shuffled chains (random event reordering within game) as controls. This 2× the cost but matches v3.
- **Option B:** No — v5 tests detection methodology generalization, not real-vs-shuffled. Skip controls.
- **Option C:** Yes but lighter — 0.25× shuffled per cell as a sanity check, not the primary comparison.

**Recommendation:** Need both authors to confirm. v3 used shuffled as the primary control; v5's primary research question may differ.

**Decision:** _________________

---

### CF-4 — What's the chain provenance metadata kept in the prompt?
Does the model see the source game/match identifier, tournament/event, date, etc.?

- **Option A:** Show none (chains are anonymous event sequences).
- **Option B:** Show domain only (e.g., "this is a CS:GO chain") but no game/team/player identifiers.
- **Option C:** Show full provenance for grounding.

**Recommendation:** B — prevents the model from leaning on memorized facts about specific tournaments while still letting it know what game it's analyzing.

**Decision:** _________________

---

## Per-cell questions

For each cell answer these 6 questions. The answers become the T implementation.

### Fortnite

#### F-1. What's the constraint structure?
What rule(s) does the model need to detect violations of in Fortnite?
*Suggested:* storm-zone boundary rule (player must be inside zone or take damage), build-cost rule (each build consumes one material), elimination causality (a player can't act after being eliminated).

**Locked:** _________________

#### F-2. What's the chain selection criterion?
Which event sub-sequences should T extract as candidates? Examples:
- A storm-rotation phase (zone announce → players move → zone closes)
- A late-game elimination chain
- A 90s window centered on a high-stakes engagement

**Suggested:** Storm-rotation phase (cleanest constraint-bearing context).

**Locked:** _________________

#### F-3. Per-cell chain length N?
Given target ~80–120 events/match in tournament play, and stratifying for storm-rotation phases:
- **Recommended:** N = 8 (enough events to capture a rotation, short enough to keep prompt < 600 tokens)

**Locked N = _____**

#### F-4. Prediction target?
*If CF-2 = D:* "Is the next-event ordering consistent with Fortnite's rules?"  
*If CF-2 = A:* "Predict the next event type."

**Locked:** _________________

#### F-5. Constraint-context language sketch?
Write 2–3 sentences. Example:
> "In Fortnite, players must remain within the safe storm zone or take damage. Building structures consumes materials. Eliminations are one-way: an eliminated player cannot generate further actions."

**Locked:** _________________

#### F-6. Are domain-specific actionable types needed?
Should `CELL_ACTIONABLE_OVERRIDES["fortnite"]` include anything beyond v1.1 ACTIONABLE_TYPES?
*Candidates:* `storm_rotation`, `build_decision`, `loot_priority`.

**Locked:** _________________

---

### NBA

#### N-1. What's the constraint structure?
*Suggested:* shot-clock rule (offense must shoot within 24s), possession-rules (team has 5 in-bounds players), foul-out rule (player ejected at 6 fouls).

**Locked:** _________________

#### N-2. Chain selection criterion?
Per Q6-A possession-level, each chain is 1+ possessions. Selection:
- **Recommended:** consecutive possessions sampled from a single quarter, including the possession-ending event.

**Locked:** _________________

#### N-3. Per-cell chain length N?
NBA possessions average ~3 events at the granularity we extract (offensive trip + defensive set). For 5–8 plays of context:
- **Recommended:** N = 5 (one full possession + lead-in)

**Locked N = _____**

#### N-4. Prediction target?
*Suggested:* "Was the terminal action consistent with NBA rules given the preceding state?" or "Predict the next possession's type (made shot / turnover / foul)."

**Locked:** _________________

#### N-5. Constraint-context language sketch?
Example:
> "In NBA basketball, each offensive possession must produce a shot attempt within 24 seconds. A team that loses possession via turnover or made shot transitions to defense. Personal fouls accumulate; six fouls eject the player."

**Locked:** _________________

#### N-6. Domain-specific actionables?
*Candidates:* `shot_selection`, `defensive_assignment`, `clutch_decision`.

**Locked:** _________________

---

### CS:GO / CS2

#### C-1. Constraint structure?
*Suggested:* round-economy rule (each player has $X to spend pre-round), bombsite-control rule (team in possession of bomb has objective), respawn-disabled rule (eliminated players sit out the round).

**Locked:** _________________

#### C-2. Chain selection criterion?
Per Q5-A round-level. Selection:
- **Recommended:** Full round (buy phase + tactical phase + outcome). One round ≈ one chain.

**Locked:** _________________

#### C-3. Per-cell chain length N?
A round has ~10 events at our granularity. For one full round of context:
- **Recommended:** N = 10

**Locked N = _____**

#### C-4. Prediction target?
*Suggested:* "Predict round outcome (CT win / T win / draw)" — directly tests whether the model has learned the constraint structure.

**Locked:** _________________

#### C-5. Constraint-context language sketch?
Example:
> "In Counter-Strike, two teams (Counter-Terrorists and Terrorists) play 12+ rounds per half. Terrorists win by planting and detonating the bomb at site A or B; Counter-Terrorists win by defusing the bomb or eliminating all attackers. Eliminated players cannot return until the round ends."

**Locked:** _________________

#### C-6. Domain-specific actionables?
*Candidates:* `buy_decision`, `utility_deploy`, `bombsite_commit`, `clutch_decision`.

**Locked:** _________________

---

### Rocket League

#### R-1. Constraint structure?
*Suggested:* boost-economy rule (max 100, depletes with use), goal-causality rule (goal scored = ball crosses goal-line), 3v3 player-count rule.

**Locked:** _________________

#### R-2. Chain selection criterion?
Per Q7-C boost-enriched hit-level. Selection:
- **Recommended:** A "play" = sequence from one team's possession of ball to the next team's possession or goal. Boost events interleaved.

**Locked:** _________________

#### R-3. Per-cell chain length N?
Boost-enriched hits run ~250–400/game. For one play:
- **Recommended:** N = 12 (typical play has ~6–8 hits + 4–6 boost events)

**Locked N = _____**

#### R-4. Prediction target?
*Suggested:* "Predict whether the play results in a goal" or "Predict the next ball-touch outcome (shot / save / clear)."

**Locked:** _________________

#### R-5. Constraint-context language sketch?
Example:
> "In Rocket League, two teams of three rocket-powered cars compete to score by hitting a ball into the opposing goal. Each car has a boost meter (max 100) that depletes with use; players collect boost from pads on the field. A goal ends the play and resets ball position to centerfield."

**Locked:** _________________

#### R-6. Domain-specific actionables?
*Candidates:* `aerial_commit`, `demo_attempt`, `boost_steal`, `rotation_back`.

**Locked:** _________________

---

### Hearthstone

#### H-1. Constraint structure?
*Suggested:* mana-cost rule (each turn N mana, cards cost N to play), turn-alternation (opponent's turn vs your turn), board-state rule (minions have HP/attack; reach zero → die).

**Locked:** _________________

#### H-2. Chain selection criterion?
Per Q8-A per-action. Selection:
- **Recommended:** All actions within a single player's turn (one card play sequence + battlecries + attacks + hero power).

**Locked:** _________________

#### H-3. Per-cell chain length N?
A turn has 3–8 actions in HS. For full-turn context:
- **Recommended:** N = 6 (median turn length)

**Locked N = _____**

#### H-4. Prediction target?
*Suggested:* "Predict whether this turn is consistent with HS rules" or "Predict the next action's type (play card / attack / hero-power)."

**Locked:** _________________

#### H-5. Constraint-context language sketch?
Example:
> "In Hearthstone, two players take alternating turns. Each turn, the active player gains one mana crystal (max 10) and may play cards costing up to that turn's mana. Cards are drawn from a deck. Minions on the board have attack and health stats; combat reduces health to zero to remove minions from play."

**Locked:** _________________

#### H-6. Domain-specific actionables?
*Candidates:* `card_play`, `combo_trigger`, `lethal_lining_up`, `mana_curve_choice`.

**Locked:** _________________

---

## After all 8 questions are answered

The session writes T per cell:

```python
class FortniteT(TranslationFunction):
    @property
    def cell(self) -> str:
        return "fortnite"

    def translate(self, stream: EventStream) -> list[ChainCandidate]:
        # Implementation per F-1 through F-5 answers
        ...
```

Then update `config/harness.yaml`:
```yaml
chain_length:
  per_cell:
    fortnite: 8       # F-3
    nba: 5            # N-3
    csgo: 10          # C-3
    rocket_league: 12 # R-3
    hearthstone: 6    # H-3
```

Then update `CELL_ACTIONABLE_OVERRIDES` in `actionables.py` based on the *-6 answers.

Then update each `*PromptBuilder.format_constraint_context()` per the *-5 answers.

---

## Out of session

After the session, I'll:
1. Implement the 5 T classes from the answers
2. Lock per-cell chain lengths in harness.yaml
3. Wire ChainBuilder into run_pilot with the locked lengths
4. Re-run pilot — first time we'll see Gate 2 retention with realistic T behavior
5. Update DECISION_LOG with D-24 through D-28 (one per cell's T design)
6. Commit + push, ready for Phase C credentials provisioning

---

## Time estimate

- Common framework questions (CF-1 through CF-4): 30 min
- Per-cell × 5 cells × 6 questions: ~30 min/cell = 2.5h
- T implementation pass with both authors: 1.5h
- Pilot rerun + verification: 30 min

**Total: ~4.5 hours** for Phase B.

If you want to split it, the natural break is after the per-cell decisions; implementation can be done by me asynchronously and reviewed via PR comments.
