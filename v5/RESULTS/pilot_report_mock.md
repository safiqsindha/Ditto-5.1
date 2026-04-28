# v5 Pilot Validation Report

**Aggregate result:** ALL PASS


## Per-Cell Summary

| Cell | Streams | Events | Raw Chains | Post-Gate2 | Retention | Result |
|------|---------|--------|------------|------------|-----------|--------|
| fortnite | 200 | 24,000 | 8,000 | 8,000 | 100.0% | PASS |
| nba | 300 | 55,200 | 18,420 | 18,420 | 100.0% | PASS |
| csgo | 150 | 45,000 | 15,000 | 15,000 | 100.0% | PASS |
| rocket_league | 250 | 50,000 | 16,750 | 16,750 | 100.0% | PASS |
| hearthstone | 300 | 23,100 | 7,740 | 7,740 | 100.0% | PASS |

---

## [fortnite]

- **Streams:** 200
- **Total events:** 24,000
- **Raw chains:** 8,000
- **Post-Gate2 chains:** 8,000
- **Retention:** 100.0% (floor=50%)
- **Gate 2:** PASS

### Chain length distribution
- mean: 4.95
- min: 3
- max: 5
- median: 5.0

### Actionable fraction distribution
- mean: 100.0%
- min: 100.0%
- max: 100.0%

### Top 10 event types
| Type | Count |
|------|-------|
| strategy_adapt | 2,293 |
| resource_gain | 2,277 |
| team_coordinate | 2,272 |
| engage_decision | 2,256 |
| rotation_commit | 2,253 |
| risk_accept | 2,248 |
| zone_enter | 2,246 |
| zone_exit | 2,231 |
| ability_use | 2,224 |
| item_use | 2,203 |

---

## [nba]

- **Streams:** 300
- **Total events:** 55,200
- **Raw chains:** 18,420
- **Post-Gate2 chains:** 18,420
- **Retention:** 100.0% (floor=50%)
- **Gate 2:** PASS

### Chain length distribution
- mean: 4.96
- min: 2
- max: 5
- median: 5.0

### Actionable fraction distribution
- mean: 100.0%
- min: 100.0%
- max: 100.0%

### Top 10 event types
| Type | Count |
|------|-------|
| engage_decision | 6,779 |
| objective_contest | 6,739 |
| rotation_commit | 6,661 |
| disengage_decision | 6,608 |
| target_select | 6,574 |
| strategy_adapt | 6,544 |
| resource_spend | 6,527 |
| timing_commit | 6,504 |
| resource_gain | 6,467 |
| team_coordinate | 6,455 |

---

## [csgo]

- **Streams:** 150
- **Total events:** 45,000
- **Raw chains:** 15,000
- **Post-Gate2 chains:** 15,000
- **Retention:** 100.0% (floor=50%)
- **Gate 2:** PASS

### Chain length distribution
- mean: 4.98
- min: 3
- max: 5
- median: 5.0

### Actionable fraction distribution
- mean: 100.0%
- min: 100.0%
- max: 100.0%

### Top 10 event types
| Type | Count |
|------|-------|
| zone_enter | 4,249 |
| position_commit | 4,240 |
| strategy_adapt | 4,239 |
| risk_reject | 4,217 |
| objective_capture | 4,197 |
| resource_budget | 4,163 |
| objective_contest | 4,162 |
| resource_spend | 4,155 |
| disengage_decision | 4,148 |
| timing_commit | 4,143 |

---

## [rocket_league]

- **Streams:** 250
- **Total events:** 50,000
- **Raw chains:** 16,750
- **Post-Gate2 chains:** 16,750
- **Retention:** 100.0% (floor=50%)
- **Gate 2:** PASS

### Chain length distribution
- mean: 4.96
- min: 2
- max: 5
- median: 5.0

### Actionable fraction distribution
- mean: 100.0%
- min: 100.0%
- max: 100.0%

### Top 10 event types
| Type | Count |
|------|-------|
| timing_commit | 5,131 |
| objective_contest | 5,007 |
| engage_decision | 4,936 |
| risk_accept | 4,927 |
| team_coordinate | 4,926 |
| zone_enter | 4,903 |
| strategy_adapt | 4,893 |
| zone_exit | 4,882 |
| ability_use | 4,879 |
| rotation_commit | 4,875 |

---

## [hearthstone]

- **Streams:** 300
- **Total events:** 23,100
- **Raw chains:** 7,740
- **Post-Gate2 chains:** 7,740
- **Retention:** 100.0% (floor=50%)
- **Gate 2:** PASS

### Chain length distribution
- mean: 4.91
- min: 2
- max: 5
- median: 5.0

### Actionable fraction distribution
- mean: 100.0%
- min: 100.0%
- max: 100.0%

### Top 10 event types
| Type | Count |
|------|-------|
| resource_spend | 2,329 |
| zone_enter | 2,307 |
| timing_commit | 2,285 |
| resource_gain | 2,285 |
| concede | 2,270 |
| draft_pick | 2,244 |
| position_commit | 2,241 |
| target_select | 2,241 |
| strategy_adapt | 2,234 |
| resource_budget | 2,224 |
