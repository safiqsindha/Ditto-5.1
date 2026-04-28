# v5 Pilot Performance Benchmark

Run on: 2026-04-27 17:45:26

## Summary

| Cell | Streams | Events | Chains | Mock-gen (s) | Pilot (s) | Total (s) | Events/s | Chains/s | Mem Δ (MB) |
|------|---------|--------|--------|--------------|-----------|-----------|----------|----------|------------|
| fortnite | 200 | 24,000 | 8,000 | 0.07 | 0.036 | 0.106 | 343,708 | 221,227 | 26.3 |
| nba | 300 | 55,200 | 18,420 | 0.154 | 0.093 | 0.247 | 358,774 | 198,610 | 32.7 |
| csgo | 150 | 45,000 | 15,000 | 0.122 | 0.094 | 0.215 | 370,015 | 160,125 | 0.0 |
| rocket_league | 250 | 50,000 | 16,750 | 0.184 | 0.099 | 0.283 | 271,647 | 169,969 | 4.1 |
| hearthstone | 300 | 23,100 | 7,740 | 0.079 | 0.039 | 0.118 | 292,818 | 196,508 | 0.0 |

**Total wall-clock:** 0.97s for 197,300 events and 65,910 chains across all 5 cells.

## Implications for real-data acquisition

These numbers are mock-data only. Real-data acquisition will be bound by:
- Network I/O (HLTV demo downloads, NBA API, BallChasing API, HSReplay API)
- Parser CPU (awpy, carball, hslog)
- Disk I/O for raw replays (~18GB total estimated, see REAL_DATA_GUIDE.md)

Pilot validation itself (T + Gate 2) is fast — well under 1 minute even at
production scale (1,200 chains/cell × 5 cells = 6,000 chains).
