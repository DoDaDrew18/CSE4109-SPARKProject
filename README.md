# SPARK

An AI pipeline for nowcasting county-level respiratory illness. We extract
structured signals from local news with an LLM, feed them to a supervised
nowcaster, and test one claim:

> Does text improve county-level respiratory nowcasts once the model sees only
> the data that existed on the prediction date?

A clean negative under honest evaluation is a real result, and we report it as
one.

CSE 4109: Introduction to AI for Health, Fall 2026.
Andrew Aviado, Brian Zhou, Noah Wolk.

## The one rule

Official respiratory surveillance has two defects. **Reporting lag**: this
week's ED percentage is not published this week. **Backfill**: the first value
is incomplete and revised upward for weeks.

Most models are scored against the finalized series, using numbers that did
not exist at prediction time. That inflates apparent skill. So:

**Nothing trains or scores against the finalized series. Every model reads a
vintage.**

A *vintage* is what was known on a given date. `SnapshotStore.load_vintage`
is the only sanctioned way to read the target, and `latest_vintage` is for
descriptive reporting only — never for scoring. If you find yourself reaching
past it to a CSV of final values, that is the leak, and the result will not
survive review.

```python
from src.snapshot import SnapshotStore

store = SnapshotStore("raw/snapshots")

# What we knew on 2026-01-14: week of 01-04 at 4.1%
store.load_vintage("nssp", "2026-01-14")

# What we know now: that same week settled at 6.0%
store.latest_vintage("nssp")

# How it got there -- the backfill curve
store.revision_history("nssp", geo_value="29189", time_value="2026-01-04")
```

Every observation carries two dates, following the Delphi Epidata convention:

| column       | meaning                                  |
| ------------ | ---------------------------------------- |
| `time_value` | the reference week the number describes   |
| `issue`      | the date that number was published        |

Snapshots are immutable. Corrections go in under a later `issue`, never as an
edit to a stored file — rewriting a snapshot would silently change what a past
model "knew" and quietly invalidate every backtest built on it.

## Layout

```
src/            pipeline modules
  snapshot.py   append-only snapshot store + vintage loader
tests/          pytest suite
raw/            immutable source pulls, partitioned by issue date (gitignored)
data/           derived/aligned feature tables (gitignored)
notebooks/      exploration
```

`raw/` and `data/` are gitignored except for their `.gitkeep`. Snapshots are
large and regenerable; they are versioned by issue date in the Box folder, not
in git. **Do not commit parquet.**

## Setup

```bash
git clone https://github.com/DoDaDrew18/CSE4109-SPARKProject.git
cd CSE4109-SPARKProject
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -q      # expect 12 passed
```

`requirements.txt` is deliberately lean: it installs from a clean venv with no
system libraries. Heavier dependencies (`geopandas`, `lightgbm`, `statsmodels`)
are listed as comments and get added when the stage that needs them lands, so a
failed wheel build never blocks the test gate.

API keys go in `.env`, which is gitignored. Never commit a key.

## Division of labor

| Member        | Role       | Owns                                                        |
| ------------- | ---------- | ----------------------------------------------------------- |
| Andrew Aviado | Data       | Multi-source ingestion, spatial/temporal alignment, vintages |
| Noah Wolk     | ML         | Baselines, forecasting models, backtesting harness, ablations |
| Brian Zhou    | Software   | LLM extraction & validation, GIS visualization, demo         |

## Status

- [x] **Week 1 gate** — snapshot store and vintage loader, 12 tests passing
- [ ] NSSP county coverage check (confirm before Demo I; fall back to state level if uneven)
- [ ] Ingestion: NSSP target, NWSS wastewater, AirNow, NOAA
- [ ] Seasonal-naive + SARIMA controls
- [ ] LLM extraction with quoted-span validation
- [ ] Gradient-boosted treatment model + quantile intervals
- [ ] Rolling-origin backtest, skill-over-baseline by horizon
- [ ] Leave-one-source-out ablation
- [ ] GeoPandas map + natural-language query (Demo I/II)
