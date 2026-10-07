# Using Local Data to Nowcast and Track County-Level Outbreaks

Andrew Aviado · Brian Zhou · Noah Wolk
CSE 4109: Introduction to AI for Health, Fall 2026 · Washington University in St. Louis

## The question

Emergency department (ED) visit counts and wastewater readings come out one to
two weeks late, and the early numbers keep changing for weeks after that. Local
news covers outbreaks sooner, but as plain text, not data.

We ask whether an LLM can pull useful facts out of local news, and whether
those facts help track county-level respiratory illness **when the model sees
only the delayed, still-changing official numbers**. If the answer is no, that
is still a useful result.

## Why "still-changing" matters — measured, not assumed

Most models are scored against the final, corrected numbers, which nobody had
on the day of the prediction, so they look better than they are. We checked
whether that is a real problem for our target:

> At the January 2026 flu peak (epiweek 202601), **19% of US counties had their
> first published value revised by 10% or more**, and revisions ran upward
> 1,370 times versus 215 downward.

So every model in this repo trains and scores on a **vintage**: the data
exactly as it was published on the prediction date.

## Approach

**(a) Extraction.** An LLM reads local news and turns each article into a short
record: county, illness, how bad it sounds, and the date. Articles carry no
county tag, so the model infers it; a second pass checks every field against a
direct quote and drops anything it cannot ground.

**(b) Nowcasting.** Estimate this week's share of ED visits that are
respiratory, per county. Start with linear regression on lagged official
numbers, add SARIMA and gradient boosting, then layer news features on top to
see whether they buy anything.

**(c) Ablation.** Drop one source at a time and measure the accuracy lost.

**Evaluation.** Extraction: quote-grounding on every article, plus a hand-check
of 50–80 articles for precision and recall, weighted toward the inferred county
field. Nowcasting: R², the fraction of the gap between late and final data that
each model closes, and a Wilcoxon signed-rank test on paired weekly errors.
Rolling-origin splits, results grouped by county population.

## Data

| Source | What it gives us | Access | Status |
| --- | --- | --- | --- |
| **CDC NSSP**, via Delphi Epidata | **Target**: % of ED visits for COVID, flu, RSV — weekly, ~2,400 counties, with revision history | Public API, no key | ✅ `src/ingest_nssp.py` |
| **CDC NWSS** wastewater, via data.cdc.gov | Early warning that doesn't depend on anyone seeking care | Public API, no key | Next — dataset `atcp-73re` |
| **GDELT** | Raw news text for the LLM | Public API, no key | After the county list is set |
| County boundaries (Census) | Map shapes, joined on 5-digit FIPS | Static file, no API | For the map |
| EPA AirNow | PM2.5, ozone | Free key required | Deferred past Demo I |
| NOAA | Temperature, humidity | Free token required | Deferred past Demo I |

What we verified against the live API (October 2026):

- **The combined respiratory signal is discontinued** (ends epiweek 202439).
  We use the three live signals — COVID, influenza, RSV — and combine them at
  the modeling stage.
- **Honest backtests start in April 2024.** Delphi archives first prints from
  epiweek 202416 on; weeks before that only exist as already-revised values.
  Some later weeks have holes in the archive.
- **`as_of` takes an epiweek, not a date.** `as_of=202607` works.
  `as_of=20260215` is silently read as a far-future week and returns the
  *latest* data — a leak that raises no error. The ingester refuses it.

## Data collection protocol

1. **Every week, run** `python -m src.ingest_nssp`. Each run saves one vintage
   of all counties, stamped with that day's date. A missed week can be rebuilt
   later with `--as-of`, but only as well as Delphi's archive allows.
2. **Read the target only through a vintage.** Use
   `SnapshotStore.load_vintage(source, as_of)` for anything that trains or
   scores. `latest_vintage` is for plots and descriptive stats, never for scoring.
3. **Snapshots are never edited.** A correction goes in under a new issue date.
4. **Data never goes in git.** `raw/` and `data/` are gitignored; share
   snapshots through the Box folder.

```python
from src.snapshot import SnapshotStore

store = SnapshotStore("raw/snapshots")
store.load_vintage("nssp_influenza", "2026-08-08")   # what we knew on Aug 8
store.revision_history("nssp_influenza", "17031", "2026-07-26")  # one county-week's revisions
```

## Setup

```bash
git clone https://github.com/DoDaDrew18/CSE4109-SPARKProject.git
cd CSE4109-SPARKProject
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -q          # expect 15 passed
python -m src.ingest_nssp           # pull this week's vintage
```

## Layout

```
src/snapshot.py      append-only snapshot store + vintage loader
src/ingest_nssp.py   NSSP target ingestion (today's vintage, or --as-of epiweek)
tests/               pytest suite (offline; no network needed)
raw/                 snapshots by source and issue date (gitignored)
data/                aligned county-week tables (gitignored)
notebooks/           exploration
```

## Team

| Member | Role | Owns |
| --- | --- | --- |
| Andrew Aviado | Data Lead | Ingestion, county/week alignment, vintage handling |
| Noah Wolk | ML Engineer | Baselines, nowcasting models, backtesting, ablations |
| Brian Zhou | Software Engineer | LLM extraction and validation, map, demo |

## Status

- [x] Snapshot store and vintage loader
- [x] NSSP target ingestion, verified end to end against the live API
- [ ] Choose the 50–100 study counties (population × NSSP coverage × news coverage)
- [ ] NWSS wastewater ingestion
- [ ] Aligned county-week training table for modeling
- [ ] GDELT article pull for extraction
- [ ] Baselines → SARIMA → gradient boosting → + news features
- [ ] Map with natural-language search (Demo I/II)
