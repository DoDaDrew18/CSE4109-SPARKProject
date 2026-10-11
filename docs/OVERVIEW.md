# Project overview: what we're building and what we hope to find

Schematic: [`schematic.png`](schematic.png) (simple, by layer) ·
[`pipeline.png`](pipeline.png) (detailed, every input and output)

## In one paragraph

This is a **retrospective** study. We rebuild respiratory-illness data for past
weeks **exactly as it looked on the day it was published**. Then, one date at a
time, we ask which other signals line up with what emergency department (ED)
visits turned out to be: weather (NOAA), wastewater (CDC), and local news. We
**nowcast** (estimate this week's value) and do not forecast the future. St.
Louis is the home city. Five other Midwest/South metros test whether the result
carries over.

## The layers

| # | Layer | Question it answers | Output |
| --- | --- | --- | --- |
| 1 | **Collect** | What raw data exists and where does it come from? | Raw downloads, each stamped with its publish date |
| 2 | **Reconstruct** | What did we actually know on date D? | `load_vintage(source, D)` views (already built for NSSP) |
| 3 | **Align** | Can everyone work from one table? | `county_week` table (`data/county_week.parquet`) |
| 4 | **Read the news** | Can an LLM label articles as reliably as we can? | `article_labels` + LLM-vs-human agreement |
| 5 | **Analyze & nowcast** | Which signals move with illness, and does adding them help? | Correlations at different lags; error for each model in the ladder |
| 6 | **Place & generalize** | Where does it cluster, and does St. Louis carry over to other cities? | Map + hot spots; the same model run on 5 other metros |

### What we hope to find (hypotheses, any of which may come out null)

1. **Weather:** cold, dry weeks (low temperature, low dew point) come before
   flu/RSV rises by 1–3 weeks.
2. **Wastewater:** wastewater moves before ED visits and helps most in weeks when
   the official ED numbers are still incomplete.
3. **News:** local news about a surge appears before the official numbers are
   final, so it helps on recent weeks most of all.
4. **Generalization:** a model built on St. Louis keeps most of its accuracy in
   similar metros. News helps less where local coverage is thin.

A "no" on any of these still counts as a result, as long as the test was fair.

## Layer 1: data sources (checked live, Oct 2026)

| Source | What we take | How | Key? | Notes |
| --- | --- | --- | --- | --- |
| **CDC NSSP** via Delphi Epidata | % of ED visits for COVID, flu, RSV; county, weekly | `src/ingest_nssp.py` (done) | No | Honest history starts with epiweek 202416 (Apr 2024) |
| **CDC NWSS** wastewater | Site-level viral activity per week, COVID/flu/RSV | `data.cdc.gov/resource/atcp-73re.json` (Socrata) | No | County names, not FIPS (`src/fips.py` maps them). **Revised silently**: whole history re-posted weekly; Aug 2026 method change. Honest vintages only from 2026-04-17. |
| **NOAA NCEI** daily summaries + **IEM ASOS** | Temp, precip (NCEI); dew point, humidity (IEM) | `ncei.noaa.gov/access/services/data/v1` · `mesonet.agron.iastate.edu` | **No** | NCEI humidity ends 2024-12-31, so humidity comes from IEM (same airport sensors, r = 0.999). Treated as available 7 days after the fact. |
| **GDELT** | Local article URLs + titles | DOC API, or raw GKG files (`data.gdeltproject.org/gdeltv2/`) | No | DOC API reaches past peaks with `startdatetime`, but over-limit use blocks the IP for hours. Raw GKG files have no limit (~0.5 GB/day). |
| **Hand-picked news** | 60–200 articles around the 2024–25 and 2025–26 peaks | Local outlet sites/archives (e.g. St. Louis Public Radio, Post-Dispatch, local TV) | No | This doubles as our labeled set |
| Census county shapes | Boundaries + population | Static TIGER/cartographic file | No | For the map only |
| EPA air quality | PM2.5, ozone | AirNow/AQS | Yes | Stretch goal, not this week |

**Cities (county FIPS):** St. Louis city `29510` + St. Louis County `29189`;
Kansas City/Jackson `29095`; Chicago/Cook `17031`; Indianapolis/Marion `18097`;
Memphis/Shelby `47157`; Louisville/Jefferson `21111`.

## Layer 3: the two tables

**`county_week`** has one row per county × week × as-of date. It is the only table
the models read.

| Column | Meaning | Source |
| --- | --- | --- |
| `fips`, `metro` | County and city group | – |
| `week_end`, `as_of` | Week described; date the numbers are "as known on" | – |
| `ed_covid`, `ed_flu`, `ed_rsv` | % ED visits as known on `as_of` | NSSP |
| `wastewater_level` | County average of site levels | NWSS |
| `temp_mean_f`, `dewpoint_f`, `precip_in` | Weekly mean / mean / sum | NOAA |
| `news_n`, `news_concern` | Article count, mean concern 0–3 | News labels |
| `ed_resp_final` | Final corrected value: **answer key only, never an input** | NSSP latest |

**`article_labels`** has one row per article per labeler: `article_id, url,
outlet, published, fips, illness, concern (0–3), quote, labeler
(human_a | human_b | llm)`.

## Layer 4: keeping the LLM honest

Our main worry is the LLM making up labels. The plan:

1. **Measure "concern" instead of open-ended "sentiment".** Outbreak news reads
   negative almost every time, so a general sentiment score tells us little. A
   0–3 scale (none / mention / rising / surge) is easier to check and easier to
   agree on.
2. **Every label needs a quote.** The LLM has to copy the sentence that supports
   each field. Code checks that the sentence is really in the article and drops
   the label if it isn't. This catches made-up labels automatically.
3. **Humans first.** Two of us label the same 30 articles, which gives our own
   agreement (Cohen's κ). One person labels the rest.
4. **Compare.** The LLM labels all the articles, and we report its agreement with
   the human labels per field: county, illness, concern. If the LLM agrees with
   humans **less** than humans agree with each other, we use human labels only
   and say so in the report.
5. **Simple baseline.** A word-list sentiment scorer (VADER) runs on the same
   articles, so we can show whether the LLM adds anything at all.

Scale stays small on purpose: 60 articles is the minimum, and 200 if time
allows. With this few, news is treated as a **signal around flu peaks**, not
something present every week. We report it that way.

## Layer 6: GIS and SaTScan, realistically

- **Map:** a Census county shapefile joined on FIPS, drawn with geopandas or
  folium. This is doable within the week.
- **Hot spots:** Getis-Ord Gi\* or local Moran's I from **PySAL** (`esda`). It is
  pure Python, works on the `county_week` table directly, and finds
  statistically significant clusters.
- **SaTScan:** a desktop program with **no web API**. It does have a batch
  mode, so we can export case/population/coordinate files, run it, and read the
  cluster output back. **Stretch goal only.**

## One-week plan (data collection first)

| Day | Andrew (data) | Noah (ML) | Brian (LLM & demo) |
| --- | --- | --- | --- |
| 1 | Lock the 6 metros / 7 counties; NOAA ingester | Set up the `county_week` schema in code | Pick outlets; start article list |
| 2 | NWSS ingester + county-name→FIPS map | Last-known-value baseline on NSSP vintages | Write labeling guide (concern 0–3) |
| 3 | First `county_week` build (STL only) | Correlation-at-lag plots (STL) | 2-person labeling of 30 articles |
| 4 | Add the other 5 metros | Linear model + weather | LLM labeling + quote check |
| 5 | Fix gaps, document missing data | Add wastewater; rolling evaluation | LLM-vs-human agreement |
| 6 | Freeze data snapshot in Box | Results table | STL map + Gi\* hot spots |
| 7 | **Demo:** STL map, one nowcast date walked through, agreement numbers | | |

**Not this week:** EPA air quality, SaTScan, gradient boosting, news features
for every week, all ~2,400 counties.

## Status after the first build (Oct 2026)

Everything in Layers 1–3 and 6 is built and run on live data, and Layers 4–5
are built and tested; see the README for the findings. The things that changed
the plan:

- **Missouri had no real-time NSSP data until June 2026**, and its 2024–25
  back-filled values are implausibly low. St. Louis is still the home metro,
  but the scored comparison relies on the four metros with real-time data, and
  "nowcasting with no official number" became its own experiment.
- **Wastewater is only honestly testable from April 2026.**
- **Weather adds nothing beyond season** in the baselines. That raises the bar
  for news: it must beat a seasonal model, not a flat mean.
- **The corpus is ready:** 196 candidate articles (173 with text). The next
  step is human labels.
