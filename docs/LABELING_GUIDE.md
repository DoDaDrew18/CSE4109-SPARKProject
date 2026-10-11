# Labeling guide: local news about respiratory illness

One guide for both human labelers and the LLM. The LLM receives this file
verbatim as its system prompt (`src/news/label.py`), so any edit here changes
the prompt hash and the LLM labels must be re-run. Keep it short and exact.

## What we are measuring

For each article we want to know: does it describe **current respiratory
illness activity in one of our study counties**, and if so, which illness and
how worrying it sounds. These labels become two weekly features per county
(`news_n`, `news_concern`), so every rule below is chosen to keep those
features honest.

## Study counties

| FIPS | County | Metro |
| --- | --- | --- |
| 29510 | St. Louis city | St. Louis |
| 29189 | St. Louis County | St. Louis |
| 29095 | Jackson County, MO | Kansas City |
| 17031 | Cook County, IL | Chicago |
| 18097 | Marion County, IN | Indianapolis |
| 47157 | Shelby County, TN | Memphis |
| 21111 | Jefferson County, KY | Louisville |

## Fields

Label each field from the article text only. Never use outside knowledge
of what the illness numbers actually were that week.

1. **relevant** (`y` / `n`). `y` only if all three hold:
   - the article is about respiratory illness (COVID-19, influenza, RSV, or
     "respiratory illness" in general) **activity** — cases, hospitalizations,
     ED visits, school absences, outbreaks, testing positivity, or a clear
     statement that activity is low/high/rising/falling;
   - that activity is in a study county (see **fips**);
   - it describes the current season, i.e. activity within about 4 weeks
     of the publish date.
   Otherwise `n`. When `n`: `illness = none`, `concern = 0`, `fips` may be
   left blank or set to `other`/`unclear`, `event_week` blank, no quotes needed.

2. **fips** — the county whose activity the article describes. One of the
   seven FIPS above, `other` (a named place outside the study counties, or
   national/state-wide only), or `unclear` (local, but cannot tell which
   county). A city name maps to its county: Chicago → 17031, Indianapolis →
   18097, Memphis → 47157, Louisville → 21111, Kansas City (MO) → 29095.
   "St. Louis" alone, from a St. Louis outlet, about the city's health
   department or city schools → 29510; "St. Louis County" or a
   county municipality (Clayton, Florissant, Kirkwood, Chesterfield...) →
   29189. If the article covers both St. Louis counties equally, pick the one
   named first. **Do not infer the county from the outlet alone**: the
   supporting quote must name the place (see Grounding).

3. **illness** — `covid`, `flu`, `rsv`, `respiratory_general`, or `none`.
   **Several allowed**, written with `|` (e.g. `flu|rsv`), when the article
   reports activity for each. Use `respiratory_general` only when no specific
   virus is named ("respiratory illnesses", "cold and flu season" without flu
   data counts as `flu`). `none` is used alone and only when relevant = `n`.

4. **concern** (0–3) — how bad the article says local activity is *now*.
   Score what is reported, not the tone of the writing.

   | Level | Meaning | Typical wording |
   | --- | --- | --- |
   | 0 | Low, falling, or over | "cases are declining", "season winding down", "levels remain low" |
   | 1 | Mention / routine | vaccine clinics, "flu season is here", routine weekly update with no direction |
   | 2 | Rising / warning | "cases doubled", "rising", "health officials urge", "above last year" |
   | 3 | Surge / strain | hospitals full or diverting, ED waits from respiratory patients, school closures, "highest in years" |

   If an article contains several levels, use the **highest level that
   describes the study county now**. Not-relevant articles are 0.

5. **event_week** — the Saturday (YYYY-MM-DD) ending the week the reported
   activity happened, only if the article clearly dates it to a different week
   than the publish week ("last week's data show..."). Otherwise blank.

6. **quote** — exact sentence(s) copied from the article that support the
   label. Humans give one quote supporting **concern** (and the county, if the
   same sentence names it). The LLM gives one quote per field:
   `fips_quote`, `illness_quote`, `concern_quote`, `event_quote`.

## Grounding rules (what counts as support)

- A quote must be copied **verbatim** from the article body or title. Fixing
  curly quotes, dashes, spacing and capitalization is fine; rewording is not.
- At least one full clause (4 words or more). A single word like "flu" is not
  a quote.
- `...` may join two parts of the **same paragraph**; each part must still
  be verbatim.
- The `fips_quote` must contain the county name, its main city, or a place in
  that county. If no sentence names the place, use `unclear`, not a guess.
- The quote must support the field it is attached to. A quote about
  vaccines does not support concern 3.
- No quote is needed for `relevant = n`, `illness = none`, `fips = unclear`,
  blank `event_week`, or concern 0 on a not-relevant article.
- If you cannot find a quote for a field, leave the field blank. A blank is
  better than an unsupported label: code drops every label whose quote is not
  found in the article.

## Worked examples

**Concern 0.** KSDK, Mar 2026: "Flu cases in St. Louis County fell for the
third straight week, the county health department said." → relevant y,
fips 29189, illness flu, concern 0.

**Concern 1.** St. Louis Public Radio, Oct 2025: "The St. Louis County
Department of Public Health will offer free flu shots at its Clayton clinic
on Saturday." → relevant y (season-start activity in a study county),
fips 29189, flu, concern 1. A vaccine-clinic PSA is never above 1.

**Concern 2.** WTHR, Dec 2025: "Marion County Public Health reports flu
positivity has doubled in two weeks and urges residents to get vaccinated."
→ y, 18097, flu, 2.

**Concern 3.** Commercial Appeal, Jan 2026: "Several Memphis hospitals
went on diversion Tuesday as emergency rooms filled with flu and RSV
patients." → y, 47157, flu|rsv, 3.

## Tricky cases

- **National story on a local station.** A CDC national flu update on
  KMOV's website with no local numbers → relevant n, fips other. If the same
  story adds "Locally, SSM Health says its St. Louis ERs are seeing more flu
  patients" → relevant y, quote that local sentence, fips 29510.
- **State-wide numbers.** "Missouri flu cases rose 40%" with no county →
  relevant n, fips other. Illinois data in a Chicago outlet is still `other`
  unless Cook County or Chicago is named.
- **Last-year retrospective.** "Last winter's tripledemic overwhelmed
  Louisville hospitals" in a September preview piece → relevant n (not
  current). If the piece also says "and this year cases are already rising in
  Louisville," label that current part.
- **Hard negatives.** Allergy season, wildfire smoke, measles, norovirus,
  pneumonia vaccine ads, hospital finance news, obituaries, and COVID policy or
  lawsuits with no activity → relevant n.
- **Two counties.** Pick the county the article mainly describes; mention
  the other in `notes` (humans only).
- **Hospital system, not county.** "BJC hospitals saw a spike" — BJC
  serves both St. Louis counties. Use the hospital's location if named
  (Barnes-Jewish is in the city → 29510); otherwise `unclear`.

## Human workflow

Copy `labels/human_template.csv` to `labels/human_labels.csv`, delete the two
example rows, and add one row per article you label. `labeler` is `human_a`
or `human_b`; on the 30 double-labeled articles, label independently and do
not compare until both are done. Disagreements are resolved by adding a third
row with `labeler = adjudicated`.
