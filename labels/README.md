# labels/

## articles.csv: the news article catalog (metadata only)

One row per candidate local news article. This file holds **no article text**
because the text is copyrighted. Text and HTML go to `raw/news/` (gitignored), and
you can rebuild them with `python -m src.news.fetch`.

Built by `python -m src.news.collect build` (merges GDELT, web-search and
hand-found candidates) and then updated in place by `python -m src.news.fetch`.

| Column | Meaning |
| --- | --- |
| `article_id` | First 12 hex chars of `sha1(normalize_url(url))`. See `src/news/collect.py`. It stays the same across http/https, `www.`, trailing slashes, `#anchors`, `/amp` and tracking parameters. |
| `url` | The article URL as found |
| `outlet` | Host without `www.` (e.g. `ksdk.com`, `chicago.suntimes.com`) |
| `published` | Publish date `YYYY-MM-DD`, local (US Central) calendar date. `date_source` says where it came from. |
| `metro` | Study metro key from `src/study.py`: `stl`, `kc`, `chi`, `ind`, `mem`, `lou` (the outlet's market, not necessarily the place in the story) |
| `fips_guess` | Starting county guess. It is set for single-county metros. For St. Louis it is set only when the text names just the city (`29510`) or just the county (`29189`); otherwise it is blank. Labelers make the real call. |
| `title` | Headline (from search, GDELT, or the page's `og:title`) |
| `found_via` | `gdelt` (DOC API), `search` (web search for past seasons), `manual` (hand-added with `collect add-url`) |
| `fetched_ok` | `True` if the page downloaded and yielded at least 80 words of main text. `False` covers paywalls, robots.txt disallows, 403/404 and video-only pages. Blank means not attempted yet. |
| `n_words` | Words of extracted main text |
| `dup_of` | Blank, or the `article_id` of the earliest copy of the same story (same normalized title, or six-word-shingle Jaccard of 0.6 or more). Copies are kept but should be labeled once. |
| `date_source` | `jsonld`, `meta:<tag>`, `itemprop` or `time_tag` come from page metadata (preferred). `search` means the date was read off the page by a human/agent during web search and the page could not be fetched here. `gdelt_seendate` means GDELT's crawl date was the only date available. |
| `gdelt_seendate` | Date GDELT first crawled the URL (UTC), if it came from GDELT |
| `stratum` | **Why the article was collected, not a label**: `respiratory` (surge-season respiratory coverage), `quiet` (off-season respiratory coverage), `hard_negative` (local health news that is not about respiratory illness, so labeler precision can be measured). It is blank for GDELT rows. Hide this column from labelers. |

Fetch outcomes per attempt (HTTP status, robots result, page date vs. GDELT
seendate gap) live in `raw/news/articles.parquet`.
