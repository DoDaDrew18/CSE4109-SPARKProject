"""Hand-label news articles in the browser, under the rules of docs/LABELING_GUIDE.md.

Usage, from the repo root (article text first: ``python -m src.news.fetch``):

    python -m src.news.label_app assign                       # once; commit labels/assignments.csv
    python -m src.news.label_app serve --labeler human_a      # opens http://127.0.0.1:8765
    python -m src.news.label_app serve --labeler adjudicated  # resolve human_a vs human_b
    python -m src.news.label_app status

Why a tool and not the spreadsheet
----------------------------------
A label only helps the agreement report if it is complete, valid and
grounded, and a spreadsheet checks none of that until ``load_human_labels``
rejects the whole file. Here every save goes through the same
``schema.validate_label`` the LLM's output goes through, and the quote is
taken by selecting text in the article and checked with the same
``ground.find_quote`` that polices the LLM. Humans and the model are held to
one standard, so their agreement compares like with like.

What the labeler sees
---------------------
Title, outlet, publish date, a link to the original and the extracted body:
what a reader of the page saw. Never the catalog's ``stratum``,
``fips_guess`` or ``found_via`` (they say why an article was collected, which
would prime the label), never the LLM's labels and never another person's.
The one exception is ``--labeler adjudicated``, which shows human_a and
human_b side by side on the double-labeled articles they disagree on,
because resolving that disagreement is its whole job.

The order: ``labels/assignments.csv``
-------------------------------------
Everyone labels in one fixed order. First the double-labeled set (30 by
default), stratified by collection stratum and spread across metros in
proportion to the corpus, so the human-human kappa is measured on surges,
quiet weeks and hard negatives alike, on the same mix of places the LLM is
scored on. Then everything else, shuffled: the catalog is sorted by metro and
date, so stopping after 60 articles in catalog order would leave whole metros
unlabeled, while any prefix of a shuffled list is a fair sample. The seed is
fixed, and re-running ``assign`` only appends articles the file lacks, so the
double set never moves under a labeler's feet.

Where labels go
---------------
``labels/human/<labeler>.csv``, in the columns of ``labels/human_template.csv``
plus ``labeled_at`` and ``seconds`` (time on the article, for the cost
comparison with the LLM). One file per person, so two people labeling on two
laptops never edit the same file; ``schema.load_human_labels`` reads the
directory, so nothing downstream changes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd

from src.news.collect import CATALOG_PATH, read_catalog
from src.news.ground import find_quote, names_county
from src.news.schema import HUMAN_COLUMNS, HUMAN_DIR, HUMAN_LABELERS, LABEL_FIELDS, validate_label
from src.study import METROS, STUDY_FIPS

__all__ = ["ASSIGNMENTS_PATH", "FILE_COLUMNS", "eligible", "make_assignments", "read_assignments",
           "label_file", "read_labels", "check_label", "save_label", "disagreements", "LabelApp",
           "make_server", "status"]

ASSIGNMENTS_PATH = Path("labels/assignments.csv")
TEXT_DIR = Path("raw/news/text")
GUIDE_PATH = Path(__file__).resolve().parents[2] / "docs" / "LABELING_GUIDE.md"
PAGE_PATH = Path(__file__).with_name("label_app.html")
FILE_COLUMNS: tuple[str, ...] = (*HUMAN_COLUMNS, "labeled_at", "seconds")
DOUBLE = 30
SEED = 4109
PORT = 8765
MAX_BODY = 1 << 20
_ID = re.compile(r"^[0-9a-f]{12}$")
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


# -- the labeling order ------------------------------------------------------

def eligible(catalog: pd.DataFrame) -> pd.DataFrame:
    """Articles worth a person's time: text fetched, one copy per syndicated story."""
    fetched = catalog["fetched_ok"].astype(str).str.strip().str.lower() == "true"
    original = catalog["dup_of"].fillna("").astype(str).str.strip() == ""
    return catalog[fetched & original]


def _allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Split ``n`` across strata in proportion to size (largest remainder),
    with at least one per stratum whenever ``n`` covers every stratum."""
    sizes = {k: v for k, v in sizes.items() if v > 0}
    n = min(n, sum(sizes.values()))
    if n <= 0:
        return {k: 0 for k in sizes}
    quota = {k: n * v / sum(sizes.values()) for k, v in sizes.items()}
    alloc = {k: int(q) for k, q in quota.items()}
    if n >= len(sizes):
        alloc = {k: max(a, 1) for k, a in alloc.items()}
    by_remainder = sorted(sizes, key=lambda k: (quota[k] - int(quota[k]), k), reverse=True)
    while sum(alloc.values()) < n:
        for k in by_remainder:
            if sum(alloc.values()) < n and alloc[k] < sizes[k]:
                alloc[k] += 1
    while sum(alloc.values()) > n:
        for k in reversed(by_remainder):
            if sum(alloc.values()) > n and alloc[k] > 1:
                alloc[k] -= 1
    return alloc


def _systematic(frame: pd.DataFrame, k: int, rng: np.random.Generator) -> list[str]:
    """``k`` articles at evenly spaced positions (random start) of the stratum
    listed metro by metro, each metro shuffled: every metro gets its share of
    the sample to within one article. St. Louis, the metro with the hardest
    county call (city or county?), keeps its weight instead of being diluted
    to one-sixth."""
    ids = [str(a) for _, g in frame.groupby("metro", sort=True)
           for a in rng.permutation(g["article_id"].to_numpy())]
    if k <= 0:
        return []
    if k >= len(ids):
        return ids
    step = len(ids) / k
    start = rng.uniform(0, step)
    return [ids[int(start + i * step)] for i in range(k)]


def make_assignments(catalog: pd.DataFrame, n_double: int = DOUBLE, seed: int = SEED,
                     existing: pd.DataFrame | None = None) -> pd.DataFrame:
    """The labeling order: one row per article, ``article_id, position, double``.

    With ``existing``, its rows keep their positions and flags and only
    eligible articles it lacks are appended (shuffled, single-labeled), so
    re-running after the catalog grows never reshuffles work under way.
    """
    rng = np.random.default_rng(seed)
    pool = eligible(catalog).sort_values("article_id")       # independent of catalog order
    pool = pool.assign(stratum=pool["stratum"].fillna("").replace("", "gdelt"))
    shuffled = [str(a) for a in rng.permutation(pool["article_id"].to_numpy())]

    if existing is not None and len(existing):
        known = set(existing["article_id"])
        new = [a for a in shuffled if a not in known]
        start = int(existing["position"].max()) + 1
        tail = pd.DataFrame({"article_id": new, "position": range(start, start + len(new)),
                             "double": False})
        keep = existing[["article_id", "position", "double"]]
        return keep if tail.empty else pd.concat([keep, tail], ignore_index=True)

    alloc = _allocate(pool.groupby("stratum").size().to_dict(), n_double)
    double = []
    for stratum in sorted(alloc):
        double += _systematic(pool[pool["stratum"] == stratum], alloc[stratum], rng)
    double = [str(a) for a in rng.permutation(double)]
    chosen = set(double)
    order = double + [a for a in shuffled if a not in chosen]
    return pd.DataFrame({"article_id": order, "position": range(1, len(order) + 1),
                         "double": [a in chosen for a in order]})


def read_assignments(path=ASSIGNMENTS_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame["position"] = frame["position"].astype(int)
    frame["double"] = frame["double"].str.strip().str.lower().isin({"true", "1", "y", "yes"})
    return frame.sort_values("position").reset_index(drop=True)


# -- one file per labeler ----------------------------------------------------------

def label_file(labeler: str, root=HUMAN_DIR) -> Path:
    if labeler not in HUMAN_LABELERS:
        raise ValueError(f"labeler must be one of {HUMAN_LABELERS}, got {labeler!r}")
    return Path(root) / f"{labeler}.csv"


def read_labels(labeler: str, root=HUMAN_DIR) -> pd.DataFrame:
    """This labeler's saved rows, all as strings (empty frame if none yet)."""
    path = label_file(labeler, root)
    if not path.exists():
        return pd.DataFrame(columns=list(FILE_COLUMNS), dtype=str)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in FILE_COLUMNS:
        if col not in frame:
            frame[col] = ""
    return frame[list(FILE_COLUMNS)]


def _write_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write beside the target, then rename: a crash mid-write never leaves a
    half-written labels file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def check_label(form: dict, text: str, labeler: str) -> tuple[dict, list[str], list[str]]:
    """Validate one submitted label against the guide and the article text.

    Returns (row to store, errors, warnings). Errors block the save; warnings
    are shown and the label is kept. A relevant article must be complete and
    carry a quote; a not-relevant one gets the guide's defaults (illness
    ``none``, concern 0).
    """
    raw = {k: str(form.get(k) or "").strip() for k in ("article_id", *LABEL_FIELDS, "quote", "notes")}
    if isinstance(form.get("illness"), list):
        raw["illness"] = "|".join(str(i) for i in form["illness"])
    clean, errors = validate_label(raw)
    warnings: list[str] = []

    relevant = clean["relevant"]
    if relevant is None and not raw["relevant"]:
        errors.append("relevant: choose y or n")
    if relevant is True:
        for field, what in (("fips", "the study county"), ("illness", "at least one illness"),
                            ("concern", "a concern level 0-3")):
            if clean[field] is None and not raw[field]:
                errors.append(f"{field}: pick {what}")
        if not raw["quote"]:
            errors.append("quote: a relevant article needs a quote supporting its concern level")
    if relevant is False:
        clean["illness"] = clean["illness"] or "none"
        clean["concern"] = 0 if clean["concern"] is None else clean["concern"]

    if raw["quote"]:
        match = find_quote(raw["quote"], text)
        if match.method == "too_short":
            errors.append("quote: needs at least one full clause (4+ words)")
        elif not match.found:
            errors.append("quote: not found in the article; select it in the text instead of typing it")
        elif clean["fips"] in STUDY_FIPS and not names_county(raw["quote"], clean["fips"]):
            warnings.append(f"the quote does not name {clean['fips']} or a place in it; "
                            "fine if the article names the place elsewhere")

    try:
        seconds = int(min(max(float(form.get("seconds") or 0), 0), 86_400))
    except (TypeError, ValueError):
        seconds = 0
    row = {
        "article_id": clean["article_id"], "labeler": labeler,
        "relevant": "" if relevant is None else ("y" if relevant else "n"),
        "fips": clean["fips"] or "", "illness": clean["illness"] or "",
        "concern": "" if clean["concern"] is None else str(int(clean["concern"])),
        "event_week": "" if clean["event_week"] is None else f"{clean['event_week']:%Y-%m-%d}",
        "quote": raw["quote"], "notes": raw["notes"], "is_example": "",
        "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": str(seconds),
    }
    return row, errors, warnings


def save_label(row: dict, root=HUMAN_DIR) -> Path:
    """Insert or replace this labeler's row for the article; rows sorted by id
    so an edited label shows up in git as a one-line change."""
    labels = read_labels(row["labeler"], root)
    labels = labels[labels["article_id"] != row["article_id"]]
    new = pd.DataFrame([row], columns=list(FILE_COLUMNS))
    labels = new if labels.empty else pd.concat([labels, new], ignore_index=True)
    path = label_file(row["labeler"], root)
    _write_atomic(labels.sort_values("article_id"), path)
    return path


def disagreements(a: dict, b: dict) -> list[str]:
    """Fields where two labels of one article differ in a way the agreement
    report scores: county, illness and concern only matter if either side
    calls the article relevant."""
    ca, _ = validate_label(dict(a))
    cb, _ = validate_label(dict(b))
    fields = ["relevant"] if ca["relevant"] != cb["relevant"] else []
    if ca["relevant"] is True or cb["relevant"] is True:
        fields += [f for f in ("fips", "illness", "concern") if ca[f] != cb[f]]
    if ca["event_week"] != cb["event_week"]:
        fields.append("event_week")
    return fields


# -- the app ---------------------------------------------------------------------

def _county_names() -> dict[str, str]:
    return {f: f"{name} ({m.name})" for m in METROS.values() for f, name in m.counties.items()}


class LabelApp:
    """Everything the web page asks for, as plain methods (tested without HTTP)."""

    def __init__(self, labeler: str, root="."):
        label_file(labeler)                                 # validates the name
        self.labeler = labeler
        self.root = Path(root)
        self.text_dir = self.root / TEXT_DIR
        self.labels_dir = self.root / HUMAN_DIR
        assignments = self.root / ASSIGNMENTS_PATH
        if not assignments.exists():
            raise FileNotFoundError(f"{assignments} not found: run `python -m src.news.label_app "
                                    f"assign` once and commit it, so everyone labels in one order")
        self.assignments = read_assignments(assignments)
        catalog = read_catalog(self.root / CATALOG_PATH)
        self.catalog = {r["article_id"]: r for r in catalog.to_dict("records")}
        self.lock = threading.Lock()

    # -- reading -----------------------------------------------------------
    def text(self, article_id: str) -> str | None:
        path = self.text_dir / f"{article_id}.txt"
        return path.read_text(encoding="utf-8") if path.exists() else None

    def _labels(self, labeler: str) -> dict[str, dict]:
        return {r["article_id"]: r for r in read_labels(labeler, self.labels_dir).to_dict("records")}

    def _to_adjudicate(self) -> set[str]:
        """Double-labeled articles both humans finished and disagree on, plus
        any already adjudicated (so they can be reviewed)."""
        a, b = self._labels("human_a"), self._labels("human_b")
        double = set(self.assignments.loc[self.assignments["double"], "article_id"])
        todo = {aid for aid in double & a.keys() & b.keys() if disagreements(a[aid], b[aid])}
        return todo | (double & self._labels("adjudicated").keys())

    def queue(self) -> pd.DataFrame:
        if self.labeler != "adjudicated":
            return self.assignments
        return self.assignments[self.assignments["article_id"].isin(self._to_adjudicate())]

    def state(self) -> dict:
        queue, done = self.queue(), self._labels(self.labeler)
        items = []
        for row in queue.itertuples(index=False):
            meta = self.catalog.get(row.article_id, {})
            items.append({"article_id": row.article_id, "position": int(row.position),
                          "double": bool(row.double), "title": meta.get("title", ""),
                          "outlet": meta.get("outlet", ""), "published": meta.get("published", ""),
                          "done": row.article_id in done,
                          "has_text": (self.text_dir / f"{row.article_id}.txt").exists()})
        double = [i for i in items if i["double"]]
        return {"labeler": self.labeler, "adjudicating": self.labeler == "adjudicated",
                "items": items, "counties": _county_names(),
                "counts": {"done": sum(i["done"] for i in items), "total": len(items),
                           "double_done": sum(i["done"] for i in double),
                           "double_total": len(double),
                           "with_text": sum(i["has_text"] for i in items)}}

    def article(self, article_id: str) -> dict | None:
        if not _ID.match(article_id or "") or article_id not in set(self.queue()["article_id"]):
            return None
        meta = self.catalog.get(article_id, {})
        row = self.assignments.loc[self.assignments["article_id"] == article_id].iloc[0]
        out = {"article_id": article_id, "position": int(row["position"]),
               "double": bool(row["double"]), "title": meta.get("title", ""),
               "outlet": meta.get("outlet", ""), "published": meta.get("published", ""),
               "url": meta.get("url", ""), "text": self.text(article_id),
               "label": self._labels(self.labeler).get(article_id)}
        if self.labeler == "adjudicated":
            a = self._labels("human_a").get(article_id)
            b = self._labels("human_b").get(article_id)
            out["others"] = {"human_a": a, "human_b": b,
                             "differ": disagreements(a, b) if a and b else []}
        return out

    # -- writing -----------------------------------------------------------
    def check(self, payload: dict) -> dict:
        """Live feedback on a quote while the labeler is still editing."""
        text = self.text(str(payload.get("article_id", ""))) if _ID.match(
            str(payload.get("article_id", ""))) else None
        quote = str(payload.get("quote") or "")
        if text is None or not quote.strip():
            return {"found": False, "method": "missing", "names_county": None}
        match = find_quote(quote, text)
        fips = str(payload.get("fips") or "")
        return {"found": match.found, "method": match.method, "score": match.score,
                "names_county": names_county(quote, fips) if fips in STUDY_FIPS else None}

    def save(self, payload: dict) -> dict:
        article_id = str(payload.get("article_id", ""))
        if not _ID.match(article_id) or article_id not in set(self.queue()["article_id"]):
            return {"ok": False, "errors": [f"unknown article {article_id!r}"], "warnings": []}
        text = self.text(article_id)
        if text is None:
            return {"ok": False, "errors": ["no article text on this machine: run "
                                            "`python -m src.news.fetch` first"], "warnings": []}
        row, errors, warnings = check_label(payload, text, self.labeler)
        if errors:
            return {"ok": False, "errors": errors, "warnings": warnings}
        with self.lock:
            save_label(row, self.labels_dir)
        return {"ok": True, "errors": [], "warnings": warnings, "next": self._next(article_id)}

    def _next(self, article_id: str) -> str | None:
        """The next unlabeled article with text, after this one, wrapping around."""
        ids = list(self.queue()["article_id"])
        done = self._labels(self.labeler)
        start = ids.index(article_id) + 1 if article_id in ids else 0
        for aid in ids[start:] + ids[:start]:
            if aid not in done and (self.text_dir / f"{aid}.txt").exists():
                return aid
        return None


# -- HTTP ------------------------------------------------------------------------

def make_server(app: LabelApp, host: str = "127.0.0.1", port: int = PORT) -> ThreadingHTTPServer:
    """A local server for one labeler. Localhost only; POSTs must be JSON and
    the Host header local, so another web page cannot write labels."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "SPARK-labeler"

        def log_message(self, *args):            # keep the terminal quiet
            pass

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200) -> None:
            self._send(status, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

        def _local(self) -> bool:
            return urlsplit("//" + (self.headers.get("Host") or "")).hostname in _LOCAL_HOSTS

        def do_GET(self):
            if not self._local():
                return self._json({"error": "forbidden"}, 403)
            url = urlsplit(self.path)
            if url.path == "/":
                return self._send(200, PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
            if url.path == "/guide":
                return self._send(200, GUIDE_PATH.read_bytes(), "text/plain; charset=utf-8")
            if url.path == "/api/state":
                return self._json(app.state())
            if url.path == "/api/article":
                found = app.article(parse_qs(url.query).get("id", [""])[0])
                return self._json(found) if found else self._json({"error": "not found"}, 404)
            return self._json({"error": "not found"}, 404)

        def do_POST(self):
            if not self._local():
                return self._json({"error": "forbidden"}, 403)
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                return self._json({"error": "JSON only"}, 415)
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._json({"error": "too large"}, 413)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._json({"error": "bad JSON"}, 400)
            if not isinstance(payload, dict):
                return self._json({"error": "expected an object"}, 400)
            path = urlsplit(self.path).path
            if path == "/api/check":
                return self._json(app.check(payload))
            if path == "/api/save":
                return self._json(app.save(payload))
            return self._json({"error": "not found"}, 404)

    return ThreadingHTTPServer((host, port), Handler)


# -- status ----------------------------------------------------------------------

def status(root=".") -> list[str]:
    """One line per labeler plus the human-human overlap; for the terminal."""
    root = Path(root)
    assignments = read_assignments(root / ASSIGNMENTS_PATH)
    ids = set(assignments["article_id"])
    double = set(assignments.loc[assignments["double"], "article_id"])
    text = {a for a in ids if (root / TEXT_DIR / f"{a}.txt").exists()}
    lines = [f"{len(ids)} articles assigned ({len(double)} double-labeled); "
             f"{len(text)} have text on this machine"
             + ("" if len(text) == len(ids) else " -- run `python -m src.news.fetch` for the rest")]
    labels = {}
    for labeler in HUMAN_LABELERS:
        done = read_labels(labeler, root / HUMAN_DIR)
        labels[labeler] = {r["article_id"]: r for r in done.to_dict("records")}
        if labeler != "adjudicated" or len(done):
            lines.append(f"  {labeler:<12} {len(done):>4} labeled, double set "
                         f"{len(double & labels[labeler].keys())}/{len(double)}")
    a, b = labels["human_a"], labels["human_b"]
    both = double & a.keys() & b.keys()
    split = sorted(aid for aid in both if disagreements(a[aid], b[aid]))
    pending = [aid for aid in split if aid not in labels["adjudicated"]]
    lines.append(f"  both humans on {len(both)} double-labeled articles; disagree on {len(split)}, "
                 f"{len(pending)} not yet adjudicated")
    return lines


# -- CLI ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Hand-label news articles in the browser.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("assign", help="fix the labeling order -> labels/assignments.csv")
    a.add_argument("--double", type=int, default=DOUBLE, help="size of the double-labeled set")
    a.add_argument("--seed", type=int, default=SEED)
    a.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    a.add_argument("--out", type=Path, default=ASSIGNMENTS_PATH)
    a.add_argument("--reshuffle", action="store_true",
                   help="start over instead of appending new articles (changes the double set!)")
    s = sub.add_parser("serve", help="label in the browser")
    s.add_argument("--labeler", required=True, choices=HUMAN_LABELERS)
    s.add_argument("--port", type=int, default=PORT)
    s.add_argument("--no-browser", action="store_true")
    s.add_argument("--root", type=Path, default=Path("."),
                   help="folder holding labels/ and raw/news/text/ (default: repo root)")
    sub.add_parser("status", help="progress per labeler")
    args = parser.parse_args(argv)

    if args.cmd == "assign":
        existing = None if args.reshuffle or not args.out.exists() else read_assignments(args.out)
        out = make_assignments(read_catalog(args.catalog), args.double, args.seed, existing)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        added = len(out) - (0 if existing is None else len(existing))
        print(f"{len(out)} articles ({int(out['double'].sum())} double-labeled, {added} new) "
              f"-> {args.out}; commit it so every labeler uses the same order")
    elif args.cmd == "status":
        if not ASSIGNMENTS_PATH.exists():
            raise SystemExit(f"error: {ASSIGNMENTS_PATH} not found; run "
                             f"`python -m src.news.label_app assign` first")
        print("\n".join(status()))
    else:
        try:
            app = LabelApp(args.labeler, args.root)
            server = make_server(app, port=args.port)
        except FileNotFoundError as exc:
            raise SystemExit(f"error: {exc}")
        except OSError as exc:
            raise SystemExit(f"error: port {args.port} is busy ({exc}); try --port {args.port + 1}")
        url = f"http://127.0.0.1:{args.port}/"
        counts = app.state()["counts"]
        print(f"labeling as {args.labeler}: {counts['done']}/{counts['total']} done, "
              f"{counts['with_text']} with text. Open {url}  (Ctrl+C to stop)")
        if not args.no_browser:
            threading.Timer(0.5, webbrowser.open, args=(url,)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped; labels are saved after every article")
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
