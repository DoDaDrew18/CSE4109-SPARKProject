"""FastAPI server: the map page, region lookup, and the streamed ask pipeline.

``POST /api/ask`` streams Server-Sent Events, one JSON object per event,
modeled on CalHacks 13's RunEvent protocol:

    region         counties in the selection, their outlines, the as-of date
    plan           which workers run and why, plus the LLM backend
    agent_status   queued -> running -> done | error, with a finding count
    agent_log      a worker's progress line
    layer          per-county values for the map (ED share, nowcast, Gi* class)
    finding        one validated finding (id F1, F2, ...)
    rejected       a finding the validator dropped, and why
    answer_delta   a chunk of the streamed answer
    answer_done    the citation check
    error / done
"""

from __future__ import annotations

import json
import logging
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Iterator

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from src.app import llm
from src.app.region import RegionError, counties, resolve
from src.app.validate import check_answer, check_finding, check_question
from src.app.workers import WORKERS, Context, Finding, run_worker

log = logging.getLogger(__name__)

__all__ = ["app", "run_ask", "parse_as_of", "FIRST_DATE"]

STATIC = Path(__file__).parent / "static"
FIRST_DATE = date(2024, 4, 20)      # first archived NSSP issue (epiweek 202416 ends Sat 2024-04-20)
WORKER_TIMEOUT = 60.0

app = FastAPI(title="SPARK Region Ask")


class AskBody(BaseModel):
    geometry: dict
    question: str
    as_of: str | None = None


class RegionBody(BaseModel):
    geometry: dict


def parse_as_of(text: str | None, today: date | None = None) -> pd.Timestamp:
    today = today or date.today()
    if not text:
        return pd.Timestamp(today)
    try:
        d = pd.Timestamp(text).date()
    except Exception as exc:
        raise ValueError(f"bad date {text!r}; use YYYY-MM-DD") from exc
    if d > today:
        raise ValueError("the date can't be in the future: this tool nowcasts, it doesn't forecast")
    if d < FIRST_DATE:
        raise ValueError(f"no archived NSSP data before {FIRST_DATE}")
    return pd.Timestamp(d)


def _outlines(fips: list[str]) -> dict:
    shapes = counties()
    sub = shapes[shapes["fips"].isin(fips)][["fips", "name", "state", "population", "geometry"]]
    return json.loads(sub.to_json())


def run_ask(geometry: dict, question: str, as_of: str | None = None,
            workers: dict | None = None, kind: str | None = None) -> Iterator[dict]:
    """The whole pipeline as a stream of event dicts (also used by the tests)."""
    workers = workers or WORKERS
    try:
        question = check_question(question)
        day = parse_as_of(as_of)
        region = resolve(geometry)
    except (ValueError, RegionError) as exc:
        yield {"type": "error", "message": str(exc)}
        return
    iso = day.date().isoformat()
    yield {"type": "region", "region": region.as_dict(), "as_of": iso, "outlines": _outlines(region.fips)}

    kind = kind or llm.backend()
    picks, reason = llm.plan(question, region.label, iso, kind)
    picks = [p for p in picks if p in workers]
    yield {"type": "plan", "workers": [{"id": p, "name": workers[p].name} for p in picks],
           "reason": reason, "llm": kind, "model": llm.model_name(kind)}

    events: queue.Queue = queue.Queue()
    findings: list[Finding] = []
    counter = iter(range(1, 10_000))

    def job(wid: str):
        w = workers[wid]
        events.put({"type": "agent_status", "worker": wid, "status": "running"})
        ctx = Context(region, day, question,
                      log=lambda m: events.put({"type": "agent_log", "worker": wid, "message": m}),
                      emit_layer=lambda layer: events.put({"type": "layer", "worker": wid, **layer}))
        t0 = time.monotonic()
        got = run_worker(w, ctx, WORKER_TIMEOUT)
        kept = 0
        for f in got:
            ok, why = check_finding(f, day)
            if ok:
                f.id = f"F{next(counter)}"
                findings.append(f)
                kept += 1
                events.put({"type": "finding", "finding": f.as_dict()})
            else:
                events.put({"type": "rejected", "worker": wid, "claim": f.claim[:120], "reason": why})
        events.put({"type": "agent_status", "worker": wid, "status": "done",
                    "findings": kept, "seconds": round(time.monotonic() - t0, 1)})

    for wid in picks:
        yield {"type": "agent_status", "worker": wid, "status": "queued"}
    with ThreadPoolExecutor(max_workers=len(picks) or 1) as pool:
        futures = [pool.submit(job, wid) for wid in picks]
        while not all(f.done() for f in futures) or not events.empty():
            try:
                yield events.get(timeout=0.2)
            except queue.Empty:
                continue

    findings.sort(key=lambda f: int(f.id[1:]))
    text = []
    for chunk in llm.synthesize(question, region.as_dict(), iso, findings, kind):
        if chunk:
            text.append(chunk)
            yield {"type": "answer_delta", "text": chunk}
    yield {"type": "answer_done", **check_answer("".join(text), findings)}
    yield {"type": "done", "n_findings": len(findings)}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config():
    kind = llm.backend()
    return {"llm": kind, "model": llm.model_name(kind), "today": date.today().isoformat(),
            "first_date": FIRST_DATE.isoformat(),
            "workers": [{"id": w.id, "name": w.name, "description": w.description} for w in WORKERS.values()]}


@app.post("/api/region")
def region(body: RegionBody):
    try:
        r = resolve(body.geometry)
    except RegionError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"region": r.as_dict(), "outlines": _outlines(r.fips)}


def _sse(events: Iterator[dict]) -> Iterator[str]:
    try:
        for e in events:
            yield f"data: {json.dumps(e, default=str)}\n\n"
    except Exception as exc:   # never leave the browser hanging on a half-open stream
        log.exception("ask failed")
        yield f"data: {json.dumps({'type': 'error', 'message': str(exc)[:200]})}\n\n"


@app.post("/api/ask")
def ask(body: AskBody):
    return StreamingResponse(_sse(run_ask(body.geometry, body.question, body.as_of)),
                             media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
