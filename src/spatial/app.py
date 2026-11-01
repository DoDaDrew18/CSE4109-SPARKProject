"""Demo II: type a question, get the map. A local web page around ``src.spatial.query``.

    python -m src.spatial.app                    # http://127.0.0.1:8766
    python -m src.spatial.app --parser rules     # never call an LLM

The data and county shapes load once at start (a few seconds); each question
then takes about three seconds, most of it the 9,999-permutation cluster
test. The answer panel inside each map says how the question was read and by
which parser, so the audience can see when the LLM was not involved.

Same safety rules as the labeling app: it listens on localhost only, POSTs
must be JSON, and the Host header must be local, so no other web page can
drive it.
"""

from __future__ import annotations

import argparse
import itertools
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from src.spatial import query as Q
from src.spatial.query_map import render

PORT = 8766
PAGE_PATH = Path(__file__).with_name("app.html")
MAX_BODY = 64 * 1024
MAX_MAPS = 30
EXAMPLES = (
    "Where was flu rising around St. Louis at Christmas 2025?",
    "Flu hot spots at the peak of the 2025-26 season",
    "RSV in the Midwest at the peak of the 2024-25 season",
    "Where is COVID rising right now?",
    "Kansas City flu peak",
    "Respiratory illness in Cook County, IL on Jan 10 2026",
)
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class MapApp:
    """Holds the data and the rendered maps; plain methods, tested without HTTP."""

    def __init__(self, parser: str = "auto", values=None, shapes=None, source: str = "",
                 permutations: int = 9999, **answer_kwargs):
        if values is None:
            values, source = Q.load_values()
        self.values, self.source = values, source
        self.shapes = Q.load_shapes() if shapes is None else shapes
        self.gaz = Q.Gazetteer(self.shapes)
        self.parser, self.client = parser, None
        if parser == "auto":
            self.parser, self.client = Q._auto_client()
        elif parser in ("ollama", "anthropic"):
            self.client = Q._client(parser)
        self.permutations = permutations
        self.answer_kwargs = answer_kwargs
        self.maps: dict[str, bytes] = {}
        self._ids = itertools.count(1)
        self.lock = threading.Lock()

    def status(self) -> dict:
        weeks = self.values["week_end"]
        return {"parser": self.parser, "source": self.source, "examples": list(EXAMPLES),
                "first_week": f"{weeks.min():%Y-%m-%d}", "latest_week": f"{weeks.max():%Y-%m-%d}"}

    def ask(self, question: str) -> dict:
        question = " ".join(str(question or "").split())[:300]
        if not question:
            return {"error": "type a question"}
        with self.lock:
            ans = Q.ask(question, self.values, self.shapes, self.gaz, self.source, self.parser,
                        client=self.client, permutations=self.permutations, **self.answer_kwargs)
            html = render(ans, self.gaz).get_root().render().encode("utf-8")
            map_id = str(next(self._ids))
            self.maps[map_id] = html
            for old in list(self.maps)[:-MAX_MAPS]:
                del self.maps[old]
        text = Q.summary_lines(ans, self.gaz)
        return {"id": map_id, "read_as": ans.query.describe(), "headline": text["headline"],
                "notes": text["notes"], "week_end": f"{ans.query.week_end:%Y-%m-%d}",
                "place": self.gaz.label(ans.query.place)}


def make_server(app: MapApp, host: str = "127.0.0.1", port: int = PORT) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        server_version = "SPARK-map"

        def log_message(self, *args):
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
            path = urlsplit(self.path).path
            if path == "/":
                return self._send(200, PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/status":
                return self._json(app.status())
            if path.startswith("/map/") and path[5:] in app.maps:
                return self._send(200, app.maps[path[5:]], "text/html; charset=utf-8")
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
            if urlsplit(self.path).path != "/api/ask" or not isinstance(payload, dict):
                return self._json({"error": "not found"}, 404)
            try:
                return self._json(app.ask(payload.get("question", "")))
            except Exception as exc:   # noqa: BLE001 - show the failure in the page, keep serving
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Ask the county map questions in a browser.")
    p.add_argument("--parser", choices=("auto", "rules", "ollama", "anthropic"), default="auto")
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--permutations", type=int, default=9999)
    args = p.parse_args(argv)
    print("loading CDC table and county shapes...")
    app = MapApp(args.parser, permutations=args.permutations)
    try:
        server = make_server(app, port=args.port)
    except OSError as exc:
        raise SystemExit(f"error: port {args.port} is busy ({exc}); try --port {args.port + 1}")
    url = f"http://127.0.0.1:{args.port}/"
    s = app.status()
    print(f"data {s['first_week']} .. {s['latest_week']}; questions read by: {s['parser']}. "
          f"Open {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
