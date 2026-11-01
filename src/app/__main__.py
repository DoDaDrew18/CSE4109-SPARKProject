"""``python -m src.app [--port 8050] [--host 127.0.0.1]``: serve Region Ask locally."""

from __future__ import annotations

import argparse
import logging


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m src.app", description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--reload", action="store_true", help="restart on code changes (development)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    import uvicorn

    from src.app.llm import backend, model_name
    kind = backend()
    print(f"Region Ask on http://{args.host}:{args.port}  (answers: {kind}, {model_name(kind)})")
    uvicorn.run("src.app.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
