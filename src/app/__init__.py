"""Region Ask: draw an area on a map, ask an outbreak question, get a sourced answer.

    python -m src.app            # http://127.0.0.1:8050

Adapted from the CalHacks 13 "Coordinate" app (select a region, agents fetch
real data, an orchestrator writes the answer), rebuilt on this study's own
loaders and its one rule: every number is read as it was public on the
chosen date. See ``src/app/README.md``.
"""
