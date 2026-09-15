"""Put the repository root on sys.path so tests can `import src.<module>`.

Keeps `python -m pytest tests/` working from a clean clone with no install
step and no editable-install ceremony.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
