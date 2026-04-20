"""Pytest path bootstrap for hotpotqa tests.

Allows running tests from repository roots like `speculative-action/`
without manually exporting PYTHONPATH.
"""

import sys
from pathlib import Path


HOTPOTQA_ROOT = Path(__file__).resolve().parents[1]
if str(HOTPOTQA_ROOT) not in sys.path:
    sys.path.insert(0, str(HOTPOTQA_ROOT))
