"""MLX Multi-Paradigm Aero-Structural MDO & Bionic Wing-Box Engine for XFOIL-Designer-Public."""

from __future__ import annotations

import sys
from pathlib import Path

_MDO_DIR = Path(__file__).resolve().parent
if str(_MDO_DIR) not in sys.path:
    sys.path.insert(0, str(_MDO_DIR))
