#!/usr/bin/env python3
"""Compatibility launcher. Use this script or ``python -m xfoil_mac``."""

# Copyright (c) 2026, 展一了
from xfoil_mac.cli import main, run

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(run())
