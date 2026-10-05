"""Run independent solver jobs concurrently, contiguous and ordered.

Parallelism here is deliberately conservative, because measurement does
not support an aggressive default. On a 15 angle NACA 2412 sweep the
whole chain finishes in about 0.4 s wall clock, so process start and
disk traffic are a large share of the cost. Measured speedup was 1.17x
at 2 workers, 1.68x at 4, and *slower than serial* at 8, where
contention outweighs the work being spread.

Two properties matter more than the speedup:

* Contiguous angles stay in one job. A solver chain warms each angle
  from the previous one, and splitting a chain costs about 46% more
  iterations in total, so angles are blocked rather than scattered.
* Results are returned in the order they were submitted, so a caller
  cannot tell a parallel run from a serial one.

The default worker count is 1, which runs the work inline and is what
every existing caller already does.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

WORKERS_ENV = "XFOIL_PARALLEL_WORKERS"

DEFAULT_WORKERS = 1
MAX_WORKERS = 8
"""Above this, measured throughput falls rather than rises."""


def resolve_workers(requested: int | None = None) -> int:
    """Worker count from the argument, the environment, or the default."""
    value = requested
    if value is None:
        raw = os.environ.get(WORKERS_ENV)
        if raw is None or not raw.strip():
            return DEFAULT_WORKERS
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(
                f"{WORKERS_ENV} must be an integer, got {raw!r}"
            ) from None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("Worker count must be an integer")
    if value < 1:
        raise ValueError("Worker count must be at least 1")
    return min(value, MAX_WORKERS)


def partition_contiguous(items: list, workers: int) -> list[list]:
    """Split into at most ``workers`` contiguous, near-equal blocks.

    Contiguity is the point: neighbouring angles share a solver chain,
    so an interleaved split would pay a warm start per angle instead of
    per block.
    """
    if workers < 1:
        raise ValueError("Worker count must be at least 1")
    if workers == 1 or len(items) <= 1:
        return [list(items)] if items else []
    count = min(workers, len(items))
    base, extra = divmod(len(items), count)
    blocks = []
    start = 0
    for index in range(count):
        size = base + (1 if index < extra else 0)
        if size:
            blocks.append(list(items[start : start + size]))
        start += size
    return blocks


def _run_one(payload: tuple) -> object:
    """Worker entry point. Must be importable and take one argument."""
    function, argument = payload
    return function(argument)


def map_blocks(function, blocks: list, workers: int) -> list:
    """Apply ``function`` to each block, preserving submission order.

    With one worker the work runs inline, so no process is created and
    behaviour is identical to a plain loop.
    """
    if workers < 1:
        raise ValueError("Worker count must be at least 1")
    if not blocks:
        return []
    if workers == 1 or len(blocks) == 1:
        return [function(block) for block in blocks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run_one, [(function, b) for b in blocks]))
