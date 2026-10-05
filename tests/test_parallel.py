"""Concurrent blocks stay contiguous, ordered, and off by default."""

import os
import unittest
from unittest.mock import patch

from xfoil_mac.parallel import (
    DEFAULT_WORKERS,
    MAX_WORKERS,
    WORKERS_ENV,
    map_blocks,
    partition_contiguous,
    resolve_workers,
)


def double(block):
    """Module level so a process pool can import it."""
    return [value * 2 for value in block]


class ResolveWorkersTest(unittest.TestCase):
    def test_default_is_serial_when_nothing_is_set(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_workers(), DEFAULT_WORKERS)
            self.assertEqual(DEFAULT_WORKERS, 1)

    def test_explicit_argument_wins_over_environment(self):
        with patch.dict(os.environ, {WORKERS_ENV: "4"}):
            self.assertEqual(resolve_workers(2), 2)

    def test_environment_is_read_when_no_argument_is_given(self):
        with patch.dict(os.environ, {WORKERS_ENV: "3"}):
            self.assertEqual(resolve_workers(), 3)

    def test_blank_environment_falls_back_to_the_default(self):
        with patch.dict(os.environ, {WORKERS_ENV: "  "}):
            self.assertEqual(resolve_workers(), DEFAULT_WORKERS)

    def test_worker_count_is_capped_at_the_measured_limit(self):
        self.assertEqual(resolve_workers(64), MAX_WORKERS)

    def test_non_integer_environment_is_rejected(self):
        with patch.dict(os.environ, {WORKERS_ENV: "many"}):
            with self.assertRaises(ValueError):
                resolve_workers()

    def test_zero_and_negative_are_rejected(self):
        for value in (0, -1):
            with self.assertRaises(ValueError):
                resolve_workers(value)

    def test_cap_exists_because_more_workers_measured_slower(self):
        """A 15 angle sweep was slower at 8 workers than serial.

        The cap encodes that measurement so a caller cannot opt into a
        configuration this project has measured to regress.
        """
        self.assertEqual(resolve_workers(MAX_WORKERS + 1), MAX_WORKERS)
        self.assertLessEqual(MAX_WORKERS, 8)


class PartitionTest(unittest.TestCase):
    def test_single_worker_keeps_everything_in_one_block(self):
        self.assertEqual(partition_contiguous([1, 2, 3], 1), [[1, 2, 3]])

    def test_blocks_are_contiguous_and_cover_the_input(self):
        items = list(range(11))
        blocks = partition_contiguous(items, 3)
        flat = [value for block in blocks for value in block]
        self.assertEqual(flat, items)
        for block in blocks:
            self.assertEqual(
                block, list(range(block[0], block[0] + len(block)))
            )

    def test_sizes_differ_by_at_most_one(self):
        sizes = [len(b) for b in partition_contiguous(list(range(11)), 4)]
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_more_workers_than_items_gives_one_block_each(self):
        blocks = partition_contiguous([7, 8], 8)
        self.assertEqual(blocks, [[7], [8]])

    def test_empty_input_gives_no_blocks(self):
        self.assertEqual(partition_contiguous([], 4), [])

    def test_zero_workers_is_rejected(self):
        with self.assertRaises(ValueError):
            partition_contiguous([1], 0)


class MapBlocksTest(unittest.TestCase):
    def test_one_worker_runs_inline_and_keeps_order(self):
        blocks = [[1, 2], [3], [4, 5]]
        self.assertEqual(map_blocks(double, blocks, 1), [[2, 4], [6], [8, 10]])

    def test_parallel_run_matches_the_serial_result(self):
        blocks = partition_contiguous(list(range(8)), 4)
        self.assertEqual(
            map_blocks(double, blocks, 1), map_blocks(double, blocks, 4)
        )

    def test_parallel_run_preserves_submission_order(self):
        blocks = [[1], [2], [3], [4]]
        self.assertEqual(map_blocks(double, blocks, 4), [[2], [4], [6], [8]])

    def test_no_blocks_gives_no_results(self):
        self.assertEqual(map_blocks(double, [], 4), [])

    def test_zero_workers_is_rejected(self):
        with self.assertRaises(ValueError):
            map_blocks(double, [[1]], 0)

    def test_a_single_block_needs_no_process(self):
        self.assertEqual(map_blocks(double, [[1, 2]], 4), [[2, 4]])


if __name__ == "__main__":
    unittest.main()
