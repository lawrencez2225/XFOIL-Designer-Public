"""CL sequences must follow lift units and match explicit target lists."""

import unittest
from unittest.mock import patch

from xfoil_mac.cli import build_parser
from xfoil_mac.data import cl_values
from xfoil_mac.runtime import discover_app
from xfoil_mac.ui.commands import handle


class LiftSequenceTest(unittest.TestCase):
    def test_sub_millistep_matches_explicit_cli_targets(self):
        parser = build_parser()
        targets = []
        for options in (
            ["--cseq", "0", "0.001", "0.0005"],
            ["--cl-values", "0", "0.0005", "0.001"],
        ):
            args = parser.parse_args(["--lift", *options])
            with patch("xfoil_mac.ui.commands.run_job") as run_job:
                handle(discover_app(), args, parser)
            targets.append(run_job.call_args.args[1]["targets"])
        self.assertEqual(targets[0], [0, 0.0005, 0.001])
        self.assertEqual(targets[0], targets[1])

    def test_decimal_endpoint_descending_and_single_target(self):
        self.assertEqual(cl_values(0.2, 0.8, 0.2), [0.2, 0.4, 0.6, 0.8])
        self.assertEqual(cl_values(0.8, 0.2, -0.2), [0.8, 0.6, 0.4, 0.2])
        self.assertEqual(cl_values(0.5, 0.5, 0.1), [0.5])
        self.assertEqual(cl_values(0, 0.5, 0.2), [0, 0.2, 0.4])

    def test_invalid_sequences_are_rejected_in_cl_units(self):
        for start, end, step in (
            (0, 1, 0),
            (0, 1, -0.1),
            (1, 0, 0.1),
            (-6, 0, 1),
            (0, 6, 1),
            (0, 1, 0.001),
            (float("nan"), 1, 0.1),
            (0, float("inf"), 1),
        ):
            with self.subTest(start=start, end=end, step=step):
                with self.assertRaisesRegex(ValueError, "CL"):
                    cl_values(start, end, step)


if __name__ == "__main__":
    unittest.main()
