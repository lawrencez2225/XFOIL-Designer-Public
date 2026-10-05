"""Explicit transition and panel controls shared by every XFOIL workflow."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class SolverSettings:
    ncrit: float = 9.0
    xtr_top: float = 1.0
    xtr_bottom: float = 1.0
    panels: int = 240
    panel_bunching: float = 1.0
    te_le_ratio: float = 0.15

    def __post_init__(self):
        for name in (
            "ncrit",
            "xtr_top",
            "xtr_bottom",
            "panel_bunching",
            "te_le_ratio",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if not 0 < self.ncrit <= 20:
            raise ValueError("Ncrit must be greater than 0 and at most 20")
        if not all(0 <= v <= 1 for v in (self.xtr_top, self.xtr_bottom)):
            raise ValueError(
                "Transition positions must be between 0 and 1 chord"
            )
        if type(self.panels) is not int or not 40 <= self.panels <= 500:
            raise ValueError("Panel count must be an integer from 40 to 500")
        if (
            not 0.1 <= self.panel_bunching <= 3
            or not 0.01 <= self.te_le_ratio <= 1
        ):
            raise ValueError(
                "Panel bunching must be 0.1..3; TE/LE density ratio must "
                "be 0.01..1"
            )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_config(cls, config):
        return cls(**config.get("solver_settings", {}))

    def panel_commands(self):
        return [
            "PPAR",
            f"N {self.panels}",
            f"P {self.panel_bunching:g}",
            f"T {self.te_le_ratio:g}",
            "",
            "",
            "PANE",
        ]

    def transition_commands(self):
        return [
            "VPAR",
            f"N {self.ncrit:g}",
            f"XTR {self.xtr_top:g} {self.xtr_bottom:g}",
            "",
        ]
