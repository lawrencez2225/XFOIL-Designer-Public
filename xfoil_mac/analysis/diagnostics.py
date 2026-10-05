"""Evidence-based convergence explanations; no inference of physical stall."""

import json
import re
from pathlib import Path

from ..data import atomic_json, cp_name, read_pairs, row_for_alpha
from ..results import write_records
from .boundary_layer import boundary_layer_error
from .report import report_page


def log_evidence(path: Path, status: str, alpha: float | None = None):
    text = path.read_text(errors="replace") if path.is_file() else ""
    if alpha is not None:
        # A log includes warm-up angles and sometimes a whole sweep. Use the
        # final solve of this angle, not a later point's residual or failure.
        matching = []
        for block in text.split("Solving BL system ...")[1:]:
            angles = re.findall(r"\ba\s*=\s*([-+\d.]+)", block)
            if angles and abs(float(angles[-1]) - alpha) <= 0.00051:
                matching.append(block)
        text = matching[-1] if matching else ""
    residuals = [
        float(v.replace("D", "E"))
        for v in re.findall(r"rms:\s*([-+.\d]+[EeDd][-+]?\d+)", text)
    ]
    if status in {"timeout", "interrupted", "solver_error"}:
        reason = status
    elif "Convergence failed" in text:
        tail = residuals[-12:]
        if (
            len(tail) >= 8
            and min(tail) > 0
            and max(tail) > min(tail) * 1.1
            and all(
                abs(tail[i] - tail[i - 2]) <= max(abs(tail[i]), 1e-12) * 0.05
                for i in range(2, len(tail))
            )
        ):
            reason = "oscillating_iterations"
        else:
            reason = "iteration_limit_or_failed_warm_start"
    else:
        reason = "no_converged_record"
    return {
        "reason": reason,
        "last_residual": residuals[-1] if residuals else None,
        "residual_tail": residuals[-12:],
        "explicit_failure_in_log": "Convergence failed" in text,
    }


def diagnose(root: Path):
    root = root.expanduser().resolve()
    saved = json.loads((root / "run.json").read_text())
    from ..data import parse_polar_file

    rows = parse_polar_file(root / saved["config"]["polar_filename"])
    records = []
    for alpha in saved.get("requested_alphas", saved["config"]["alphas"]):
        attempts = [
            r
            for r in saved.get("attempts", [])
            if any(abs(a - alpha) <= 0.00051 for a in r["alphas"])
        ]
        converged = row_for_alpha(rows, alpha) is not None
        cp = len(read_pairs(root / cp_name(alpha))) >= 3
        boundary_error = None
        if saved["config"].get("boundary_layer"):
            boundary_error = boundary_layer_error(
                root
                / "boundary_layer"
                / cp_name(alpha).replace("cp_", "bl_", 1),
                read_pairs(root / "geometry.dat"),
            )
        latest = attempts[-1] if attempts else {}
        relative = Path(latest.get("directory", "attempts")) / "xfoil.log"
        evidence = log_evidence(
            root / relative, latest.get("status", "not_attempted"), alpha
        )
        if not converged:
            reason = evidence["reason"]
        elif not cp:
            reason = "missing_cp"
        elif boundary_error is not None:
            reason = "invalid_boundary_layer"
        else:
            reason = "converged"
        # A failed attempt can leave an earlier polar row in the merged data.
        # Attribute recovery only to this attempt's own complete solution.
        recovered_by_init = (
            reason == "converged"
            and latest.get("initialization") == "INIT"
            and all(
                any(abs(a - alpha) <= 0.00051 for a in latest.get(key, []))
                for key in ("converged_alphas", "valid_output_alphas")
            )
        )
        records.append(
            {
                "alpha": alpha,
                "status": reason,
                "converged": converged,
                "boundary_layer_error": boundary_error,
                "recovered_by_init": recovered_by_init,
                "attempts": len(attempts),
                "seed_history": [a.get("seed_alpha") for a in attempts],
                "strategy_history": [a.get("strategy") for a in attempts],
                "initialization_history": [
                    a.get("initialization") for a in attempts
                ],
                "log": str(relative),
                "last_residual": evidence["last_residual"],
                "residual_tail": evidence["residual_tail"],
            }
        )
    payload = {
        "points": records,
        "note": (
            "Missing or oscillating solutions do not establish "
            "a physical stall angle."
        ),
    }
    atomic_json(root / "diagnostics.json", payload)
    write_records(root / "diagnostics.csv", records)
    report_page(
        root / "diagnostics.html",
        "收敛诊断",
        "按迎角查看是否收敛、重试次数与逼近方向。日志原因只描述数值证据，不代表已经确定物理失速。",
        records,
        [
            (f"α={r['alpha']:g}° 的日志", r["log"])
            for r in records
            if r["status"] != "converged"
        ],
    )
    return payload


def mark_missing(ax, angles):
    if not angles:
        return
    ax.scatter(
        angles,
        [0.025] * len(angles),
        transform=ax.get_xaxis_transform(),
        marker="x",
        color="#bd3030",
        s=34,
        zorder=10,
        label="Missing requested point",
    )
    for alpha in angles[:12]:
        ax.annotate(
            f"{alpha:g}°",
            (alpha, 0.025),
            xycoords=ax.get_xaxis_transform(),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="#a82424",
        )
