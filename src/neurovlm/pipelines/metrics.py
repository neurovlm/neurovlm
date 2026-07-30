"""Canonical long-form metric recording and derived summaries."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import MetricDirection, RunConfig
from .serialization import atomic_write_csv, atomic_write_json, json_safe, read_csv_rows


METRIC_COLUMNS = (
    "run_id",
    "task",
    "family",
    "variant",
    "domain",
    "split",
    "epoch",
    "step",
    "metric",
    "value",
    "n",
)


def metric_row(
    config: RunConfig,
    *,
    split: str,
    metric: str,
    value: float,
    epoch: int | None = None,
    step: int | None = None,
    n: int | None = None,
) -> dict[str, Any]:
    """Build one canonical long-form metric row."""

    if not split:
        raise ValueError("split must not be empty")
    if not metric:
        raise ValueError("metric must not be empty")
    return {
        "run_id": config.run_id,
        "task": config.task,
        "family": config.family,
        "variant": config.variant,
        "domain": config.domain,
        "split": split,
        "epoch": epoch,
        "step": step,
        "metric": metric,
        "value": float(value),
        "n": n,
    }


def curve_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return epoch/step-addressable rows suitable for plotting curves."""

    return [dict(row) for row in rows if row.get("epoch") is not None or row.get("step") is not None]


def summary_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    primary_metric: str | None = None,
    direction: MetricDirection | str = MetricDirection.MIN,
) -> list[dict[str, Any]]:
    """Summarize each split/metric series with first, last, min, max and best."""

    direction = MetricDirection(direction)
    groups: OrderedDict[tuple[Any, ...], list[Mapping[str, Any]]] = OrderedDict()
    for row in rows:
        key = (
            row.get("run_id"),
            row.get("task"),
            row.get("family"),
            row.get("variant"),
            row.get("domain"),
            row.get("split"),
            row.get("metric"),
        )
        groups.setdefault(key, []).append(row)
    output = []
    for key, group in groups.items():
        finite = [float(row["value"]) for row in group if json_safe(row.get("value")) is not None]
        first = group[0]
        last = group[-1]
        is_primary = key[-1] == primary_metric
        best_direction = direction if is_primary else None
        best_value = (
            (min(finite) if direction is MetricDirection.MIN else max(finite))
            if finite and is_primary
            else None
        )
        output.append(
            {
                "run_id": key[0],
                "task": key[1],
                "family": key[2],
                "variant": key[3],
                "domain": key[4],
                "split": key[5],
                "metric": key[6],
                "count": len(group),
                "first": first.get("value"),
                "last": last.get("value"),
                "min": min(finite) if finite else None,
                "max": max(finite) if finite else None,
                "best": best_value,
                "best_direction": best_direction,
                "last_epoch": last.get("epoch"),
                "last_step": last.get("step"),
                "n": last.get("n"),
            }
        )
    return output


class MetricRecorder:
    """In-memory recorder that atomically emits canonical metric artifacts."""

    def __init__(
        self,
        config: RunConfig,
        metrics_dir: str | Path | None = None,
        *,
        resume: bool = True,
    ):
        self.config = config
        self.metrics_dir = Path(metrics_dir) if metrics_dir is not None else config.run_dir / "metrics"
        self.rows: list[dict[str, Any]] = []
        if resume:
            for row in read_csv_rows(self.metrics_dir / "history.csv"):
                parsed: dict[str, Any] = dict(row)
                for field in ("epoch", "step", "n"):
                    parsed[field] = int(row[field]) if row.get(field) not in (None, "") else None
                parsed["value"] = (
                    float(row["value"]) if row.get("value") not in (None, "") else float("nan")
                )
                parsed["domain"] = row.get("domain") or None
                self.rows.append(parsed)

    def record(
        self,
        *,
        split: str,
        metric: str,
        value: float,
        epoch: int | None = None,
        step: int | None = None,
        n: int | None = None,
    ) -> dict[str, Any]:
        row = metric_row(
            self.config,
            split=split,
            metric=metric,
            value=value,
            epoch=epoch,
            step=step,
            n=n,
        )
        self.rows.append(row)
        return row

    def extend(self, rows: Iterable[Mapping[str, Any]]) -> None:
        for row in rows:
            missing = set(METRIC_COLUMNS).difference(row)
            if missing:
                raise ValueError(f"Metric row is missing columns: {sorted(missing)}")
            self.rows.append({column: row[column] for column in METRIC_COLUMNS})

    def summaries(self) -> list[dict[str, Any]]:
        return summary_rows(
            self.rows,
            primary_metric=self.config.primary_metric,
            direction=self.config.metric_direction,
        )

    def curves(self) -> list[dict[str, Any]]:
        return curve_rows(self.rows)

    def flush(self) -> dict[str, Path]:
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        history = atomic_write_csv(
            self.metrics_dir / "history.csv", self.rows, fieldnames=METRIC_COLUMNS
        )
        summaries = self.summaries()
        summary_csv = atomic_write_csv(self.metrics_dir / "summary.csv", summaries)
        summary_json = atomic_write_json(self.metrics_dir / "summary.json", summaries)
        curves = atomic_write_csv(
            self.metrics_dir / "curves.csv", self.curves(), fieldnames=METRIC_COLUMNS
        )
        return {
            "history": history,
            "summary_csv": summary_csv,
            "summary_json": summary_json,
            "curves": curves,
        }


__all__ = [
    "METRIC_COLUMNS",
    "MetricRecorder",
    "curve_rows",
    "metric_row",
    "summary_rows",
]
