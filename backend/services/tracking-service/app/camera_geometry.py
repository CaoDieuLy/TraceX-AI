from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorldProjectionCalibration:
    camera_projection_matrix: tuple[tuple[float, float, float, float], ...]
    homography_matrix: tuple[tuple[float, float, float], ...]
    reprojection_error: float


def _coerce_float_matrix(raw_value: object, expected_rows: int, expected_columns: int, field_name: str) -> tuple[tuple[float, ...], ...]:
    if not isinstance(raw_value, list) or len(raw_value) != expected_rows:
        raise ValueError(f"{field_name} must contain exactly {expected_rows} rows.")

    rows: list[tuple[float, ...]] = []
    for row_index, row in enumerate(raw_value):
        if not isinstance(row, list) or len(row) != expected_columns:
            raise ValueError(
                f"{field_name}[{row_index}] must contain exactly {expected_columns} numeric values."
            )
        rows.append(tuple(float(value) for value in row))
    return tuple(rows)


def load_world_projection_calibration(path: str | Path) -> WorldProjectionCalibration:
    calibration_path = Path(path).expanduser().resolve()
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))

    camera_projection_matrix = _coerce_float_matrix(
        payload.get("camera projection matrix"),
        expected_rows=3,
        expected_columns=4,
        field_name="camera projection matrix",
    )
    homography_matrix = _coerce_float_matrix(
        payload.get("homography matrix"),
        expected_rows=3,
        expected_columns=3,
        field_name="homography matrix",
    )
    reprojection_error = float(payload.get("reprojection_error", 0.0))

    return WorldProjectionCalibration(
        camera_projection_matrix=camera_projection_matrix,
        homography_matrix=homography_matrix,
        reprojection_error=reprojection_error,
    )
