from __future__ import annotations

import argparse
import errno
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
import re
from typing import Iterable, Sequence


CAMERA_VIDEO_PATTERN = re.compile(
    r"^(?P<camera_id>cam_\d{2,})_"
    r"(?P<recorded_date>\d{4}-\d{2}-\d{2})_"
    r"(?P<hour>\d{2})-(?P<minute>\d{2})"
    r"(?:-(?P<second>\d{2}))?"
    r"(?P<suffix>\.mp4)$",
    re.IGNORECASE,
)


class MoveStatus(str, Enum):
    MOVED = "moved"
    DRY_RUN = "dry_run"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class CameraVideoName:
    """Parsed input filename contract for one camera clip."""

    camera_id: str
    recorded_date: date
    recorded_at: datetime
    suffix: str
    original_name: str

    @classmethod
    def parse(cls, filename: str) -> "CameraVideoName | None":
        match = CAMERA_VIDEO_PATTERN.fullmatch(filename)
        if not match:
            return None

        try:
            recorded_date = date.fromisoformat(match.group("recorded_date"))
            second = int(match.group("second") or "0")
            recorded_at = datetime(
                recorded_date.year,
                recorded_date.month,
                recorded_date.day,
                int(match.group("hour")),
                int(match.group("minute")),
                second,
            )
        except ValueError:
            return None
        return cls(
            camera_id=match.group("camera_id").lower(),
            recorded_date=recorded_date,
            recorded_at=recorded_at,
            suffix=match.group("suffix").lower(),
            original_name=filename,
        )


@dataclass(frozen=True)
class MoveBatchRequest:
    """Input for moving one batch from drive-root/temp into drive-root/storage."""

    drive_root: Path
    temp_dir_name: str = "temp"
    storage_dir_name: str = "storage"
    dry_run: bool = False
    overwrite: bool = False
    limit: int | None = None
    allowed_suffixes: frozenset[str] = frozenset({".mp4"})

    @property
    def temp_dir(self) -> Path:
        return self.drive_root / self.temp_dir_name

    @property
    def storage_dir(self) -> Path:
        return self.drive_root / self.storage_dir_name


@dataclass(frozen=True)
class VideoMovePlan:
    """Concrete source and destination paths for a parsed camera video."""

    source_path: Path
    destination_path: Path
    parsed_name: CameraVideoName


@dataclass(frozen=True)
class VideoMoveResult:
    """Output row for one source file handled by move.py."""

    source_path: str
    destination_path: str | None
    status: MoveStatus
    camera_id: str | None = None
    recorded_date: str | None = None
    recorded_at: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class MoveBatchResult:
    """Output summary for the whole move batch."""

    drive_root: str
    temp_dir: str
    storage_dir: str
    moved_count: int
    skipped_count: int
    failed_count: int
    dry_run_count: int
    items: list[VideoMoveResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        for item in payload["items"]:
            item["status"] = str(item["status"].value)
        return payload

    @property
    def ok(self) -> bool:
        return self.failed_count == 0


class TempVideoOrganizer:
    """
    Moves completed camera clips into the storage layout consumed by ingestion.

    Input:
        drive_root/temp/cam_01_2026-04-19_12-10.mp4

    Output:
        drive_root/storage/cam_01/2026-04-19/cam_01_2026-04-19_12-10.mp4

    The file extension remains .mp4 because the project stores H.265 video in
    an MP4 container. This class does not transcode or change model runtime.
    """

    def __init__(self, request: MoveBatchRequest) -> None:
        self.request = request

    def move_pending_videos(self) -> MoveBatchResult:
        items: list[VideoMoveResult] = []
        for source_path in self._iter_source_files():
            plan_or_result = self._build_plan(source_path)
            if isinstance(plan_or_result, VideoMoveResult):
                items.append(plan_or_result)
                continue

            items.append(self._execute_plan(plan_or_result))

        return MoveBatchResult(
            drive_root=str(self.request.drive_root),
            temp_dir=str(self.request.temp_dir),
            storage_dir=str(self.request.storage_dir),
            moved_count=sum(1 for item in items if item.status == MoveStatus.MOVED),
            skipped_count=sum(1 for item in items if item.status == MoveStatus.SKIPPED),
            failed_count=sum(1 for item in items if item.status == MoveStatus.FAILED),
            dry_run_count=sum(1 for item in items if item.status == MoveStatus.DRY_RUN),
            items=items,
        )

    def _iter_source_files(self) -> Iterable[Path]:
        temp_dir = self.request.temp_dir
        if not temp_dir.exists():
            return []
        files = sorted(path for path in temp_dir.iterdir() if path.is_file())
        if self.request.limit is None:
            return files
        return files[: max(self.request.limit, 0)]

    def _build_plan(self, source_path: Path) -> VideoMovePlan | VideoMoveResult:
        parsed_name = CameraVideoName.parse(source_path.name)
        if parsed_name is None:
            return self._skip(source_path, "filename must match cam_XX_YYYY-MM-DD_HH-MM.mp4")

        if parsed_name.suffix not in self.request.allowed_suffixes:
            return self._skip(source_path, f"unsupported suffix: {parsed_name.suffix}")

        destination_path = (
            self.request.storage_dir
            / parsed_name.camera_id
            / parsed_name.recorded_date.isoformat()
            / parsed_name.original_name
        )
        return VideoMovePlan(
            source_path=source_path,
            destination_path=destination_path,
            parsed_name=parsed_name,
        )

    def _execute_plan(self, plan: VideoMovePlan) -> VideoMoveResult:
        parsed_name = plan.parsed_name
        base_result = {
            "source_path": str(plan.source_path),
            "destination_path": str(plan.destination_path),
            "camera_id": parsed_name.camera_id,
            "recorded_date": parsed_name.recorded_date.isoformat(),
            "recorded_at": parsed_name.recorded_at.isoformat(),
        }

        if plan.destination_path.exists() and not self.request.overwrite:
            return VideoMoveResult(
                **base_result,
                status=MoveStatus.SKIPPED,
                reason="destination already exists",
            )

        if self.request.dry_run:
            return VideoMoveResult(**base_result, status=MoveStatus.DRY_RUN)

        try:
            plan.destination_path.parent.mkdir(parents=True, exist_ok=True)
            self._move_file(plan.source_path, plan.destination_path)
            return VideoMoveResult(**base_result, status=MoveStatus.MOVED)
        except Exception as exc:  # pragma: no cover - keeps batch output explicit on IO failures.
            return VideoMoveResult(
                **base_result,
                status=MoveStatus.FAILED,
                reason=str(exc),
            )

    def _move_file(self, source_path: Path, destination_path: Path) -> None:
        """
        Move atomically when possible; otherwise copy to a partial file first.

        A storage watcher should only react to final .mp4 files, never the
        temporary .partial file used for cross-device moves.
        """

        try:
            source_path.replace(destination_path)
            return
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise

        partial_path = destination_path.with_name(f".{destination_path.name}.partial")
        if partial_path.exists():
            partial_path.unlink()
        shutil.copy2(source_path, partial_path)
        partial_path.replace(destination_path)
        source_path.unlink()

    @staticmethod
    def _skip(source_path: Path, reason: str) -> VideoMoveResult:
        return VideoMoveResult(
            source_path=str(source_path),
            destination_path=None,
            status=MoveStatus.SKIPPED,
            reason=reason,
        )


def _build_request(args: argparse.Namespace) -> MoveBatchRequest:
    drive_root = Path(args.drive_root).expanduser().resolve()
    return MoveBatchRequest(
        drive_root=drive_root,
        temp_dir_name=args.temp_dir,
        storage_dir_name=args.storage_dir,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        limit=args.limit,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Move H.265-in-MP4 camera clips from temp/ into storage/<camera>/<date>/."
    )
    parser.add_argument(
        "--drive-root",
        default=".",
        help="Root folder containing temp/ and storage/. Default: current directory.",
    )
    parser.add_argument("--temp-dir", default="temp", help="Input folder name under drive root.")
    parser.add_argument("--storage-dir", default="storage", help="Output folder name under drive root.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of temp files to process.")
    parser.add_argument("--dry-run", action="store_true", help="Plan moves without touching files.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing destination files.")
    parser.add_argument("--json", action="store_true", help="Print the full machine-readable result.")
    return parser


def _print_human_summary(result: MoveBatchResult) -> None:
    print(
        "Move summary: "
        f"moved={result.moved_count}, "
        f"dry_run={result.dry_run_count}, "
        f"skipped={result.skipped_count}, "
        f"failed={result.failed_count}"
    )
    for item in result.items:
        destination = item.destination_path or "-"
        suffix = f" ({item.reason})" if item.reason else ""
        print(f"[{item.status.value}] {item.source_path} -> {destination}{suffix}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    request = _build_request(args)
    result = TempVideoOrganizer(request).move_pending_videos()

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=True, indent=2))
    else:
        _print_human_summary(result)

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
