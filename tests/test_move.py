from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOVE_SCRIPT_PATH = PROJECT_ROOT / "move.py"
SPEC = importlib.util.spec_from_file_location("move_script", MOVE_SCRIPT_PATH)
move_script = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = move_script
SPEC.loader.exec_module(move_script)
TEST_WORK_ROOT = PROJECT_ROOT / "tests" / ".tmp_move"


class TempVideoOrganizerTest(unittest.TestCase):
    def setUp(self) -> None:
        test_name = self.id().rsplit(".", 1)[-1]
        self.drive_root = TEST_WORK_ROOT / f"{test_name}_{uuid.uuid4().hex}"
        (self.drive_root / "temp").mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.drive_root, ignore_errors=True)

    def _run_mover(self, **overrides):
        request = move_script.MoveBatchRequest(drive_root=self.drive_root, **overrides)
        return move_script.TempVideoOrganizer(request).move_pending_videos()

    def test_moves_mp4_to_camera_date_folder(self) -> None:
        source_path = self.drive_root / "temp" / "cam_01_2026-04-19_12-10.mp4"
        source_path.write_bytes(b"mp4 video")

        with patch.object(move_script.TempVideoOrganizer, "_move_file") as move_file:
            result = self._run_mover()

        destination_path = (
            self.drive_root
            / "storage"
            / "cam_01"
            / "2026-04-19"
            / "cam_01_2026-04-19_12-10.mp4"
        )
        self.assertEqual(result.moved_count, 1)
        move_file.assert_called_once_with(source_path, destination_path)
        self.assertEqual(result.items[0].camera_id, "cam_01")
        self.assertEqual(result.items[0].recorded_date, "2026-04-19")

    def test_dry_run_does_not_move_files(self) -> None:
        source_path = self.drive_root / "temp" / "cam_02_2026-04-19_12-20.mp4"
        source_path.write_bytes(b"video")

        result = self._run_mover(dry_run=True)

        self.assertEqual(result.dry_run_count, 1)
        json.dumps(result.to_dict())
        self.assertTrue(source_path.exists())
        self.assertFalse((self.drive_root / "storage").exists())

    def test_skips_unrecognized_filename(self) -> None:
        source_path = self.drive_root / "temp" / "camera-1.mp4"
        source_path.write_bytes(b"video")
        invalid_date_path = self.drive_root / "temp" / "cam_01_2026-99-19_12-10.mp4"
        invalid_date_path.write_bytes(b"video")

        result = self._run_mover()

        self.assertEqual(result.skipped_count, 2)
        self.assertTrue(source_path.exists())
        self.assertTrue(invalid_date_path.exists())
        self.assertIn("filename must match", result.items[0].reason or "")

    def test_existing_destination_is_skipped_without_overwrite(self) -> None:
        source_path = self.drive_root / "temp" / "cam_03_2026-04-19_12-30.mp4"
        destination_path = (
            self.drive_root
            / "storage"
            / "cam_03"
            / "2026-04-19"
            / "cam_03_2026-04-19_12-30.mp4"
        )
        source_path.write_bytes(b"new")
        destination_path.parent.mkdir(parents=True)
        destination_path.write_bytes(b"existing")

        result = self._run_mover()

        self.assertEqual(result.skipped_count, 1)
        self.assertTrue(source_path.exists())
        self.assertEqual(destination_path.read_bytes(), b"existing")
        self.assertEqual(result.items[0].reason, "destination already exists")


if __name__ == "__main__":
    unittest.main()
