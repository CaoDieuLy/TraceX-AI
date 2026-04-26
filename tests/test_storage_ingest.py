from __future__ import annotations

import importlib.util
import sys
import unittest
import uuid
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "backend" / "services" / "metadata-service" / "app" / "storage_ingest.py"
SPEC = importlib.util.spec_from_file_location("storage_ingest", MODULE_PATH)
storage_ingest = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = storage_ingest
SPEC.loader.exec_module(storage_ingest)

TEST_WORK_ROOT = PROJECT_ROOT / "tests" / ".tmp_storage_ingest"


class StorageVideoScannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.work_root = TEST_WORK_ROOT / uuid.uuid4().hex
        self.storage_root = self.work_root / "storage"
        self.marker_dir = self.work_root / "markers"

    def _scanner(self):
        registry = storage_ingest.StorageIngestRegistry(self.marker_dir)
        return storage_ingest.StorageVideoScanner(
            storage_root=self.storage_root,
            registry=registry,
            min_file_age_seconds=0,
        )

    def test_scans_mp4_camera_date_layout_only(self) -> None:
        valid_path = self.storage_root / "cam_01" / "2026-04-19" / "cam_01_2026-04-19_12-10.mp4"
        misplaced_path = self.storage_root / "cam_02" / "2026-04-19" / "cam_01_2026-04-19_12-30.mp4"
        non_mp4_path = self.storage_root / "cam_01" / "2026-04-19" / "cam_01_2026-04-19_12-20.avi"
        wrong_name_path = self.storage_root / "cam_01" / "2026-04-19" / "camera-1.mp4"
        valid_path.parent.mkdir(parents=True)
        misplaced_path.parent.mkdir(parents=True)
        valid_path.write_bytes(b"mp4 video")
        misplaced_path.write_bytes(b"wrong folder")
        non_mp4_path.write_bytes(b"non mp4")
        wrong_name_path.write_bytes(b"wrong")

        pending = self._scanner().list_pending()

        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].camera_id, "cam_01")
        self.assertEqual(pending[0].source_filename, "cam_01_2026-04-19_12-10.mp4")
        self.assertEqual(pending[0].relative_path, "cam_01/2026-04-19/cam_01_2026-04-19_12-10.mp4")

    def test_processed_marker_skips_same_file(self) -> None:
        source_path = self.storage_root / "cam_02" / "2026-04-19" / "cam_02_2026-04-19_12-10.mp4"
        source_path.parent.mkdir(parents=True)
        source_path.write_bytes(b"video")

        registry = storage_ingest.StorageIngestRegistry(self.marker_dir)
        scanner = storage_ingest.StorageVideoScanner(
            storage_root=self.storage_root,
            registry=registry,
            min_file_age_seconds=0,
        )
        first_pending = scanner.list_pending()
        self.assertEqual(len(first_pending), 1)

        registry.mark_processed(first_pending[0], {"video": {"video_id": "video-1"}, "person_count": 3})
        self.assertEqual(scanner.list_pending(), [])


if __name__ == "__main__":
    unittest.main()
