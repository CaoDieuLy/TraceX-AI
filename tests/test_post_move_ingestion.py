from __future__ import annotations

import importlib.util
import json
import sys
import unittest
import uuid
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_INGEST_PATH = PROJECT_ROOT / "backend" / "services" / "metadata-service" / "app" / "storage_ingest.py"
POST_MOVE_INGESTION_PATH = PROJECT_ROOT / "backend" / "services" / "metadata-service" / "app" / "post_move_ingestion.py"


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


storage_ingest = _load_module("metadata_storage_ingest", STORAGE_INGEST_PATH)
post_move_ingestion = _load_module("metadata_post_move_ingestion", POST_MOVE_INGESTION_PATH)
TEST_WORK_ROOT = PROJECT_ROOT / "tests" / ".tmp_post_move_ingestion"


class PostMoveIngestionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.work_root = TEST_WORK_ROOT / uuid.uuid4().hex
        self.storage_root = self.work_root / "storage"
        self.metadata_dir = self.work_root / "metadata"

    def _storage_item(self):
        source_path = self.storage_root / "cam_01" / "2026-04-19" / "cam_01_2026-04-19_12-10.mp4"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(b"mp4")
        stat = source_path.stat()
        return storage_ingest.StorageVideoItem(
            source_path=source_path,
            relative_path="cam_01/2026-04-19/cam_01_2026-04-19_12-10.mp4",
            camera_id="cam_01",
            recorded_at=datetime(2026, 4, 19, 12, 10, 0),
            size_bytes=int(stat.st_size),
            modified_ns=int(stat.st_mtime_ns),
            fingerprint="abc123",
        )

    def test_request_factory_builds_strict_post_move_pipeline_metadata(self) -> None:
        item = self._storage_item()
        factory = post_move_ingestion.StorageTrackingRequestFactory()

        task = factory.build(item)

        self.assertEqual(task.source_mode, "local_storage_ingest")
        self.assertEqual(task.output_basename, item.source_filename)
        contract = task.metadata["ingestion_contract"]
        self.assertEqual(contract["decode_sampling"]["sample_fps"], 5)
        self.assertEqual(contract["tracking"]["execution_scope"], "per_video")
        self.assertEqual(
            contract["tracking"]["independence_rule"],
            "each_10_min_video_is_independent",
        )
        self.assertTrue(contract["basic_feature_branch"]["always_run"])
        self.assertEqual(
            contract["lazy_action_branch"]["trigger_mode"],
            "on_demand_or_low_confidence",
        )

    def test_result_assembler_persists_local_metadata_artifact(self) -> None:
        item = self._storage_item()
        assembler = post_move_ingestion.StorageTrackingResultAssembler(self.metadata_dir)
        raw_result = {
            "video": {"video_id": "video-1", "camera_id": "cam_01"},
            "people": [{"candidate_id": "cand-1", "track_id": "track-1"}],
            "person_count": 1,
        }

        processed = assembler.assemble(item, raw_result, "local_storage_ingest")

        metadata_path = Path(processed.result["metadata_path"])
        self.assertTrue(metadata_path.exists())
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["video"]["source_storage_relative_path"], item.relative_path)
        self.assertEqual(payload["video"]["compressed_path"], str(item.source_path))
        self.assertEqual(payload["people"][0]["track_id"], "track-1")


if __name__ == "__main__":
    unittest.main()
