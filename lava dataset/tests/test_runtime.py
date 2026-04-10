import json
import shutil
import unittest
from pathlib import Path

from lava_search.runtime import load_runtime_manifest, save_runtime_manifest, snapshot_dataset_sources


class RuntimeTests(unittest.TestCase):
    def test_snapshot_dataset_sources_includes_label_and_video(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        case_root = temp_root / "runtime-case"
        shutil.rmtree(case_root, ignore_errors=True)

        split_dir = case_root / "data" / "Lava_Dataset" / "amsterdam" / "test"
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "label.json").write_text(json.dumps([[]]), encoding="utf-8")
        (split_dir / "test.mp4").write_bytes(b"fake-video")

        entries = snapshot_dataset_sources(case_root / "data" / "Lava_Dataset")
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["path"], "amsterdam/test/label.json")
        self.assertEqual(entries[1]["path"], "amsterdam/test/test.mp4")

        shutil.rmtree(temp_root, ignore_errors=True)

    def test_save_and_load_runtime_manifest(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        case_root = temp_root / "manifest-case"
        shutil.rmtree(case_root, ignore_errors=True)
        case_root.mkdir(parents=True, exist_ok=True)

        payload = {"moment_count": 12, "sources": [{"path": "a"}]}
        save_runtime_manifest(case_root, payload)
        loaded = load_runtime_manifest(case_root)
        self.assertEqual(loaded, payload)

        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
