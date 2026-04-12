import json
import shutil
import unittest
from pathlib import Path

from surveillance_search.runtime import load_runtime_manifest, save_runtime_manifest, snapshot_dataset_sources


class RuntimeTests(unittest.TestCase):
    def test_snapshot_dataset_sources_for_personpath22_includes_annotation_and_video(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        case_root = temp_root / "runtime-personpath22"
        shutil.rmtree(case_root, ignore_errors=True)

        annotation_dir = case_root / "data" / "annotations"
        annotation_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = case_root / "data" / "raw_data"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (annotation_dir / "anno_visible_2022.json").write_text(
            json.dumps({"videos": [], "images": [], "annotations": []}),
            encoding="utf-8",
        )
        (raw_dir / "cam_01.mp4").write_bytes(b"fake-video")

        entries = snapshot_dataset_sources(case_root / "data", dataset_type="personpath22")
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["path"], "annotations/anno_visible_2022.json")
        self.assertEqual(entries[1]["path"], "raw_data/cam_01.mp4")

        shutil.rmtree(temp_root, ignore_errors=True)

    def test_snapshot_dataset_sources_ignores_noncanonical_cache_files(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        case_root = temp_root / "runtime-personpath22-ignore-cache"
        shutil.rmtree(case_root, ignore_errors=True)

        annotation_dir = case_root / "data" / "annotations"
        annotation_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = case_root / "data" / "raw_data"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (annotation_dir / "anno_visible_2022.json").write_text(
            json.dumps({"videos": [], "images": [], "annotations": []}),
            encoding="utf-8",
        )
        (raw_dir / "cam_01.mp4").write_bytes(b"fake-video")

        stray_dir = case_root / "data" / "_kaggle_download" / "raw_data"
        stray_dir.mkdir(parents=True, exist_ok=True)
        (stray_dir / "wrong.mp4").write_bytes(b"wrong-video")

        entries = snapshot_dataset_sources(case_root / "data", dataset_type="personpath22")
        self.assertEqual([entry["path"] for entry in entries], ["annotations/anno_visible_2022.json", "raw_data/cam_01.mp4"])

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
