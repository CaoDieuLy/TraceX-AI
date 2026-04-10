import json
import shutil
import unittest
from pathlib import Path

from lava_search.dataset import build_moments_from_label
from lava_search.video_tools import select_visual_path


class DatasetParsingTests(unittest.TestCase):
    def test_group_track_into_single_moment(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        case_root = temp_root / "case1"
        shutil.rmtree(case_root, ignore_errors=True)

        label_path = case_root / "amsterdam" / "test" / "label.json"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            [
                {
                    "caption": ["red sedan"],
                    "left": 10,
                    "top": 20,
                    "right": 30,
                    "bottom": 40,
                    "track_id": 7,
                }
            ],
            [
                {
                    "caption": ["red car"],
                    "left": 11,
                    "top": 21,
                    "right": 31,
                    "bottom": 41,
                    "track_id": 7,
                }
            ],
        ]
        label_path.write_text(json.dumps(payload), encoding="utf-8")

        moments = build_moments_from_label(label_path, fps=10.0, group_by_track=True)

        self.assertEqual(len(moments), 1)
        self.assertEqual(moments[0].start_frame, 0)
        self.assertEqual(moments[0].end_frame, 1)
        self.assertEqual(moments[0].captions, ["red car", "red sedan"])
        self.assertAlmostEqual(moments[0].end_second, 0.2)
        self.assertIn("red", moments[0].keywords)
        self.assertIsNone(select_visual_path(moments[0]))

        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
