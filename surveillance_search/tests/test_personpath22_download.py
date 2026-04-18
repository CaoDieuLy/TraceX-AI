import io
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from surveillance_search.dataset import _download_url_to_path, _materialize_personpath22_archives


class _FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, content_length: int | None = None) -> None:
        super().__init__(payload)
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class PersonPath22DownloadTests(unittest.TestCase):
    def test_materialize_raises_clear_error_for_invalid_video_zip(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests" / "invalid-video-zip"
        shutil.rmtree(temp_root, ignore_errors=True)
        raw_data_dir = temp_root / "raw_data"
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        (raw_data_dir / "videos.zip").write_bytes(b"not-a-zip")

        with self.assertRaisesRegex(RuntimeError, "Delete it and rerun the download command"):
            _materialize_personpath22_archives(temp_root, include_videos=True)

        shutil.rmtree(temp_root, ignore_errors=True)

    def test_download_cleans_partial_file_when_content_length_is_short(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests" / f"short-download-{uuid.uuid4().hex}"
        shutil.rmtree(temp_root, ignore_errors=True)
        destination_path = temp_root / "raw_data" / "videos.zip"
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        with patch(
            "surveillance_search.dataset.urllib.request.urlopen",
            return_value=_FakeResponse(b"12345", content_length=10),
        ):
            with self.assertRaisesRegex(RuntimeError, "Download was incomplete"):
                _download_url_to_path("https://example.invalid/videos.zip", destination_path)

        self.assertFalse(destination_path.exists())

        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
