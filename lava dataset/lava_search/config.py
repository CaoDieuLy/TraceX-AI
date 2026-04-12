from pathlib import Path

DEFAULT_REPO_ID = "xiaoyu123hhh/Lava_Dataset"
DEFAULT_LOCATIONS = (
    "amsterdam",
    "caldot1",
    "caldot2",
    "jackson",
    "shibuya",
    "warsaw",
)
DEFAULT_SPLITS = ("train", "test")
DEFAULT_FPS = 30.0
DEFAULT_TOP_K = 5
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_ASSETS_PER_MOMENT = 3
DEFAULT_SENTENCE_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_CLIP_MODEL = "ViT-L-14"
DEFAULT_CLIP_PRETRAINED = "laion2b_s32b_b82k"
DEFAULT_PROFILE = "strongest"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_root() -> Path:
    return project_root() / "data" / "Lava_Dataset"


def default_artifacts_root() -> Path:
    return project_root() / "artifacts"


def default_bundle_root() -> Path:
    return default_artifacts_root() / "full"


def default_visual_root() -> Path:
    return default_artifacts_root() / "visual_assets"
