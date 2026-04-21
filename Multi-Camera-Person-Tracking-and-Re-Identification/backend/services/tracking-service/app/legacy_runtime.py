import sys
from contextlib import contextmanager
from pathlib import Path

from .config import settings


@contextmanager
def legacy_workdir():
    # Legacy modules already resolve most paths from __file__, so changing the
    # process-wide cwd only creates cross-thread races without adding value.
    legacy_root = Path(settings.legacy_root).resolve()
    legacy_root_str = str(legacy_root)
    if legacy_root_str not in sys.path:
        sys.path.insert(0, legacy_root_str)
    yield legacy_root


# Initialize sys.path on first import (for legacy modules)
# NOTE: This uses the initial settings value
LEGACY_ROOT = Path(settings.legacy_root).resolve()
if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))


def update_legacy_root(new_root: str | Path):
    """
    Update LEGACY_ROOT path in sys.path when settings.legacy_root is overridden.
    Removes old path and adds new one.
    """
    global LEGACY_ROOT
    new_root_str = str(Path(new_root).resolve())
    old_root_str = str(LEGACY_ROOT)

    # Remove old path if present
    if old_root_str in sys.path:
        sys.path.remove(old_root_str)

    # Add new path if not already there
    if new_root_str not in sys.path:
        sys.path.insert(0, new_root_str)

    LEGACY_ROOT = Path(new_root_str)
    print(f"[legacy_runtime] Updated LEGACY_ROOT: {old_root_str} -> {new_root_str}")
