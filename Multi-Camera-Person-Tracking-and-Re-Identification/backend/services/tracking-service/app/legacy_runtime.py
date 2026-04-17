import os
import sys
from contextlib import contextmanager
from pathlib import Path

from .config import settings


LEGACY_ROOT = Path(settings.legacy_root).resolve()

if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))


@contextmanager
def legacy_workdir():
    previous = Path.cwd()
    os.chdir(LEGACY_ROOT)
    try:
        yield LEGACY_ROOT
    finally:
        os.chdir(previous)
