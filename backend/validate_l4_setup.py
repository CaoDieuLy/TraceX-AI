from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TRACKING_SERVICE_ROOT = ROOT / "backend" / "services" / "tracking-service"
if str(TRACKING_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKING_SERVICE_ROOT))

from app.strict_pipeline import get_strict_pipeline  # noqa: E402


def main() -> int:
    pipeline = get_strict_pipeline()
    print("[ok] strict pipeline loaded")
    print(f"[ok] detector={pipeline['components']['detector']['official_class_name']}")
    print("[ok] local substitute model stack is not validated here")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
