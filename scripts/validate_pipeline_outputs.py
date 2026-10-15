"""Validate a single pipeline output directory and fail on invalid/missing data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from valoscribe.validation.outputs import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
