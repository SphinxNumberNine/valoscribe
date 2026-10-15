"""CLI entry point; also works without installing the package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from valoscribe.validation.events import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
