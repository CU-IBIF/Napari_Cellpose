"""Guarded entry point: the Cellpose child process is spawned and re-imports __main__."""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import shots_body  # noqa: F401  (runs the screenshots)
