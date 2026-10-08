#!/usr/bin/env python3
"""Run the analytical Grad-CAM contracts, requiring the model dependencies."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Explicit invocation must fail when model dependencies are missing instead of
# reporting skipped tests as a success in the artifact-only environment.
import torch  # noqa: F401, E402


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_grad_cam.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
