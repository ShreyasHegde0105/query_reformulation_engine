"""Unit tests for the enterprise corpus validation module.

# Ensure the repository root is on sys.path so that top‑level packages such as
# ``reformulation`` can be imported when the test runner executes from the
# ``tests`` directory.
import sys
import pathlib
repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
"""
