#!/usr/bin/env python3
"""Convenience CLI entry point for running the enterprise embedding & ingestion module."""
from __future__ import annotations

import sys
from embedding.cli import main

if __name__ == "__main__":
    sys.exit(main())
