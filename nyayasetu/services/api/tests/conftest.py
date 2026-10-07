"""Pytest configuration for the services/api test suite.

Sets up:
- sys.path so ``app.*`` imports resolve without installation.
- asyncio mode for pytest-asyncio.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make ``app`` importable as a top-level package.
sys.path.insert(0, str(Path(__file__).parent.parent))
