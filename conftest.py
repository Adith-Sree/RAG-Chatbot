"""
conftest.py — Shared pytest configuration and fixtures
"""
import sys
from pathlib import Path

# Ensure the project root is in the Python path
# This allows tests to import from any project module without relative imports
sys.path.insert(0, str(Path(__file__).parent))
