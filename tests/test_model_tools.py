"""
Unit tests for model tools: prepare_5class_dataset.py and compare_models.py.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.compare_models import format_delta
from scripts.prepare_5class_dataset import TARGET_NAMES


def test_target_classes_structure() -> None:
    """Verify target classes are properly ordered and indexed."""
    assert len(TARGET_NAMES) == 5
    assert TARGET_NAMES[0] == "Person"
    assert TARGET_NAMES[1] == "Hardhat"
    assert TARGET_NAMES[2] == "NO-Hardhat"
    assert TARGET_NAMES[3] == "Safety Vest"
    assert TARGET_NAMES[4] == "NO-Safety Vest"


def test_format_delta_percentage() -> None:
    """Test format_delta with percentage output."""
    res_pos = format_delta(0.65, 0.55, is_percentage=True)
    assert res_pos == "+10.00%"

    res_neg = format_delta(0.50, 0.55, is_percentage=True)
    assert res_neg == "-5.00%"

    res_zero = format_delta(0.50, 0.50, is_percentage=True)
    assert res_zero == "+0.00%"


def test_format_delta_float() -> None:
    """Test format_delta with raw float output."""
    res_pos = format_delta(1.2345, 1.0000, is_percentage=False)
    assert res_pos == "+0.2345"

    res_neg = format_delta(1.0000, 1.2345, is_percentage=False)
    assert res_neg == "-0.2345"
