"""Unit tests for training/train.py helper functions.

Tests cover dataset validation, YAML path fixing, and config
preparation — NOT the full training loop (that requires GPU/model).
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from training.train import (
    _format_duration,
    _resolve_yaml_path,
    prepare_data_yaml,
    validate_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def good_yaml(tmp_path: Path) -> Path:
    """Create a valid data.yaml."""
    root = tmp_path / "dataset"
    root.mkdir()
    for split in ["train", "valid", "test"]:
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)

    yaml_text = textwrap.dedent("""\
        train: ../train/images
        val: ../valid/images
        test: ../test/images
        nc: 10
        names:
          0: Hardhat
          1: Mask
          2: NO-Hardhat
          3: NO-Mask
          4: NO-Safety Vest
          5: Person
          6: Safety Cone
          7: Safety Vest
          8: machinery
          9: vehicle
    """)
    yaml_path = root / "data.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


@pytest.fixture
def bad_yaml(tmp_path: Path) -> Path:
    """Create a data.yaml missing required fields."""
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text("some_key: some_value\n", encoding="utf-8")
    return yaml_path


# ---------------------------------------------------------------------------
# Tests: validate_dataset
# ---------------------------------------------------------------------------


class TestValidateDataset:
    def test_valid_yaml(self, good_yaml: Path) -> None:
        data = validate_dataset(good_yaml)
        assert data["nc"] == 10
        assert "names" in data
        assert "train" in data

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            validate_dataset(tmp_path / "nonexistent.yaml")

    def test_missing_fields(self, bad_yaml: Path) -> None:
        with pytest.raises(SystemExit):
            validate_dataset(bad_yaml)


# ---------------------------------------------------------------------------
# Tests: _resolve_yaml_path
# ---------------------------------------------------------------------------


class TestResolveYamlPath:
    def test_direct_path(self, tmp_path: Path) -> None:
        (tmp_path / "train" / "images").mkdir(parents=True)
        result = _resolve_yaml_path(tmp_path, "./train/images")
        assert Path(result).exists()

    def test_dotdot_fallback(self, tmp_path: Path) -> None:
        """../train/images should fall back to train/images under root."""
        (tmp_path / "train" / "images").mkdir(parents=True)
        result = _resolve_yaml_path(tmp_path, "../train/images")
        assert Path(result).exists()
        assert "train" in result

    def test_nonexistent_returns_path(self, tmp_path: Path) -> None:
        """Even if nothing resolves, it should return a path string."""
        result = _resolve_yaml_path(tmp_path, "../nonexistent/path")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Tests: prepare_data_yaml
# ---------------------------------------------------------------------------


class TestPrepareDataYaml:
    def test_creates_fixed_yaml(self, good_yaml: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        fixed = prepare_data_yaml(good_yaml, output_dir)
        assert fixed.exists()
        assert fixed.name == "data.yaml"
        assert fixed.parent == output_dir

    def test_fixed_yaml_has_absolute_paths(self, good_yaml: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        fixed = prepare_data_yaml(good_yaml, output_dir)
        content = fixed.read_text(encoding="utf-8")
        # Should NOT contain "../" anymore
        assert "../" not in content

    def test_original_not_modified(self, good_yaml: Path, tmp_path: Path) -> None:
        original_content = good_yaml.read_text(encoding="utf-8")
        prepare_data_yaml(good_yaml, tmp_path / "output")
        assert good_yaml.read_text(encoding="utf-8") == original_content


# ---------------------------------------------------------------------------
# Tests: _format_duration
# ---------------------------------------------------------------------------


class TestFormatDuration:
    def test_seconds_only(self) -> None:
        assert _format_duration(45) == "45s"

    def test_minutes_and_seconds(self) -> None:
        assert _format_duration(125) == "2m 5s"

    def test_hours_minutes_seconds(self) -> None:
        assert _format_duration(3723) == "1h 2m 3s"

    def test_zero(self) -> None:
        assert _format_duration(0) == "0s"
