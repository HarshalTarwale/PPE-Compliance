"""
Tests for scripts/add_lifejacket_dataset.py.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from scripts.add_lifejacket_dataset import (
    TARGET_NAMES_5CLASS,
    TARGET_NAMES_7CLASS,
    build_class_mapping,
    merge_lifejacket_dataset,
)


def test_build_class_mapping_as_safety_vest():
    """Life jacket aliases must map to 'Safety Vest' (class 3) and 'NO-Safety Vest' (class 4)."""
    source_names = ["lifejacket", "no-lifejacket", "person"]
    mapping = build_class_mapping(source_names, as_safety_vest=True)

    # 0 ('lifejacket') -> 3 ('Safety Vest')
    assert mapping[0] == (3, "Safety Vest")
    # 1 ('no-lifejacket') -> 4 ('NO-Safety Vest')
    assert mapping[1] == (4, "NO-Safety Vest")
    # 2 ('person') -> 0 ('Person')
    assert mapping[2] == (0, "Person")


def test_build_class_mapping_custom_aliases():
    """Explicit overrides like wear/notwear map correctly."""
    source_names = ["wear", "notwear", "worker"]
    mapping = build_class_mapping(
        source_names,
        positive_class="wear",
        negative_class="notwear",
        as_safety_vest=True,
    )

    assert mapping[0] == (3, "Safety Vest")
    assert mapping[1] == (4, "NO-Safety Vest")
    assert mapping[2] == (0, "Person")


def test_build_class_mapping_separate_classes():
    """When separate_classes=True, maps to dedicated 7-class schema."""
    source_names = ["life jacket", "no life jacket", "Hardhat"]
    mapping = build_class_mapping(source_names, as_safety_vest=False)

    assert mapping[0] == (5, "Life Jacket")
    assert mapping[1] == (6, "NO-Life Jacket")
    assert mapping[2] == (1, "Hardhat")


def test_merge_lifejacket_dataset_mock(tmp_path: Path):
    """Test merge logic with a mock existing dataset and mock life jacket source."""
    # Existing dataset
    exist_dir = tmp_path / "exist"
    exist_train_img = exist_dir / "train" / "images"
    exist_train_lbl = exist_dir / "train" / "labels"
    exist_train_img.mkdir(parents=True)
    exist_train_lbl.mkdir(parents=True)

    (exist_train_img / "orig1.jpg").write_text("fake image")
    (exist_train_lbl / "orig1.txt").write_text("1 0.5 0.5 0.2 0.2\n")

    exist_yaml = {
        "path": str(exist_dir).replace("\\", "/"),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 5,
        "names": TARGET_NAMES_5CLASS,
    }
    with open(exist_dir / "data.yaml", "w") as f:
        yaml.dump(exist_yaml, f)

    # Source life jacket dataset
    src_dir = tmp_path / "source"
    src_train_img = src_dir / "train" / "images"
    src_train_lbl = src_dir / "train" / "labels"
    src_train_img.mkdir(parents=True)
    src_train_lbl.mkdir(parents=True)

    (src_train_img / "boat1.jpg").write_text("fake boat image")
    # source class 0 is "lifejacket"
    (src_train_lbl / "boat1.txt").write_text("0 0.5 0.5 0.3 0.4\n")

    src_yaml = {
        "names": ["lifejacket", "no-lifejacket"],
        "train": "train/images",
    }
    with open(src_dir / "data.yaml", "w") as f:
        yaml.dump(src_yaml, f)

    out_dir = tmp_path / "output"

    out_yaml_path = merge_lifejacket_dataset(
        source_dir=src_dir,
        existing_dataset_dir=exist_dir,
        output_dir=out_dir,
        as_safety_vest=True,
    )

    assert out_yaml_path.is_file()

    # Check generated data.yaml
    with open(out_yaml_path) as f:
        data = yaml.safe_load(f)
    assert data["nc"] == 5
    assert data["names"] == TARGET_NAMES_5CLASS

    # Check that original image exists
    assert (out_dir / "train" / "images" / "orig1.jpg").is_file()
    assert (out_dir / "train" / "labels" / "orig1.txt").read_text().strip() == "1 0.5 0.5 0.2 0.2"

    # Check that remapped life jacket image exists and has class 3
    assert (out_dir / "train" / "images" / "lj_boat1.jpg").is_file()
    lbl_content = (out_dir / "train" / "labels" / "lj_boat1.txt").read_text().strip()
    assert lbl_content == "3 0.5 0.5 0.3 0.4"
