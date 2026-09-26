"""Unit tests for the PPE compliance rule engine (compliance/rules.py).

Tests every compliance combination as specified:
  - Hardhat present + Safety Vest present  => COMPLIANT
  - Hardhat missing + Safety Vest present  => NON_COMPLIANT (Missing: Hardhat)
  - Hardhat present + Safety Vest missing  => NON_COMPLIANT (Missing: Safety Vest)
  - Hardhat missing + Safety Vest missing  => NON_COMPLIANT (Missing: Hardhat, Safety Vest)

Also tests extensible equipment rules, batch evaluation, and data structures.
"""

from __future__ import annotations

from pathlib import Path
import sys
import pytest

from compliance.rules import (
    DEFAULT_REQUIRED_PPE,
    ComplianceEngine,
    ComplianceResult,
    ComplianceStatus,
    DefaultPPE,
    PPEItem,
    PersonPPEInfo,
)


# ---------------------------------------------------------------------------
# Core Rule Combination Tests
# ---------------------------------------------------------------------------


class TestFourComplianceCombinations:
    """Test all 4 canonical combinations of Hardhat & Safety Vest."""

    @pytest.fixture
    def engine(self) -> ComplianceEngine:
        return ComplianceEngine()

    def test_combination_1_both_present_is_compliant(self, engine: ComplianceEngine):
        """Hardhat present + Safety Vest present => COMPLIANT."""
        person = PersonPPEInfo(
            person_id="worker_01",
            has_hardhat=True,
            has_safety_vest=True,
        )
        result = engine.evaluate(person)

        assert result.status == ComplianceStatus.COMPLIANT
        assert result.is_compliant is True
        assert "Hardhat" in result.present_equipment
        assert "Safety Vest" in result.present_equipment
        assert len(result.missing_equipment) == 0
        assert result.summary == "COMPLIANT"

    def test_combination_2_hardhat_missing_vest_present(self, engine: ComplianceEngine):
        """Hardhat missing + Safety Vest present => NON_COMPLIANT (Missing: Hardhat)."""
        person = PersonPPEInfo(
            person_id="worker_02",
            has_hardhat=False,
            has_safety_vest=True,
        )
        result = engine.evaluate(person)

        assert result.status == ComplianceStatus.NON_COMPLIANT
        assert result.is_compliant is False
        assert result.missing_equipment == ["Hardhat"]
        assert result.present_equipment == ["Safety Vest"]
        assert "Missing: Hardhat" in result.summary
        assert result.summary == "NON_COMPLIANT (Missing: Hardhat)"

    def test_combination_3_hardhat_present_vest_missing(self, engine: ComplianceEngine):
        """Hardhat present + Safety Vest missing => NON_COMPLIANT (Missing: Safety Vest)."""
        person = PersonPPEInfo(
            person_id="worker_03",
            has_hardhat=True,
            has_safety_vest=False,
        )
        result = engine.evaluate(person)

        assert result.status == ComplianceStatus.NON_COMPLIANT
        assert result.is_compliant is False
        assert result.missing_equipment == ["Safety Vest"]
        assert result.present_equipment == ["Hardhat"]
        assert "Missing: Safety Vest" in result.summary
        assert result.summary == "NON_COMPLIANT (Missing: Safety Vest)"

    def test_combination_4_both_missing(self, engine: ComplianceEngine):
        """Hardhat missing + Safety Vest missing => NON_COMPLIANT (Missing: Hardhat, Safety Vest)."""
        person = PersonPPEInfo(
            person_id="worker_04",
            has_hardhat=False,
            has_safety_vest=False,
        )
        result = engine.evaluate(person)

        assert result.status == ComplianceStatus.NON_COMPLIANT
        assert result.is_compliant is False
        assert set(result.missing_equipment) == {"Hardhat", "Safety Vest"}
        assert len(result.present_equipment) == 0
        assert "Missing: Hardhat, Safety Vest" in result.summary


# ---------------------------------------------------------------------------
# Input Data Structure Tests
# ---------------------------------------------------------------------------


class TestPersonPPEInfoStructures:
    """Test various input formats for PersonPPEInfo."""

    def test_construction_with_equipment_dict(self):
        person = PersonPPEInfo(
            person_id="p1",
            equipment={"Hardhat": True, "Safety Vest": True},
        )
        assert person.has_hardhat is True
        assert person.has_safety_vest is True
        assert person.has_item("Hardhat") is True
        assert person.has_item("Safety Vest") is True
        assert person.has_item("Gloves") is False

    def test_construction_with_ppe_items(self):
        items = [
            PPEItem(name="Hardhat", present=True, confidence=0.95),
            PPEItem(name="Safety Vest", present=False, confidence=0.88),
        ]
        person = PersonPPEInfo(person_id="p2", items=items)
        assert person.has_hardhat is True
        assert person.has_safety_vest is False

    def test_case_insensitive_item_lookup(self):
        person = PersonPPEInfo(has_hardhat=True, has_safety_vest=True)
        assert person.has_item("hardhat") is True
        assert person.has_item("HARDHAT") is True
        assert person.has_item("safety vest") is True
        assert person.has_item("SAFETY VEST") is True

    def test_from_detected_classes_with_positive_labels(self):
        classes = ["Person", "Hardhat", "Safety Vest", "Gloves"]
        person = PersonPPEInfo.from_detected_classes(classes, person_id="worker_detect")
        assert person.has_hardhat is True
        assert person.has_safety_vest is True
        assert person.has_item("Gloves") is True

    def test_from_detected_classes_with_negative_labels(self):
        # 'NO-Hardhat' and 'NO-Safety Vest' should explicitly mark missing
        classes = ["Person", "NO-Hardhat", "NO-Safety Vest"]
        person = PersonPPEInfo.from_detected_classes(classes, person_id="worker_negative")
        assert person.has_hardhat is False
        assert person.has_safety_vest is False


# ---------------------------------------------------------------------------
# Output Data Structure Tests
# ---------------------------------------------------------------------------


class TestComplianceResult:
    """Test output ComplianceResult structure and serialization."""

    def test_to_dict_contains_all_fields(self):
        engine = ComplianceEngine()
        person = PersonPPEInfo(person_id="p_test", has_hardhat=True, has_safety_vest=False)
        result = engine.evaluate(person)
        res_dict = result.to_dict()

        assert res_dict["person_id"] == "p_test"
        assert res_dict["status"] == "NON_COMPLIANT"
        assert res_dict["is_compliant"] is False
        assert res_dict["summary"] == "NON_COMPLIANT (Missing: Safety Vest)"
        assert res_dict["required_equipment"] == ["Hardhat", "Safety Vest"]
        assert res_dict["present_equipment"] == ["Hardhat"]
        assert res_dict["missing_equipment"] == ["Safety Vest"]
        assert "evaluated_at" in res_dict
        assert "details" in res_dict
        assert res_dict["details"]["compliance_score"] == 0.5


# ---------------------------------------------------------------------------
# Extensibility Tests (Adding & Modifying Rules)
# ---------------------------------------------------------------------------


class TestExtensibility:
    """Verify that adding, removing, or changing required PPE works seamlessly."""

    def test_add_requirement_gloves(self):
        engine = ComplianceEngine()
        engine.add_requirement("Gloves")

        assert "Gloves" in engine.required_equipment
        assert len(engine.required_equipment) == 3

        # Worker with Hardhat & Vest, but no Gloves => NON_COMPLIANT
        person = PersonPPEInfo(
            has_hardhat=True,
            has_safety_vest=True,
            equipment={"Gloves": False},
        )
        result = engine.evaluate(person)
        assert result.is_compliant is False
        assert result.missing_equipment == ["Gloves"]
        assert "Missing: Gloves" in result.summary

        # Add Gloves => COMPLIANT
        person.equipment["Gloves"] = True
        result_with_gloves = engine.evaluate(person)
        assert result_with_gloves.is_compliant is True

    def test_custom_requirements_laboratory(self):
        """In a laboratory setting, require Mask and Gloves instead of hardhat."""
        lab_engine = ComplianceEngine(required_equipment=["Mask", "Gloves"])
        assert lab_engine.required_equipment == ["Mask", "Gloves"]

        person = PersonPPEInfo(equipment={"Mask": True, "Gloves": True})
        result = lab_engine.evaluate(person)
        assert result.is_compliant is True

        person_no_mask = PersonPPEInfo(equipment={"Mask": False, "Gloves": True})
        result_fail = lab_engine.evaluate(person_no_mask)
        assert result_fail.is_compliant is False
        assert result_fail.missing_equipment == ["Mask"]

    def test_remove_requirement(self):
        engine = ComplianceEngine()
        engine.remove_requirement("Safety Vest")
        assert engine.required_equipment == ["Hardhat"]

        person = PersonPPEInfo(has_hardhat=True, has_safety_vest=False)
        result = engine.evaluate(person)
        # Compliant because Safety Vest was removed from requirements
        assert result.is_compliant is True

    def test_set_requirements_empty_raises_error(self):
        engine = ComplianceEngine()
        with pytest.raises(ValueError):
            engine.set_requirements([])


# ---------------------------------------------------------------------------
# Batch Evaluation & Summary Tests
# ---------------------------------------------------------------------------


class TestBatchAndSummary:
    """Test evaluate_batch and aggregate summary statistics."""

    def test_evaluate_batch(self):
        engine = ComplianceEngine()
        persons = [
            PersonPPEInfo(person_id="p1", has_hardhat=True, has_safety_vest=True),    # Compliant
            PersonPPEInfo(person_id="p2", has_hardhat=False, has_safety_vest=True),   # Non-compliant
            PersonPPEInfo(person_id="p3", has_hardhat=True, has_safety_vest=False),   # Non-compliant
            PersonPPEInfo(person_id="p4", has_hardhat=False, has_safety_vest=False),  # Non-compliant
        ]
        results = engine.evaluate_batch(persons)
        assert len(results) == 4
        assert [r.is_compliant for r in results] == [True, False, False, False]

    def test_generate_summary_metrics(self):
        engine = ComplianceEngine()
        persons = [
            PersonPPEInfo(person_id="p1", has_hardhat=True, has_safety_vest=True),
            PersonPPEInfo(person_id="p2", has_hardhat=True, has_safety_vest=True),
            PersonPPEInfo(person_id="p3", has_hardhat=False, has_safety_vest=True),
            PersonPPEInfo(person_id="p4", has_hardhat=False, has_safety_vest=False),
        ]
        results = engine.evaluate_batch(persons)
        summary = engine.generate_summary(results)

        assert summary["total_persons"] == 4
        assert summary["compliant_count"] == 2
        assert summary["non_compliant_count"] == 2
        assert summary["compliance_rate"] == 50.0
        assert summary["missing_breakdown"]["Hardhat"] == 2
        assert summary["missing_breakdown"]["Safety Vest"] == 1

    def test_generate_summary_empty(self):
        engine = ComplianceEngine()
        summary = engine.generate_summary([])
        assert summary["total_persons"] == 0
        assert summary["compliance_rate"] == 0.0


# ---------------------------------------------------------------------------
# Pure Python Independence Test
# ---------------------------------------------------------------------------


class TestPurePythonIndependence:
    """Verify that compliance/rules.py has NO dependency on OpenCV, FastAPI, or Torch."""

    def test_no_forbidden_imports_in_rules_module(self):
        import compliance.rules as rules_mod

        # Inspect module's imported names
        module_source = Path(rules_mod.__file__).read_text(encoding="utf-8")

        assert "cv2" not in module_source
        assert "opencv" not in module_source.lower()
        assert "fastapi" not in module_source.lower()
        assert "torch" not in module_source.lower()
        assert "react" not in module_source.lower()
