"""Unit tests for the Person-to-PPE spatial association algorithm (detection/ppe_association.py).

Tests use synthetic bounding boxes to verify:
  1. Geometric helpers (centroid, intersection, containment).
  2. Anatomical prior region checking (head vs torso vs invalid regions).
  3. Single person PPE assignment for all positive and negative combinations.
  4. Multiple person disambiguation (closest/highest scoring person wins, exclusive assignment).
  5. Unassigned PPE handling (isolated gear not near any person).
  6. Conflicting detection resolution (higher confidence takes precedence).
  7. End-to-end integration with ComplianceEngine.
"""

from __future__ import annotations

import pytest

from compliance.rules import ComplianceEngine, ComplianceStatus
from detection.detector import Detection
from detection.ppe_association import (
    AssociatedPerson,
    AssociationConfig,
    AssociationResult,
    PPEAssociator,
)


# ---------------------------------------------------------------------------
# Geometric Helper Tests
# ---------------------------------------------------------------------------


class TestGeometricHelpers:
    """Test pure geometric functions."""

    def test_calculate_box_center(self):
        box = (100, 200, 300, 600)
        cx, cy = PPEAssociator.calculate_box_center(box)
        assert cx == 200.0
        assert cy == 400.0

    def test_intersection_area(self):
        # 50x50 overlap
        box_a = (0, 0, 100, 100)
        box_b = (50, 50, 150, 150)
        assert PPEAssociator.calculate_intersection_area(box_a, box_b) == 2500.0

        # Disjoint
        box_c = (200, 200, 300, 300)
        assert PPEAssociator.calculate_intersection_area(box_a, box_c) == 0.0

    def test_containment_ratio(self):
        person_box = (100, 100, 300, 500)  # w=200, h=400

        # PPE fully inside person box (100% containment)
        ppe_inside = (150, 120, 250, 180)
        assert PPEAssociator.calculate_containment_ratio(person_box, ppe_inside) == 1.0

        # PPE partially outside person box
        ppe_half = (250, 120, 350, 180)  # 50% in, 50% out
        assert pytest.approx(PPEAssociator.calculate_containment_ratio(person_box, ppe_half), 0.01) == 0.50

        # PPE completely outside
        ppe_outside = (400, 100, 500, 200)
        assert PPEAssociator.calculate_containment_ratio(person_box, ppe_outside) == 0.0


# ---------------------------------------------------------------------------
# Anatomical Region Checking Tests
# ---------------------------------------------------------------------------


class TestAnatomicalRegions:
    """Test anatomical filtering based on relative vertical positions."""

    @pytest.fixture
    def associator(self) -> PPEAssociator:
        return PPEAssociator()

    # Synthetic Person: (100, 100, 300, 500) => W=200, H=400
    # Head region (y: 100 to 252), Torso region (y: 148 to 388), Legs/Feet (y: 388 to 500)
    person_box = (100, 100, 300, 500)

    def test_hardhat_in_head_region_is_valid_candidate(self, associator: PPEAssociator):
        # Center at y=130 (rel_y = 30/400 = 0.075, ideal head region)
        hardhat_box = (160, 110, 240, 150)
        is_cand, rx, ry = associator.is_candidate_region(self.person_box, hardhat_box, "Hardhat")
        assert is_cand is True
        assert 0.0 <= ry <= 0.38

    def test_hardhat_near_feet_is_rejected(self, associator: PPEAssociator):
        # Center at y=460 (rel_y = 360/400 = 0.90, feet region)
        hardhat_box = (160, 440, 240, 480)
        is_cand, _, _ = associator.is_candidate_region(self.person_box, hardhat_box, "Hardhat")
        assert is_cand is False

    def test_safety_vest_in_torso_region_is_valid(self, associator: PPEAssociator):
        # Center at y=250 (rel_y = 150/400 = 0.375, ideal torso region)
        vest_box = (140, 180, 260, 320)
        is_cand, rx, ry = associator.is_candidate_region(self.person_box, vest_box, "Safety Vest")
        assert is_cand is True
        assert 0.12 <= ry <= 0.72

    def test_safety_vest_above_head_is_rejected(self, associator: PPEAssociator):
        # Center at y=80 (above person box)
        vest_box = (140, 60, 260, 100)
        is_cand, _, _ = associator.is_candidate_region(self.person_box, vest_box, "Safety Vest")
        assert is_cand is False

    def test_ppe_horizontally_far_is_rejected(self, associator: PPEAssociator):
        # Centered at x=500, person ends at x=300 (exceeds horizontal margin)
        hardhat_box = (450, 110, 550, 150)
        is_cand, rx, _ = associator.is_candidate_region(self.person_box, hardhat_box, "Hardhat")
        assert is_cand is False


# ---------------------------------------------------------------------------
# Single Person Association Tests
# ---------------------------------------------------------------------------


class TestSinglePersonAssociation:
    """Test full association for a single person with various PPE combinations."""

    @pytest.fixture
    def associator(self) -> PPEAssociator:
        return PPEAssociator()

    person = Detection(100, 100, 300, 500, 0, "Person", 0.95)

    def test_person_with_hardhat_and_safety_vest(self, associator: PPEAssociator):
        hardhat = Detection(160, 110, 240, 150, 2, "Hardhat", 0.90)
        vest = Detection(140, 180, 260, 320, 11, "Safety Vest", 0.88)

        result = associator.associate([self.person], [hardhat, vest])

        assert result.total_persons == 1
        p = result.persons[0]
        assert len(p.hardhats) == 1
        assert len(p.safety_vests) == 1
        assert p.has_hardhat is True
        assert p.has_safety_vest is True
        assert len(result.unassigned_ppe) == 0

    def test_person_with_no_hardhat_and_vest(self, associator: PPEAssociator):
        no_hardhat = Detection(160, 110, 240, 150, 5, "NO-Hardhat", 0.85)
        vest = Detection(140, 180, 260, 320, 11, "Safety Vest", 0.88)

        result = associator.associate([self.person], [no_hardhat, vest])

        p = result.persons[0]
        assert len(p.no_hardhats) == 1
        assert len(p.safety_vests) == 1
        assert p.has_hardhat is False
        assert p.has_safety_vest is True

    def test_person_with_hardhat_and_no_vest(self, associator: PPEAssociator):
        hardhat = Detection(160, 110, 240, 150, 2, "Hardhat", 0.92)
        no_vest = Detection(140, 180, 260, 320, 7, "NO-Safety Vest", 0.80)

        result = associator.associate([self.person], [hardhat, no_vest])

        p = result.persons[0]
        assert p.has_hardhat is True
        assert p.has_safety_vest is False

    def test_person_with_no_hardhat_and_no_vest(self, associator: PPEAssociator):
        no_hardhat = Detection(160, 110, 240, 150, 5, "NO-Hardhat", 0.90)
        no_vest = Detection(140, 180, 260, 320, 7, "NO-Safety Vest", 0.85)

        result = associator.associate([self.person], [no_hardhat, no_vest])

        p = result.persons[0]
        assert p.has_hardhat is False
        assert p.has_safety_vest is False


# ---------------------------------------------------------------------------
# Multiple Persons Disambiguation Tests
# ---------------------------------------------------------------------------


class TestMultiplePersonsDisambiguation:
    """Test spatial disambiguation when multiple workers are present."""

    @pytest.fixture
    def associator(self) -> PPEAssociator:
        return PPEAssociator()

    def test_two_workers_side_by_side(self, associator: PPEAssociator):
        # Worker A at x in [100, 300]
        person_a = Detection(100, 100, 300, 500, 0, "Person", 0.95)
        # Worker B at x in [450, 650]
        person_b = Detection(450, 100, 650, 500, 0, "Person", 0.95)

        # Hardhat A directly over Worker A (center cx=200, cy=130)
        hat_a = Detection(160, 110, 240, 150, 2, "Hardhat", 0.90)
        # Vest B directly over Worker B (center cx=550, cy=250)
        vest_b = Detection(490, 180, 610, 320, 11, "Safety Vest", 0.88)

        result = associator.associate([person_a, person_b], [hat_a, vest_b])

        res_a = result.persons[0]
        res_b = result.persons[1]

        # Worker A has hat, no vest
        assert res_a.has_hardhat is True
        assert res_a.has_safety_vest is False

        # Worker B has vest, no hat
        assert res_b.has_hardhat is False
        assert res_b.has_safety_vest is True

    def test_overlapping_candidates_highest_score_wins(self, associator: PPEAssociator):
        """Worker A and B partially overlap horizontally. A PPE centered on A must go to A."""
        # Worker A: [100, 100, 300, 500] (center cx=200)
        person_a = Detection(100, 100, 300, 500, 0, "Person", 0.95)
        # Worker B: [220, 100, 420, 500] (center cx=320)
        person_b = Detection(220, 100, 420, 500, 0, "Person", 0.95)

        # Hardhat at [170, 110, 230, 150] (center cx=200, dead center on Worker A)
        hat_a = Detection(170, 110, 230, 150, 2, "Hardhat", 0.90)

        result = associator.associate([person_a, person_b], [hat_a])

        # Worker A wins the hat because score on A is significantly higher
        assert len(result.persons[0].hardhats) == 1
        assert len(result.persons[1].hardhats) == 0

    def test_exclusive_one_to_one_matching(self, associator: PPEAssociator):
        """A single hardhat is never duplicated across multiple workers."""
        person_a = Detection(100, 100, 300, 500, 0, "Person", 0.95)
        person_b = Detection(250, 100, 450, 500, 0, "Person", 0.95)
        hat = Detection(220, 110, 280, 150, 2, "Hardhat", 0.90)

        result = associator.associate([person_a, person_b], [hat])

        # Exactly 1 total hardhat assigned across all persons
        total_assigned_hats = sum(len(p.hardhats) for p in result.persons)
        assert total_assigned_hats == 1


# ---------------------------------------------------------------------------
# Unassigned PPE Tests
# ---------------------------------------------------------------------------


class TestUnassignedPPE:
    """Test gear not associated with any person."""

    def test_lone_hardhat_on_ground_is_unassigned(self):
        associator = PPEAssociator()
        person = Detection(100, 100, 300, 500, 0, "Person", 0.95)

        # Hardhat sitting far away at x=[800, 900], y=[700, 800]
        isolated_hat = Detection(800, 700, 900, 800, 2, "Hardhat", 0.92)

        result = associator.associate([person], [isolated_hat])

        assert result.persons[0].has_hardhat is False
        assert len(result.unassigned_ppe) == 1
        assert result.unassigned_ppe[0] == isolated_hat

    def test_all_ppe_unassigned_when_no_persons(self):
        associator = PPEAssociator()
        hat = Detection(100, 100, 200, 200, 2, "Hardhat", 0.90)
        vest = Detection(100, 200, 200, 300, 11, "Safety Vest", 0.85)

        result = associator.associate([], [hat, vest])
        assert result.total_persons == 0
        assert len(result.unassigned_ppe) == 2


# ---------------------------------------------------------------------------
# Conflicting Detection Resolution Tests
# ---------------------------------------------------------------------------


class TestConflictingDetectionResolution:
    """Verify tie-breaking when both Hardhat and NO-Hardhat are assigned."""

    def test_higher_confidence_positive_wins(self):
        p = AssociatedPerson(person_id="p1", person_box=(100, 100, 300, 500))
        p.hardhats.append(Detection(160, 110, 240, 150, 2, "Hardhat", 0.92))
        p.no_hardhats.append(Detection(160, 110, 240, 150, 5, "NO-Hardhat", 0.45))

        assert p.has_hardhat is True

    def test_higher_confidence_negative_wins(self):
        p = AssociatedPerson(person_id="p2", person_box=(100, 100, 300, 500))
        p.hardhats.append(Detection(160, 110, 240, 150, 2, "Hardhat", 0.35))
        p.no_hardhats.append(Detection(160, 110, 240, 150, 5, "NO-Hardhat", 0.88))

        assert p.has_hardhat is False


# ---------------------------------------------------------------------------
# Compliance Engine Integration Tests
# ---------------------------------------------------------------------------


class TestComplianceIntegration:
    """Verify end-to-end integration: Detection -> Association -> ComplianceEngine."""

    def test_end_to_end_association_to_compliance(self):
        associator = PPEAssociator()
        engine = ComplianceEngine()

        # Two workers:
        # Worker 1: Wearing Hardhat and Safety Vest => Should be COMPLIANT
        worker1 = Detection(100, 100, 300, 500, 0, "Person", 0.95)
        hat1 = Detection(160, 110, 240, 150, 2, "Hardhat", 0.92)
        vest1 = Detection(140, 180, 260, 320, 11, "Safety Vest", 0.88)

        # Worker 2: Wearing Safety Vest, but NO Hardhat => Should be NON_COMPLIANT
        worker2 = Detection(500, 100, 700, 500, 0, "Person", 0.95)
        no_hat2 = Detection(560, 110, 640, 150, 5, "NO-Hardhat", 0.85)
        vest2 = Detection(540, 180, 660, 320, 11, "Safety Vest", 0.88)

        # Run association
        assoc_result = associator.associate(
            [worker1, worker2],
            [hat1, vest1, no_hat2, vest2],
        )

        assert assoc_result.total_persons == 2

        # Convert to ComplianceEngine inputs
        compliance_inputs = assoc_result.to_compliance_inputs()
        assert len(compliance_inputs) == 2

        # Evaluate compliance
        comp_results = engine.evaluate_batch(compliance_inputs)

        # Worker 1
        assert comp_results[0].status == ComplianceStatus.COMPLIANT
        assert comp_results[0].is_compliant is True

        # Worker 2
        assert comp_results[1].status == ComplianceStatus.NON_COMPLIANT
        assert comp_results[1].is_compliant is False
        assert comp_results[1].missing_equipment == ["Hardhat"]

        # Aggregate summary
        summary = engine.generate_summary(comp_results)
        assert summary["total_persons"] == 2
        assert summary["compliant_count"] == 1
        assert summary["non_compliant_count"] == 1
        assert summary["compliance_rate"] == 50.0
