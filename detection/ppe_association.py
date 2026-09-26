"""
Person-to-PPE Spatial Association Engine.

Determines which Personal Protective Equipment (PPE) detections belong to which
detected persons using deterministic geometric analysis and anatomical priors.

Core concepts:
1. Bounding-Box Center & Proximity:
   Calculates PPE centroid and checks containment and horizontal alignment with candidate persons.
2. Relative Vertical Positioning (Anatomical Priors):
   - Hardhat / NO-Hardhat: Assigned to the head region (top ~0-35% of person height).
   - Safety Vest / NO-Safety Vest: Assigned to the torso region (~15-70% of person height).
3. Candidate Scoring & Disambiguation:
   Computes a normalized geometric score (containment + proximity + confidence).
   If multiple persons are candidate matches for a single PPE item, assigns it
   to the highest-scoring candidate (exclusive 1-to-1 PPE assignment).
4. Structured Output:
   Produces AssociatedPerson objects that seamlessly convert to PersonPPEInfo
   for direct evaluation with the ComplianceEngine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Ensure project root is importable
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from compliance.rules import PersonPPEInfo
from detection.detector import Detection

logger = logging.getLogger("ppe_association")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AssociationConfig:
    """Configurable thresholds for geometric person-PPE association."""

    # Head region (Hardhat / NO-Hardhat) relative to person [0.0 = top, 1.0 = bottom]
    head_min_rel_y: float = -0.20   # Allows hardhat extending slightly above person box
    head_max_rel_y: float = 0.38   # Upper 38% of person box
    head_ideal_rel_y: float = 0.08  # Anatomical center of head/hat

    # Torso region (Safety Vest / NO-Safety Vest) relative to person
    torso_min_rel_y: float = 0.12  # Below neck
    torso_max_rel_y: float = 0.72  # Down to waist/hips
    torso_ideal_rel_y: float = 0.38  # Anatomical center of chest/vest

    # Horizontal expansion margin (fraction of person width allowed outside person box)
    horizontal_margin_ratio: float = 0.20

    # Minimum score required for an association to be considered valid
    min_association_score: float = 0.20

    # Score weighting
    weight_containment: float = 0.45
    weight_proximity: float = 0.40
    weight_confidence: float = 0.15


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class AssociatedPerson:
    """A detected person with their spatially associated PPE items."""

    person_id: str
    person_box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float = 1.0
    hardhats: List[Detection] = field(default_factory=list)
    no_hardhats: List[Detection] = field(default_factory=list)
    safety_vests: List[Detection] = field(default_factory=list)
    no_safety_vests: List[Detection] = field(default_factory=list)
    other_ppe: List[Detection] = field(default_factory=list)

    @property
    def has_hardhat(self) -> bool:
        """True if person has an associated Hardhat without a higher-confidence NO-Hardhat."""
        if not self.hardhats:
            return False
        if not self.no_hardhats:
            return True
        best_pos = max(self.hardhats, key=lambda d: d.confidence)
        best_neg = max(self.no_hardhats, key=lambda d: d.confidence)
        return best_pos.confidence >= best_neg.confidence

    @property
    def has_safety_vest(self) -> bool:
        """True if person has an associated Safety Vest without a higher-confidence NO-Safety Vest."""
        if not self.safety_vests:
            return False
        if not self.no_safety_vests:
            return True
        best_pos = max(self.safety_vests, key=lambda d: d.confidence)
        best_neg = max(self.no_safety_vests, key=lambda d: d.confidence)
        return best_pos.confidence >= best_neg.confidence

    def to_person_ppe_info(self) -> PersonPPEInfo:
        """Convert this associated person to PersonPPEInfo for compliance evaluation."""
        equipment: Dict[str, bool] = {
            "Hardhat": self.has_hardhat,
            "Safety Vest": self.has_safety_vest,
        }
        for item in self.other_ppe:
            equipment[item.class_name] = True

        return PersonPPEInfo(
            person_id=self.person_id,
            has_hardhat=self.has_hardhat,
            has_safety_vest=self.has_safety_vest,
            equipment=equipment,
            metadata={
                "person_box": self.person_box,
                "confidence": self.confidence,
                "hardhat_count": len(self.hardhats),
                "no_hardhat_count": len(self.no_hardhats),
                "safety_vest_count": len(self.safety_vests),
                "no_safety_vest_count": len(self.no_safety_vests),
            },
        )


@dataclass
class AssociationResult:
    """Aggregate result of person-PPE spatial association."""

    persons: List[AssociatedPerson] = field(default_factory=list)
    unassigned_ppe: List[Detection] = field(default_factory=list)

    @property
    def total_persons(self) -> int:
        return len(self.persons)

    @property
    def total_assigned_ppe(self) -> int:
        return sum(
            len(p.hardhats) + len(p.no_hardhats) + len(p.safety_vests) + len(p.no_safety_vests) + len(p.other_ppe)
            for p in self.persons
        )

    def to_compliance_inputs(self) -> List[PersonPPEInfo]:
        """Convert all associated persons to PersonPPEInfo for the compliance engine."""
        return [p.to_person_ppe_info() for p in self.persons]


# ---------------------------------------------------------------------------
# Association Engine
# ---------------------------------------------------------------------------


class PPEAssociator:
    """Performs geometric association between Person and PPE bounding boxes."""

    def __init__(self, config: Optional[AssociationConfig] = None) -> None:
        """Initialize associator with configurable thresholds.

        Args:
            config: Optional AssociationConfig instance. Defaults to standard configuration.
        """
        self.config = config or AssociationConfig()

    # -----------------------------------------------------------------------
    # Geometric Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def calculate_box_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Return center (cx, cy) of bounding box (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def calculate_intersection_area(
        box_a: Tuple[int, int, int, int],
        box_b: Tuple[int, int, int, int],
    ) -> float:
        """Return pixel area of intersection between box_a and box_b."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, float(ix2 - ix1))
        ih = max(0.0, float(iy2 - iy1))
        return iw * ih

    @staticmethod
    def calculate_containment_ratio(
        person_box: Tuple[int, int, int, int],
        ppe_box: Tuple[int, int, int, int],
    ) -> float:
        """Calculate fraction of the PPE bounding box contained within the person box."""
        px1, py1, px2, py2 = ppe_box
        ppe_area = max(1.0, float(px2 - px1) * float(py2 - py1))
        inter_area = PPEAssociator.calculate_intersection_area(person_box, ppe_box)
        return min(1.0, inter_area / ppe_area)

    def is_candidate_region(
        self,
        person_box: Tuple[int, int, int, int],
        ppe_box: Tuple[int, int, int, int],
        class_name: str,
    ) -> Tuple[bool, float, float]:
        """Check if PPE center falls within the person's expected anatomical region.

        Returns:
            (is_candidate, rel_x, rel_y)
        """
        px1, py1, px2, py2 = person_box
        pw = max(1.0, float(px2 - px1))
        ph = max(1.0, float(py2 - py1))

        cx, cy = self.calculate_box_center(ppe_box)

        # Relative coordinates relative to person box
        rel_x = (cx - px1) / pw
        rel_y = (cy - py1) / ph

        # Horizontal check: must be within [0 - margin, 1 + margin]
        margin = self.config.horizontal_margin_ratio
        if rel_x < -margin or rel_x > (1.0 + margin):
            return False, rel_x, rel_y

        # Vertical anatomical check based on class
        norm_name = class_name.strip()
        if norm_name in ("Hardhat", "NO-Hardhat", "Mask", "NO-Mask"):
            # Head region
            if self.config.head_min_rel_y <= rel_y <= self.config.head_max_rel_y:
                return True, rel_x, rel_y
        elif norm_name in ("Safety Vest", "NO-Safety Vest", "Gloves"):
            # Torso / Upper Body region
            if self.config.torso_min_rel_y <= rel_y <= self.config.torso_max_rel_y:
                return True, rel_x, rel_y
        else:
            # Generic PPE: allowed anywhere within person box
            if -margin <= rel_y <= (1.0 + margin):
                return True, rel_x, rel_y

        return False, rel_x, rel_y

    def calculate_score(
        self,
        person_box: Tuple[int, int, int, int],
        ppe_detection: Detection,
        rel_x: float,
        rel_y: float,
    ) -> float:
        """Calculate geometric association score between a person and a PPE detection."""
        # 1. Containment score (0.0 to 1.0)
        containment = self.calculate_containment_ratio(person_box, ppe_detection.box)

        # 2. Proximity / Alignment score (0.0 to 1.0)
        # Horizontal alignment: penalty for distance from center (rel_x = 0.5)
        dist_x = abs(rel_x - 0.5) * 2.0  # 0 at center, 1 at edge
        h_score = max(0.0, 1.0 - dist_x / (1.0 + 2.0 * self.config.horizontal_margin_ratio))

        # Vertical alignment: penalty for distance from anatomical ideal
        norm_name = ppe_detection.class_name.strip()
        if norm_name in ("Hardhat", "NO-Hardhat", "Mask", "NO-Mask"):
            ideal_y = self.config.head_ideal_rel_y
            span_y = max(0.1, self.config.head_max_rel_y - self.config.head_min_rel_y)
        else:
            ideal_y = self.config.torso_ideal_rel_y
            span_y = max(0.1, self.config.torso_max_rel_y - self.config.torso_min_rel_y)

        dist_y = abs(rel_y - ideal_y) / (span_y / 2.0)
        v_score = max(0.0, 1.0 - dist_y)

        proximity_score = 0.5 * h_score + 0.5 * v_score

        # 3. Confidence score (0.0 to 1.0)
        conf_score = max(0.0, min(1.0, ppe_detection.confidence))

        # Combined weighted score
        score = (
            self.config.weight_containment * containment
            + self.config.weight_proximity * proximity_score
            + self.config.weight_confidence * conf_score
        )
        return float(score)

    # -----------------------------------------------------------------------
    # Main Association Routine
    # -----------------------------------------------------------------------

    def associate(
        self,
        person_detections: List[Detection],
        ppe_detections: List[Detection],
    ) -> AssociationResult:
        """Associate PPE detections to detected persons using spatial geometry.

        Args:
            person_detections: List of Person detections.
            ppe_detections: List of PPE detections (Hardhat, Safety Vest, NO-*, etc.).

        Returns:
            AssociationResult containing list of AssociatedPerson objects and unassigned PPE.
        """
        # If no persons detected, all PPE is unassigned
        if not person_detections:
            return AssociationResult(persons=[], unassigned_ppe=list(ppe_detections))

        # Initialize AssociatedPerson containers
        persons = [
            AssociatedPerson(
                person_id=f"person_{i + 1}",
                person_box=p.box,
                confidence=p.confidence,
            )
            for i, p in enumerate(person_detections)
        ]

        unassigned_ppe: List[Detection] = []

        # For each PPE detection, determine candidate persons and score them
        for ppe in ppe_detections:
            candidates: List[Tuple[int, float]] = []  # (person_index, score)

            for p_idx, person in enumerate(persons):
                is_candidate, rx, ry = self.is_candidate_region(
                    person.person_box,
                    ppe.box,
                    ppe.class_name,
                )
                if is_candidate:
                    score = self.calculate_score(person.person_box, ppe, rx, ry)
                    if score >= self.config.min_association_score:
                        candidates.append((p_idx, score))

            if not candidates:
                # No person satisfied spatial criteria
                unassigned_ppe.append(ppe)
                continue

            # Pick highest-scoring candidate person (exclusive 1-to-1 matching)
            candidates.sort(key=lambda item: item[1], reverse=True)
            best_person_idx, best_score = candidates[0]
            matched_person = persons[best_person_idx]

            # Assign to appropriate bucket on matched person
            cname = ppe.class_name.strip()
            if cname == "Hardhat":
                matched_person.hardhats.append(ppe)
            elif cname == "NO-Hardhat":
                matched_person.no_hardhats.append(ppe)
            elif cname == "Safety Vest":
                matched_person.safety_vests.append(ppe)
            elif cname == "NO-Safety Vest":
                matched_person.no_safety_vests.append(ppe)
            else:
                matched_person.other_ppe.append(ppe)

        return AssociationResult(persons=persons, unassigned_ppe=unassigned_ppe)
