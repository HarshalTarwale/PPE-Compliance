"""
PPE Compliance Rule Engine.

Pure Python compliance evaluation engine that assesses whether individuals on a site
satisfy mandated Personal Protective Equipment (PPE) standards.

By default, checks for:
  - Hardhat
  - Safety Vest

Compliance Rules:
  - Hardhat present + Safety Vest present  => COMPLIANT
  - Hardhat missing + Safety Vest present  => NON_COMPLIANT (Missing: Hardhat)
  - Hardhat present + Safety Vest missing  => NON_COMPLIANT (Missing: Safety Vest)
  - Hardhat missing + Safety Vest missing  => NON_COMPLIANT (Missing: Hardhat, Safety Vest)

Extensible architecture allows easily configuring, adding, or modifying required safety equipment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union


# ---------------------------------------------------------------------------
# Enums and Constants
# ---------------------------------------------------------------------------


class ComplianceStatus(str, Enum):
    """Enumeration of compliance statuses."""

    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"


class DefaultPPE(str, Enum):
    """Canonical names of standard PPE items."""

    HARDHAT = "Hardhat"
    SAFETY_VEST = "Safety Vest"
    MASK = "Mask"
    GLOVES = "Gloves"
    SAFETY_BOOTS = "Safety Boots"
    EYE_PROTECTION = "Eye Protection"


DEFAULT_REQUIRED_PPE: List[str] = [
    DefaultPPE.HARDHAT.value,
    DefaultPPE.SAFETY_VEST.value,
]


# ---------------------------------------------------------------------------
# Input Data Structures
# ---------------------------------------------------------------------------


@dataclass
class PPEItem:
    """Represents a specific PPE item observation associated with a person."""

    name: str
    present: bool = True
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PersonPPEInfo:
    """Structured person-level PPE information for compliance evaluation.

    Supports both simple convenience booleans (has_hardhat, has_safety_vest)
    and an extensible equipment mapping / item list.
    """

    person_id: Optional[str] = None
    has_hardhat: bool = False
    has_safety_vest: bool = False
    equipment: Dict[str, bool] = field(default_factory=dict)
    items: List[PPEItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Synchronize convenience booleans with the equipment mapping."""
        # Ensure hardhat status is recorded in equipment map
        if "Hardhat" not in self.equipment and self.has_hardhat:
            self.equipment["Hardhat"] = True
        elif self.equipment.get("Hardhat") is True:
            self.has_hardhat = True

        # Ensure safety vest status is recorded in equipment map
        if "Safety Vest" not in self.equipment and self.has_safety_vest:
            self.equipment["Safety Vest"] = True
        elif self.equipment.get("Safety Vest") is True:
            self.has_safety_vest = True

        # Incorporate any explicit items
        for item in self.items:
            self.equipment[item.name] = item.present
            if item.name == "Hardhat":
                self.has_hardhat = item.present
            elif item.name == "Safety Vest":
                self.has_safety_vest = item.present

    def has_item(self, item_name: str) -> bool:
        """Check whether a specific PPE item is confirmed present.

        Args:
            item_name: Name of the PPE item (case-insensitive search supported).

        Returns:
            True if item is present, False otherwise.
        """
        # Exact match
        if item_name in self.equipment:
            return bool(self.equipment[item_name])

        # Case-insensitive / normalized lookup
        norm_target = item_name.strip().lower()
        for k, v in self.equipment.items():
            if k.strip().lower() == norm_target:
                return bool(v)

        return False

    @classmethod
    def from_detected_classes(
        cls,
        detected_classes: List[str],
        person_id: Optional[str] = None,
    ) -> PersonPPEInfo:
        """Construct PersonPPEInfo from a list of detected class names.

        Handles both positive and negative YOLO class labels
        (e.g., 'Hardhat' vs 'NO-Hardhat', 'Safety Vest' vs 'NO-Safety Vest').

        Args:
            detected_classes: List of class labels associated with the person.
            person_id: Optional identifier for the person.

        Returns:
            Populated PersonPPEInfo instance.
        """
        class_set = set(detected_classes)

        # Hardhat logic: present if 'Hardhat' detected and 'NO-Hardhat' not overriding
        has_hardhat = "Hardhat" in class_set and "NO-Hardhat" not in class_set

        # Safety vest logic: present if 'Safety Vest' detected and 'NO-Safety Vest' not overriding
        has_vest = "Safety Vest" in class_set and "NO-Safety Vest" not in class_set

        equipment: Dict[str, bool] = {
            "Hardhat": has_hardhat,
            "Safety Vest": has_vest,
        }

        # Record other equipment
        for name in class_set:
            if not name.startswith("NO-") and name not in equipment:
                equipment[name] = True
            elif name.startswith("NO-"):
                base_name = name[3:]  # Strip 'NO-'
                equipment[base_name] = False

        return cls(
            person_id=person_id,
            has_hardhat=has_hardhat,
            has_safety_vest=has_vest,
            equipment=equipment,
        )


# ---------------------------------------------------------------------------
# Output Data Structures
# ---------------------------------------------------------------------------


@dataclass
class ComplianceResult:
    """Result of compliance rule evaluation for an individual person."""

    person_id: Optional[str]
    status: ComplianceStatus
    is_compliant: bool
    required_equipment: List[str]
    present_equipment: List[str]
    missing_equipment: List[str]
    evaluated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        """Return a human-readable one-line compliance summary."""
        if self.is_compliant:
            return ComplianceStatus.COMPLIANT.value
        missing_str = ", ".join(self.missing_equipment)
        return f"{ComplianceStatus.NON_COMPLIANT.value} (Missing: {missing_str})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a plain serializable dictionary."""
        return {
            "person_id": self.person_id,
            "status": self.status.value,
            "is_compliant": self.is_compliant,
            "summary": self.summary,
            "required_equipment": self.required_equipment,
            "present_equipment": self.present_equipment,
            "missing_equipment": self.missing_equipment,
            "evaluated_at": self.evaluated_at,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Compliance Rule Engine
# ---------------------------------------------------------------------------


class ComplianceEngine:
    """Pure Python rule engine for evaluating PPE compliance.

    Configurable with required equipment and extensible for future safety standards.
    """

    def __init__(
        self,
        required_equipment: Optional[List[str]] = None,
    ) -> None:
        """Initialize the rule engine.

        Args:
            required_equipment: List of required PPE item names.
                                Defaults to ['Hardhat', 'Safety Vest'].
        """
        if required_equipment is not None:
            self._required_equipment = list(required_equipment)
        else:
            self._required_equipment = list(DEFAULT_REQUIRED_PPE)

    @property
    def required_equipment(self) -> List[str]:
        """List of currently required PPE items."""
        return list(self._required_equipment)

    def set_requirements(self, items: List[str]) -> None:
        """Replace the list of required equipment."""
        if not items:
            raise ValueError("Required equipment list cannot be empty.")
        self._required_equipment = list(items)

    def add_requirement(self, item_name: str) -> None:
        """Add a new required PPE item."""
        if item_name not in self._required_equipment:
            self._required_equipment.append(item_name)

    def remove_requirement(self, item_name: str) -> None:
        """Remove a required PPE item."""
        if item_name in self._required_equipment:
            self._required_equipment.remove(item_name)

    def evaluate(self, person: PersonPPEInfo) -> ComplianceResult:
        """Evaluate compliance for an individual person.

        Rules applied:
          - If all required equipment items are present => COMPLIANT
          - If any required equipment item is missing   => NON_COMPLIANT (with missing items listed)

        Args:
            person: PersonPPEInfo containing the person's detected/observed PPE.

        Returns:
            ComplianceResult with status, present items, and missing items.
        """
        present: List[str] = []
        missing: List[str] = []

        for req in self._required_equipment:
            if person.has_item(req):
                present.append(req)
            else:
                missing.append(req)

        is_compliant = len(missing) == 0
        status = (
            ComplianceStatus.COMPLIANT
            if is_compliant
            else ComplianceStatus.NON_COMPLIANT
        )

        return ComplianceResult(
            person_id=person.person_id,
            status=status,
            is_compliant=is_compliant,
            required_equipment=list(self._required_equipment),
            present_equipment=present,
            missing_equipment=missing,
            details={
                "total_required": len(self._required_equipment),
                "total_present": len(present),
                "total_missing": len(missing),
                "compliance_score": (
                    round(len(present) / len(self._required_equipment), 4)
                    if self._required_equipment
                    else 1.0
                ),
            },
        )

    def evaluate_batch(
        self,
        persons: List[PersonPPEInfo],
    ) -> List[ComplianceResult]:
        """Evaluate compliance for multiple persons.

        Args:
            persons: List of PersonPPEInfo instances.

        Returns:
            List of ComplianceResult instances.
        """
        return [self.evaluate(p) for p in persons]

    def generate_summary(
        self,
        results: List[ComplianceResult],
    ) -> Dict[str, Any]:
        """Generate high-level aggregate compliance metrics for a batch.

        Args:
            results: List of ComplianceResult objects.

        Returns:
            Dictionary containing compliance rate, counts, and missing item breakdowns.
        """
        total = len(results)
        if total == 0:
            return {
                "total_persons": 0,
                "compliant_count": 0,
                "non_compliant_count": 0,
                "compliance_rate": 0.0,
                "missing_breakdown": {},
            }

        compliant_count = sum(1 for r in results if r.is_compliant)
        non_compliant_count = total - compliant_count
        compliance_rate = round((compliant_count / total) * 100.0, 2)

        # Missing breakdown
        missing_counts: Dict[str, int] = {}
        for r in results:
            for item in r.missing_equipment:
                missing_counts[item] = missing_counts.get(item, 0) + 1

        return {
            "total_persons": total,
            "compliant_count": compliant_count,
            "non_compliant_count": non_compliant_count,
            "compliance_rate": compliance_rate,
            "missing_breakdown": missing_counts,
        }
