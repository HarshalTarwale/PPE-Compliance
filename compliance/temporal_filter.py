"""
Temporal Smoothing Filter for PPE Compliance.

Single-frame object detections often fluctuate or drop momentarily due to motion blur,
occlusion, or rapid head movement. This module maintains a sliding temporal window of
recent PPE observations for each tracked person and applies configurable voting logic
to compute stable, flicker-free PPE states and compliance outcomes.

Key features:
1. Configurable Sliding Window:
   Maintains the last N frames per person (e.g., window_size=10).
2. Configurable Voting Thresholds:
   Determines stable PPE presence if observed in >= threshold fraction of window frames.
3. Anti-Flicker Protection:
   Prevents isolated single-frame detection drops from immediately generating false warnings.
4. Extensible Equipment Support:
   Smooths Hardhat, Safety Vest, and any arbitrary safety equipment (e.g., Gloves, Mask).
5. Track Lifetime Management:
   Provides cleanup utilities for persons who exit the camera frame.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Ensure project root is importable
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from compliance.rules import (
    ComplianceEngine,
    ComplianceResult,
    ComplianceStatus,
    PersonPPEInfo,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TemporalFilterConfig:
    """Configurable parameters for the temporal smoothing filter."""

    # Number of recent frames to retain per person
    window_size: int = 10

    # Fraction of frames in window required to declare an item stable PRESENT
    # e.g., 0.60 on a 10-frame window requires >= 6 frames
    presence_threshold: float = 0.60

    # Minimum observations required before producing a stable state (defaults to 1)
    min_observations: int = 1

    # Threshold for direct compliance status voting (fraction of compliant frames)
    compliance_threshold: float = 0.60

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if not (0.0 <= self.presence_threshold <= 1.0):
            raise ValueError(
                f"presence_threshold must be in [0.0, 1.0], got {self.presence_threshold}"
            )
        if not (0.0 <= self.compliance_threshold <= 1.0):
            raise ValueError(
                f"compliance_threshold must be in [0.0, 1.0], got {self.compliance_threshold}"
            )


# ---------------------------------------------------------------------------
# Internal & Output Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameObservation:
    """A single frame's PPE observation for an individual person."""

    has_hardhat: bool
    has_safety_vest: bool
    equipment: Dict[str, bool] = field(default_factory=dict)
    is_compliant: Optional[bool] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SmoothedPersonState:
    """Temporally filtered and stabilized PPE state for a tracked person."""

    person_id: str
    has_hardhat: bool
    has_safety_vest: bool
    is_compliant: bool
    status: ComplianceStatus
    history_length: int
    window_size: int
    presence_ratios: Dict[str, float] = field(default_factory=dict)
    raw_votes: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # item -> (votes, total)
    missing_equipment: List[str] = field(default_factory=list)
    equipment: Dict[str, bool] = field(default_factory=dict)
    evaluated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def summary(self) -> str:
        """Human-readable one-line summary."""
        if self.is_compliant:
            return f"{self.person_id}: {ComplianceStatus.COMPLIANT.value} ({self.history_length}/{self.window_size} frames)"
        missing_str = ", ".join(self.missing_equipment) or "None"
        return f"{self.person_id}: {ComplianceStatus.NON_COMPLIANT.value} (Missing: {missing_str})"

    def to_person_ppe_info(self) -> PersonPPEInfo:
        """Convert smoothed state into PersonPPEInfo for rule engine evaluation."""
        return PersonPPEInfo(
            person_id=self.person_id,
            has_hardhat=self.has_hardhat,
            has_safety_vest=self.has_safety_vest,
            equipment=dict(self.equipment),
            metadata={
                "temporal_smoothing": True,
                "history_length": self.history_length,
                "window_size": self.window_size,
                "presence_ratios": dict(self.presence_ratios),
                "raw_votes": dict(self.raw_votes),
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize smoothed state to dictionary."""
        return {
            "person_id": self.person_id,
            "has_hardhat": self.has_hardhat,
            "has_safety_vest": self.has_safety_vest,
            "is_compliant": self.is_compliant,
            "status": self.status.value,
            "summary": self.summary,
            "history_length": self.history_length,
            "window_size": self.window_size,
            "presence_ratios": self.presence_ratios,
            "raw_votes": {k: list(v) for k, v in self.raw_votes.items()},
            "missing_equipment": self.missing_equipment,
            "equipment": self.equipment,
            "evaluated_at": self.evaluated_at,
        }


# ---------------------------------------------------------------------------
# Temporal Smoothing Filter
# ---------------------------------------------------------------------------


class PPETemporalFilter:
    """Sliding-window temporal filter for tracked persons' PPE observations."""

    def __init__(
        self,
        config: Optional[TemporalFilterConfig] = None,
        compliance_engine: Optional[ComplianceEngine] = None,
    ) -> None:
        """Initialize the temporal filter.

        Args:
            config: Configurable window size and voting thresholds.
            compliance_engine: Optional compliance rule engine instance.
        """
        self.config = config or TemporalFilterConfig()
        self.engine = compliance_engine or ComplianceEngine()

        # History per person: person_id -> deque of FrameObservation
        self._tracks: Dict[str, deque[FrameObservation]] = {}

    @property
    def active_person_ids(self) -> List[str]:
        """List of person IDs currently tracked in memory."""
        return list(self._tracks.keys())

    def get_history_length(self, person_id: str) -> int:
        """Return the number of recorded frames for a person."""
        return len(self._tracks.get(person_id, []))

    def reset(self, person_id: Optional[str] = None) -> None:
        """Clear history for a specific person or all persons."""
        if person_id is not None:
            self._tracks.pop(person_id, None)
        else:
            self._tracks.clear()

    def remove_inactive(self, active_ids: Set[str]) -> None:
        """Remove history for persons no longer detected in the scene."""
        stale_ids = [pid for pid in self._tracks if pid not in active_ids]
        for pid in stale_ids:
            del self._tracks[pid]

    # -----------------------------------------------------------------------
    # Update & Voting Logic
    # -----------------------------------------------------------------------

    def update(
        self,
        person: Union[PersonPPEInfo, Dict[str, Any]],
        person_id: Optional[str] = None,
    ) -> SmoothedPersonState:
        """Add a new frame observation and compute the smoothed PPE state.

        Args:
            person: PersonPPEInfo instance or dict describing the current frame observation.
            person_id: Optional ID override (used if person is dict without person_id).

        Returns:
            SmoothedPersonState with stabilized PPE and compliance status.
        """
        # 1. Parse observation
        if isinstance(person, PersonPPEInfo):
            pid = person.person_id or person_id or "person_default"
            has_hat = person.has_hardhat
            has_vest = person.has_safety_vest
            equip = dict(person.equipment)
            # Evaluate compliance for this single frame
            frame_comp = self.engine.evaluate(person).is_compliant
        elif isinstance(person, dict):
            pid = str(person.get("person_id") or person_id or "person_default")
            has_hat = bool(person.get("has_hardhat", False))
            has_vest = bool(person.get("has_safety_vest", False))
            equip = dict(person.get("equipment", {}))
            equip["Hardhat"] = has_hat
            equip["Safety Vest"] = has_vest
            frame_comp = person.get("is_compliant")
        else:
            raise TypeError(f"Unsupported person observation type: {type(person)}")

        # Ensure canonical equipment keys exist
        equip.setdefault("Hardhat", has_hat)
        equip.setdefault("Safety Vest", has_vest)

        obs = FrameObservation(
            has_hardhat=has_hat,
            has_safety_vest=has_vest,
            equipment=equip,
            is_compliant=frame_comp,
        )

        # 2. Append to person's sliding window
        if pid not in self._tracks:
            self._tracks[pid] = deque(maxlen=self.config.window_size)

        window = self._tracks[pid]
        window.append(obs)

        # 3. Calculate smoothed state via voting
        return self._compute_smoothed_state(pid, window)

    def update_batch(
        self,
        persons: List[Union[PersonPPEInfo, Dict[str, Any]]],
    ) -> List[SmoothedPersonState]:
        """Update multiple person observations simultaneously."""
        return [self.update(p) for p in persons]

    def get_smoothed_state(self, person_id: str) -> Optional[SmoothedPersonState]:
        """Return the current smoothed state for a person without adding a new frame."""
        if person_id not in self._tracks or len(self._tracks[person_id]) == 0:
            return None
        return self._compute_smoothed_state(person_id, self._tracks[person_id])

    # -----------------------------------------------------------------------
    # Internal Voting Engine
    # -----------------------------------------------------------------------

    def _compute_smoothed_state(
        self,
        person_id: str,
        window: deque[FrameObservation],
    ) -> SmoothedPersonState:
        """Perform voting aggregation across all observations in the sliding window."""
        n_frames = len(window)
        thresh = self.config.presence_threshold

        # Collect all equipment items mentioned across the window
        all_item_keys: Set[str] = {"Hardhat", "Safety Vest"}
        for obs in window:
            all_item_keys.update(obs.equipment.keys())

        presence_ratios: Dict[str, float] = {}
        raw_votes: Dict[str, Tuple[int, int]] = {}
        smoothed_equipment: Dict[str, bool] = {}

        for item in all_item_keys:
            # Count positive observations
            if item == "Hardhat":
                votes = sum(1 for obs in window if obs.has_hardhat)
            elif item == "Safety Vest":
                votes = sum(1 for obs in window if obs.has_safety_vest)
            else:
                votes = sum(1 for obs in window if obs.equipment.get(item, False))

            ratio = votes / n_frames if n_frames > 0 else 0.0
            is_present = ratio >= thresh

            presence_ratios[item] = round(ratio, 4)
            raw_votes[item] = (votes, n_frames)
            smoothed_equipment[item] = is_present

        # Canonical items
        stable_hardhat = smoothed_equipment.get("Hardhat", False)
        stable_vest = smoothed_equipment.get("Safety Vest", False)

        # Evaluate compliance on smoothed state
        smoothed_person_info = PersonPPEInfo(
            person_id=person_id,
            has_hardhat=stable_hardhat,
            has_safety_vest=stable_vest,
            equipment=smoothed_equipment,
        )
        compliance_eval: ComplianceResult = self.engine.evaluate(smoothed_person_info)

        return SmoothedPersonState(
            person_id=person_id,
            has_hardhat=stable_hardhat,
            has_safety_vest=stable_vest,
            is_compliant=compliance_eval.is_compliant,
            status=compliance_eval.status,
            history_length=n_frames,
            window_size=self.config.window_size,
            presence_ratios=presence_ratios,
            raw_votes=raw_votes,
            missing_equipment=compliance_eval.missing_equipment,
            equipment=smoothed_equipment,
        )
