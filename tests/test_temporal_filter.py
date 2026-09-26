"""Unit tests for temporal smoothing filter (compliance/temporal_filter.py).

Tests:
  1. The 10-frame window example from specification:
     - 8/10 frames => stable hardhat PRESENT.
     - 2/10 frames => stable hardhat MISSING.
  2. Single-frame flicker resistance (1 bad frame does not immediately flip status).
  3. Support for Hardhat, Safety Vest, and overall Compliance Status.
  4. Configurable window sizes and presence thresholds.
  5. Multi-person track independence.
  6. Track lifetime management (reset, remove_inactive).
  7. Extensible equipment smoothing (e.g. Gloves, Mask).
  8. Pure Python independence check (no OpenCV, React, FastAPI, Torch).
"""

from __future__ import annotations

from pathlib import Path
import pytest

from compliance.rules import ComplianceEngine, ComplianceStatus, PersonPPEInfo
from compliance.temporal_filter import (
    PPETemporalFilter,
    SmoothedPersonState,
    TemporalFilterConfig,
)


# ---------------------------------------------------------------------------
# Specification Example Tests
# ---------------------------------------------------------------------------


class TestSpecificationExample:
    """Test the exact 10-frame scenario outlined in the user requirements."""

    def test_eight_out_of_ten_frames_is_stable_present(self):
        """Hardhat detected in 8/10 frames => stable hardhat PRESENT."""
        config = TemporalFilterConfig(window_size=10, presence_threshold=0.60)
        filt = PPETemporalFilter(config=config)

        # Feed 8 frames with hardhat, 2 frames without
        frames_pattern = [True] * 8 + [False] * 2
        for has_hat in frames_pattern:
            state = filt.update(
                PersonPPEInfo(person_id="worker_spec", has_hardhat=has_hat, has_safety_vest=True)
            )

        assert state.history_length == 10
        assert state.has_hardhat is True
        assert state.presence_ratios["Hardhat"] == 0.80
        assert state.raw_votes["Hardhat"] == (8, 10)
        assert state.is_compliant is True

    def test_two_out_of_ten_frames_is_stable_missing(self):
        """Hardhat detected in 2/10 frames => stable hardhat MISSING."""
        config = TemporalFilterConfig(window_size=10, presence_threshold=0.60)
        filt = PPETemporalFilter(config=config)

        # Feed 2 frames with hardhat, 8 frames without
        frames_pattern = [True] * 2 + [False] * 8
        for has_hat in frames_pattern:
            state = filt.update(
                PersonPPEInfo(person_id="worker_spec", has_hardhat=has_hat, has_safety_vest=True)
            )

        assert state.history_length == 10
        assert state.has_hardhat is False
        assert state.presence_ratios["Hardhat"] == 0.20
        assert state.raw_votes["Hardhat"] == (2, 10)
        assert state.is_compliant is False
        assert "Hardhat" in state.missing_equipment


# ---------------------------------------------------------------------------
# Anti-Flicker Protection Tests
# ---------------------------------------------------------------------------


class TestAntiFlickerProtection:
    """Verify that a single bad frame does not immediately disrupt compliance."""

    def test_single_frame_detection_drop_does_not_break_compliance(self):
        """Worker is compliant for 9 frames; 1 momentary missed detection should NOT cause false warning."""
        config = TemporalFilterConfig(window_size=10, presence_threshold=0.60)
        filt = PPETemporalFilter(config=config)

        # 9 frames with full PPE
        for _ in range(9):
            filt.update(PersonPPEInfo(person_id="worker_1", has_hardhat=True, has_safety_vest=True))

        # 10th frame: detector momentarily misses the hardhat (motion blur)
        flicker_frame = filt.update(
            PersonPPEInfo(person_id="worker_1", has_hardhat=False, has_safety_vest=True)
        )

        # Should remain stable COMPLIANT (9/10 = 90% >= 60%)
        assert flicker_frame.has_hardhat is True
        assert flicker_frame.has_safety_vest is True
        assert flicker_frame.is_compliant is True
        assert flicker_frame.status == ComplianceStatus.COMPLIANT

    def test_single_frame_false_positive_does_not_mask_violation(self):
        """Worker has no hardhat for 9 frames; 1 false positive glitch should NOT grant compliance."""
        config = TemporalFilterConfig(window_size=10, presence_threshold=0.60)
        filt = PPETemporalFilter(config=config)

        # 9 frames missing hardhat
        for _ in range(9):
            filt.update(PersonPPEInfo(person_id="worker_2", has_hardhat=False, has_safety_vest=True))

        # 10th frame: detector falsely sees a hardhat for 1 frame
        glitch_frame = filt.update(
            PersonPPEInfo(person_id="worker_2", has_hardhat=True, has_safety_vest=True)
        )

        # Should remain stable NON_COMPLIANT (1/10 = 10% < 60%)
        assert glitch_frame.has_hardhat is False
        assert glitch_frame.is_compliant is False
        assert glitch_frame.status == ComplianceStatus.NON_COMPLIANT
        assert "Hardhat" in glitch_frame.missing_equipment


# ---------------------------------------------------------------------------
# Transition and State Evolution Tests
# ---------------------------------------------------------------------------


class TestStateTransitions:
    """Verify smooth state transitions when a worker actually changes equipment."""

    def test_transition_from_non_compliant_to_compliant(self):
        """Worker puts on their hardhat: transitions once threshold (60%) is reached."""
        config = TemporalFilterConfig(window_size=10, presence_threshold=0.60)
        filt = PPETemporalFilter(config=config)

        # Worker begins without hardhat (10 frames)
        for _ in range(10):
            filt.update(PersonPPEInfo(person_id="worker_t", has_hardhat=False, has_safety_vest=True))

        assert filt.get_smoothed_state("worker_t").is_compliant is False

        # Worker puts on hardhat: observe positive frames one by one
        # At frame 1 to 5: votes are 1/10 to 5/10 (< 60%), remains Non-Compliant
        for i in range(1, 6):
            s = filt.update(PersonPPEInfo(person_id="worker_t", has_hardhat=True, has_safety_vest=True))
            assert s.is_compliant is False

        # At frame 6: votes are 6/10 (60% == threshold), switches to Compliant!
        s6 = filt.update(PersonPPEInfo(person_id="worker_t", has_hardhat=True, has_safety_vest=True))
        assert s6.is_compliant is True
        assert s6.has_hardhat is True


# ---------------------------------------------------------------------------
# Configurable Thresholds & Parameters Tests
# ---------------------------------------------------------------------------


class TestConfigurability:
    """Verify that different window sizes and voting thresholds work as expected."""

    def test_small_window_size_5(self):
        config = TemporalFilterConfig(window_size=5, presence_threshold=0.60)
        filt = PPETemporalFilter(config=config)

        # 3 out of 5 frames (60%) => PRESENT
        frames = [True, True, True, False, False]
        state = None
        for h in frames:
            state = filt.update(PersonPPEInfo(person_id="p5", has_hardhat=h, has_safety_vest=True))

        assert state.history_length == 5
        assert state.has_hardhat is True

    def test_strict_threshold_80_percent(self):
        # Requires 80% of frames to be present
        config = TemporalFilterConfig(window_size=10, presence_threshold=0.80)
        filt = PPETemporalFilter(config=config)

        # 7 out of 10 frames (70% < 80%) => MISSING under strict threshold
        for h in [True] * 7 + [False] * 3:
            state = filt.update(PersonPPEInfo(person_id="p_strict", has_hardhat=h, has_safety_vest=True))

        assert state.has_hardhat is False

    def test_invalid_config_raises_error(self):
        with pytest.raises(ValueError):
            TemporalFilterConfig(window_size=0)
        with pytest.raises(ValueError):
            TemporalFilterConfig(presence_threshold=1.5)
        with pytest.raises(ValueError):
            TemporalFilterConfig(presence_threshold=-0.1)


# ---------------------------------------------------------------------------
# Multi-Person Independence Tests
# ---------------------------------------------------------------------------


class TestMultiPersonIndependence:
    """Verify that histories of multiple tracked persons are completely independent."""

    def test_multiple_tracks(self):
        filt = PPETemporalFilter()

        # Person A is fully compliant
        # Person B has missing safety vest
        for _ in range(10):
            filt.update(PersonPPEInfo(person_id="person_A", has_hardhat=True, has_safety_vest=True))
            filt.update(PersonPPEInfo(person_id="person_B", has_hardhat=True, has_safety_vest=False))

        state_a = filt.get_smoothed_state("person_A")
        state_b = filt.get_smoothed_state("person_B")

        assert state_a.is_compliant is True
        assert state_b.is_compliant is False
        assert state_b.missing_equipment == ["Safety Vest"]
        assert len(filt.active_person_ids) == 2


# ---------------------------------------------------------------------------
# Track Lifetime Management Tests
# ---------------------------------------------------------------------------


class TestTrackLifetime:
    """Verify reset and cleanup of departed tracks."""

    def test_reset_specific_person(self):
        filt = PPETemporalFilter()
        filt.update(PersonPPEInfo(person_id="p1", has_hardhat=True, has_safety_vest=True))
        filt.update(PersonPPEInfo(person_id="p2", has_hardhat=True, has_safety_vest=True))

        filt.reset("p1")
        assert filt.get_smoothed_state("p1") is None
        assert filt.get_smoothed_state("p2") is not None

    def test_reset_all(self):
        filt = PPETemporalFilter()
        filt.update(PersonPPEInfo(person_id="p1", has_hardhat=True, has_safety_vest=True))
        filt.update(PersonPPEInfo(person_id="p2", has_hardhat=True, has_safety_vest=True))

        filt.reset()
        assert len(filt.active_person_ids) == 0

    def test_remove_inactive(self):
        filt = PPETemporalFilter()
        filt.update(PersonPPEInfo(person_id="p1", has_hardhat=True, has_safety_vest=True))
        filt.update(PersonPPEInfo(person_id="p2", has_hardhat=True, has_safety_vest=True))
        filt.update(PersonPPEInfo(person_id="p3", has_hardhat=True, has_safety_vest=True))

        # Only p1 and p3 are currently visible in the camera frame
        filt.remove_inactive(active_ids={"p1", "p3"})

        assert "p1" in filt.active_person_ids
        assert "p3" in filt.active_person_ids
        assert "p2" not in filt.active_person_ids


# ---------------------------------------------------------------------------
# Extensible Equipment Tests
# ---------------------------------------------------------------------------


class TestExtensibleEquipment:
    """Verify smoothing works for arbitrary equipment (Gloves, Mask)."""

    def test_smooth_custom_equipment(self):
        filt = PPETemporalFilter()

        # 8 frames with Gloves present, 2 without
        for present in [True] * 8 + [False] * 2:
            filt.update(
                PersonPPEInfo(
                    person_id="worker_gloves",
                    has_hardhat=True,
                    has_safety_vest=True,
                    equipment={"Gloves": present},
                )
            )

        state = filt.get_smoothed_state("worker_gloves")
        assert state.equipment["Gloves"] is True
        assert state.presence_ratios["Gloves"] == 0.80

    def test_to_person_ppe_info_conversion(self):
        filt = PPETemporalFilter()
        for _ in range(5):
            filt.update(PersonPPEInfo(person_id="conv_test", has_hardhat=True, has_safety_vest=True))

        state = filt.get_smoothed_state("conv_test")
        ppe_info = state.to_person_ppe_info()

        assert isinstance(ppe_info, PersonPPEInfo)
        assert ppe_info.person_id == "conv_test"
        assert ppe_info.has_hardhat is True
        assert ppe_info.has_safety_vest is True
        assert ppe_info.metadata["temporal_smoothing"] is True


# ---------------------------------------------------------------------------
# Pure Python Independence Test
# ---------------------------------------------------------------------------


class TestPurePythonIndependence:
    """Verify compliance/temporal_filter.py has no dependency on OpenCV, FastAPI, React, Torch."""

    def test_no_forbidden_imports(self):
        import compliance.temporal_filter as tf_mod

        source = Path(tf_mod.__file__).read_text(encoding="utf-8")
        assert "cv2" not in source
        assert "opencv" not in source.lower()
        assert "fastapi" not in source.lower()
        assert "torch" not in source.lower()
        assert "react" not in source.lower()
