"""Exercise the camera trigger with synthetic frames; no camera or API key needed."""

import unittest

import numpy as np

from camera_sorter import ChangeDetector


class ChangeDetectorTests(unittest.TestCase):
    def setUp(self):
        self.empty = np.zeros((160, 160, 3), dtype=np.uint8)
        self.item = self.empty.copy()
        self.item[30:90, 30:90] = 255
        self.replacement = self.empty.copy()
        self.replacement[70:130, 90:150] = 255
        self.detector = ChangeDetector(
            change_fraction=0.04,
            motion_fraction=0.01,
            reset_fraction=0.01,
            pixel_threshold=25,
            settle_seconds=1.0,
        )
        self.detector.calibrate(self.empty)

    def settle_first_item(self):
        self.assertFalse(self.detector.update(self.item, now=0.0).ready)
        settled = self.detector.update(self.item, now=1.1)
        self.assertTrue(settled.ready)
        self.detector.mark_submitted(self.item, now=1.1)
        return settled

    def test_unchanged_platform_and_low_amplitude_noise_do_not_trigger(self):
        noise = np.random.default_rng(42).integers(
            0, 11, size=self.empty.shape, dtype=np.uint8
        )
        for frame, now in [(self.empty, 0.0), (self.empty, 2.0), (noise, 5.0)]:
            with self.subTest(now=now):
                state = self.detector.update(frame, now=now)
                self.assertFalse(state.ready)
                self.assertLess(state.foreground_fraction, 0.04)

    def test_substantial_change_waits_for_a_stable_view(self):
        appeared = self.detector.update(self.item, now=0.0)
        self.assertGreater(appeared.foreground_fraction, 0.04)
        self.assertGreater(appeared.motion_fraction, 0.01)
        self.assertFalse(appeared.ready)
        self.assertFalse(self.detector.update(self.item, now=0.8).ready)

        settled = self.detector.update(self.item, now=1.1)
        self.assertTrue(settled.stable)
        self.assertTrue(settled.ready)

    def test_motion_restarts_the_settling_timer(self):
        self.detector.update(self.item, now=0.0)
        moving = self.detector.update(self.replacement, now=0.7)
        self.assertFalse(moving.stable)
        self.assertFalse(moving.ready)
        self.assertFalse(self.detector.update(self.replacement, now=1.2).ready)
        self.assertTrue(self.detector.update(self.replacement, now=1.8).ready)

    def test_cumulative_slow_motion_does_not_appear_stable(self):
        self.detector.update(self.item, now=0.0)
        for offset, now in [(1, 0.4), (2, 0.8), (3, 1.2)]:
            shifted = np.roll(self.item, offset, axis=1)
            state = self.detector.update(shifted, now=now)
            with self.subTest(offset=offset):
                self.assertFalse(state.ready)
        self.assertTrue(self.detector.update(shifted, now=2.3).ready)

    def test_submitted_item_is_not_repeated_even_after_cooldown(self):
        self.settle_first_item()
        for now in [2.5, 5.0, 100.0]:
            with self.subTest(now=now):
                self.assertFalse(self.detector.update(self.item, now=now).ready)

    def test_stable_empty_platform_rearms_an_identical_item(self):
        self.settle_first_item()
        self.assertFalse(self.detector.update(self.empty, now=5.0).ready)
        self.assertFalse(self.detector.update(self.empty, now=6.2).ready)
        empty_state = self.detector.update(self.empty, now=7.3)
        self.assertTrue(empty_state.stable)
        self.assertFalse(empty_state.ready)

        self.assertFalse(self.detector.update(self.item, now=8.0).ready)
        self.assertTrue(self.detector.update(self.item, now=9.1).ready)

    def test_cooldown_ignores_changes_without_queuing_a_later_capture(self):
        captured = self.settle_first_item()
        self.assertEqual(self.detector.cooldown_seconds, 5)
        for frame, now in [(self.empty, 1.2), (self.replacement, 3.0), (self.empty, 6.0)]:
            with self.subTest(now=now):
                ignored = self.detector.update(frame, now=now)
                self.assertFalse(ignored.ready)
                self.assertTrue(ignored.stable)
                self.assertEqual(ignored.motion_fraction, 0)
                self.assertEqual(ignored.foreground_fraction, captured.foreground_fraction)
        for now in [6.1, 7.2, 100]:
            with self.subTest(after_expiry=now):
                waiting = self.detector.update(self.replacement, now=now)
                self.assertFalse(waiting.ready)
                self.assertTrue(waiting.waiting_for_change)

    def test_post_expiry_motion_can_return_to_the_last_captured_view(self):
        self.settle_first_item()
        self.detector.update(self.replacement, now=2.0)
        self.assertFalse(self.detector.update(self.replacement, now=6.1).ready)
        self.assertFalse(self.detector.update(self.replacement, now=7.2).ready)
        self.assertFalse(self.detector.update(self.item, now=8.0).ready)
        self.assertTrue(self.detector.update(self.item, now=9.1).ready)

    def test_explicit_rearm_cannot_skip_cooldown_or_the_fresh_change_requirement(self):
        self.settle_first_item()
        self.detector.rearm()
        self.assertFalse(self.detector.update(self.item, now=2.3).ready)
        self.assertFalse(self.detector.update(self.item, now=6.1).ready)
        self.detector.rearm()
        self.assertFalse(self.detector.update(self.item, now=8.0).ready)
        self.assertFalse(self.detector.update(self.replacement, now=9.0).ready)
        self.assertTrue(self.detector.update(self.replacement, now=10.1).ready)

    def test_recalibration_preserves_active_cooldown_and_fresh_monitoring_anchor(self):
        self.settle_first_item()
        self.detector.calibrate(self.empty)
        ignored = self.detector.update(self.empty, now=2.3)
        self.assertFalse(ignored.ready)
        self.assertEqual(ignored.motion_fraction, 0)
        self.assertFalse(self.detector.update(self.replacement, now=6.1).ready)
        self.assertFalse(self.detector.update(self.replacement, now=8.0).ready)
        self.assertFalse(self.detector.update(self.item, now=9.0).ready)
        self.assertTrue(self.detector.update(self.item, now=10.1).ready)

    def test_zero_cooldown_keeps_continuous_detection_without_an_expiry_gate(self):
        detector = ChangeDetector(cooldown_seconds=0)
        detector.calibrate(self.empty)
        detector.update(self.item, now=0.0)
        self.assertTrue(detector.update(self.item, now=1.1).ready)
        detector.mark_submitted(self.item, now=1.1)
        self.assertFalse(detector.update(self.replacement, now=1.2).ready)
        self.assertTrue(detector.update(self.replacement, now=2.3).ready)
        detector.mark_submitted(self.replacement, now=2.3)
        detector.update(self.empty, now=2.4)
        self.assertFalse(detector.update(self.empty, now=3.5).ready)
        detector.update(self.item, now=3.6)
        self.assertTrue(detector.update(self.item, now=4.7).ready)

    def test_cumulative_slow_motion_is_detected_after_cooldown_expiry(self):
        self.settle_first_item()
        self.assertFalse(self.detector.update(self.item, now=6.1).ready)
        for offset in range(1, 13):
            shifted = np.roll(self.item, offset, axis=1)
            self.assertFalse(self.detector.update(shifted, now=6.1 + 0.2 * offset).ready)
        self.assertTrue(self.detector.update(shifted, now=9.6).ready)

    def test_tiny_object_does_not_cross_substantial_change_threshold(self):
        tiny = self.empty.copy()
        tiny[75:85, 75:85] = 255
        self.detector.update(tiny, now=0.0)
        state = self.detector.update(tiny, now=2.0)
        self.assertLess(state.foreground_fraction, 0.04)
        self.assertFalse(state.ready)

    def test_scene_matching_rejects_stale_classification(self):
        noise = np.random.default_rng(12).integers(
            0, 8, size=self.item.shape, dtype=np.uint8
        )
        noisy_item = np.maximum(self.item, noise)
        self.assertTrue(self.detector.matches(noisy_item, self.item))
        self.assertFalse(self.detector.matches(self.replacement, self.item))
        self.assertFalse(self.detector.matches(self.empty, self.item))


if __name__ == "__main__":
    unittest.main()
