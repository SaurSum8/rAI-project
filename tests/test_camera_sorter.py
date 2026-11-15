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
            cooldown_seconds=3.0,
        )
        self.detector.calibrate(self.empty)

    def settle_first_item(self):
        self.assertFalse(self.detector.update(self.item, now=0.0).ready)
        self.assertTrue(self.detector.update(self.item, now=1.1).ready)
        self.detector.mark_submitted(self.item, now=1.1)

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
        empty_state = self.detector.update(self.empty, now=6.1)
        self.assertTrue(empty_state.stable)
        self.assertFalse(empty_state.ready)

        self.assertFalse(self.detector.update(self.item, now=7.0).ready)
        self.assertTrue(self.detector.update(self.item, now=8.1).ready)

    def test_changed_item_waits_for_cooldown(self):
        self.settle_first_item()
        self.assertFalse(self.detector.update(self.replacement, now=1.2).ready)
        settled = self.detector.update(self.replacement, now=2.3)
        self.assertTrue(settled.stable)
        self.assertFalse(settled.ready)
        self.assertTrue(self.detector.update(self.replacement, now=4.2).ready)

    def test_explicit_rearm_preserves_cooldown_for_the_same_scene(self):
        self.settle_first_item()
        self.detector.rearm()
        self.assertFalse(self.detector.update(self.item, now=2.3).ready)
        self.assertTrue(self.detector.update(self.item, now=4.2).ready)

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
