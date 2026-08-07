from __future__ import annotations

import unittest

from control.validation_window import ConsecutiveValidityWindow


class ConsecutiveValidityWindowTests(unittest.TestCase):
    def test_requires_consecutive_valid_samples_and_resets_on_miss(self):
        window = ConsecutiveValidityWindow(required_frames=3)

        self.assertFalse(window.observe(True))
        self.assertEqual(window.valid_frames, 1)
        self.assertFalse(window.observe(True))
        self.assertEqual(window.valid_frames, 2)

        self.assertFalse(window.observe(False))
        self.assertEqual(window.valid_frames, 0)

        self.assertFalse(window.observe(True))
        self.assertFalse(window.observe(True))
        self.assertTrue(window.observe(True))
        self.assertEqual(window.valid_frames, 3)

    def test_rejects_nonpositive_required_frame_count(self):
        with self.assertRaisesRegex(ValueError, "required_frames"):
            ConsecutiveValidityWindow(required_frames=0)


if __name__ == "__main__":
    unittest.main()
