import unittest
import sys
from pathlib import Path

import numpy as np

import cv2
from PIL import Image

TOOL_DIR = Path(__file__).resolve().parents[1] / "tool"
sys.path.insert(0, str(TOOL_DIR))

from c3stisr_one_image_pipeline import (
    c3_mask_to_text_mask,
    color_match_crop,
    gray3_image,
    preserve_canvas_chroma,
)


class PasteMaskTests(unittest.TestCase):
    def test_pre_input_repeats_gray_into_three_channels(self):
        image = Image.fromarray(np.array([[[255, 0, 0]]], dtype=np.uint8), mode="RGB")

        gray3 = np.asarray(gray3_image(image))

        np.testing.assert_array_equal(gray3[..., 0], gray3[..., 1])
        np.testing.assert_array_equal(gray3[..., 1], gray3[..., 2])

    def test_dark_text_on_light_background_keeps_c3_polarity(self):
        c3_mask = np.zeros((5, 5), dtype=np.uint8)
        c3_mask[2, 2] = 255

        text_mask, inverted = c3_mask_to_text_mask(c3_mask)

        self.assertFalse(inverted)
        self.assertEqual(int(text_mask[2, 2]), 255)
        self.assertEqual(int(text_mask[0, 0]), 0)

    def test_light_text_on_dark_background_inverts_c3_polarity(self):
        c3_mask = np.full((5, 5), 255, dtype=np.uint8)
        c3_mask[2, 2] = 0

        text_mask, inverted = c3_mask_to_text_mask(c3_mask)

        self.assertTrue(inverted)
        self.assertEqual(int(text_mask[2, 2]), 255)
        self.assertEqual(int(text_mask[0, 0]), 0)

    def test_color_match_uses_only_non_text_pixels(self):
        source = np.full((5, 5, 3), 80, dtype=np.uint8)
        target = np.full((5, 5, 3), 140, dtype=np.uint8)
        source[2, 2] = 220
        target[2, 2] = 10
        text_mask = np.zeros((5, 5), dtype=np.uint8)
        text_mask[2, 2] = 255

        matched, debug = color_match_crop(source, target, text_mask)

        self.assertTrue(debug["applied"])
        np.testing.assert_allclose(matched[0, 0], [140, 140, 140], atol=1)

    def test_luma_mode_preserves_destination_chroma(self):
        source = np.full((5, 5, 3), [220, 80, 80], dtype=np.uint8)
        target = np.full((5, 5, 3), [70, 110, 150], dtype=np.uint8)
        text_mask = np.zeros((5, 5), dtype=np.uint8)

        matched, debug = preserve_canvas_chroma(source, target, text_mask)

        matched_ycc = cv2.cvtColor(matched, cv2.COLOR_RGB2YCrCb)
        target_ycc = cv2.cvtColor(target, cv2.COLOR_RGB2YCrCb)
        self.assertTrue(debug["applied"])
        np.testing.assert_allclose(matched_ycc[..., 1:], target_ycc[..., 1:], atol=1)


if __name__ == "__main__":
    unittest.main()
