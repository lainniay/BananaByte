import unittest

from text_region_ensemble import candidate_groups, decide


class EnsembleTests(unittest.TestCase):
    def test_overlapping_detectors_are_accepted(self):
        craft = {"source": "craft", "bbox_xyxy": [0, 0, 10, 5], "box": [[0, 0], [10, 0], [10, 5], [0, 5]], "confidence": 0.1}
        paddle = {"source": "paddle", "bbox_xyxy": [1, 0, 11, 5], "box": [[1, 0], [11, 0], [11, 5], [1, 5]], "confidence": 0.4}

        groups = candidate_groups([craft, paddle])
        accepted, reason = decide(groups[0], (100, 100, 3))

        self.assertEqual(len(groups), 1)
        self.assertTrue(accepted)
        self.assertEqual(reason, "craft_paddle_agreement")


if __name__ == "__main__":
    unittest.main()
