import unittest
import sys
from pathlib import Path

import torch

TOOL_DIR = Path(__file__).resolve().parents[1] / "tool"
sys.path.insert(0, str(TOOL_DIR))

from c3stisr_infer import infer_with_priors, tensor_to_image


class DummyRecognizer:
    def __call__(self, tensor):
        assert tensor.shape == (2, 1, 32, 100)
        return torch.zeros((26, 2, 37))


class DummyLanguageModel:
    def __call__(self, probabilities, lengths):
        assert probabilities.shape == (2, 26, 37)
        assert lengths.tolist() == [26, 26]
        return {"logits": torch.zeros_like(probabilities)}


class DummyC3:
    def __init__(self):
        self.lm = DummyLanguageModel()
        self.clue_args = None

    def __call__(self, tensor, *clues):
        self.clue_args = clues
        return tensor


class PriorTests(unittest.TestCase):
    def test_clamp_keeps_zero_output_black(self):
        tensor = torch.zeros((3, 2, 2))

        clamp_image = tensor_to_image(tensor, "clamp")
        legacy_tanh_image = tensor_to_image(tensor, "tanh")

        self.assertEqual(clamp_image.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(legacy_tanh_image.getpixel((0, 0)), (128, 128, 128))

    def test_rec_ling_clue_shapes_match_released_c3_path(self):
        model = DummyC3()
        lr_tensor = torch.zeros((2, 4, 16, 64))

        output, clues = infer_with_priors(model, lr_tensor, "rec-ling", DummyRecognizer())

        self.assertIs(output, lr_tensor)
        self.assertEqual(tuple(clues["rec"].shape), (2, 26, 37))
        self.assertEqual(tuple(clues["ling"].shape), (2, 26, 37))
        self.assertEqual(tuple(model.clue_args[0].shape), (2, 37, 1, 26))
        self.assertEqual(tuple(model.clue_args[1].shape), (2, 37, 1, 26))
        self.assertIsNone(model.clue_args[2])


if __name__ == "__main__":
    unittest.main()
