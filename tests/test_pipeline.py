from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pmfa.attacks import multi_views, partial_observation
from pmfa.experiment import confidence_sync_indices, reliable_dcss_logits
from pmfa.models import PMFA
from pmfa.pimog import decoder_features, embed, load_pimog, original_logits, screen_shoot


class PMFAPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cpu")
        cls.model = load_pimog(PROJECT_ROOT / "third_party" / "PIMoG", cls.device)

    def test_official_model_forward_is_cpu_compatible(self) -> None:
        image = torch.zeros(1, 3, 128, 128)
        message = torch.zeros(1, 30)
        encoded = embed(self.model, image, message)
        torch.manual_seed(5)
        noised = screen_shoot(self.model, encoded)
        features = decoder_features(self.model.Decoder, noised)
        logits = original_logits(self.model.Decoder, features)
        self.assertEqual(encoded.shape, (1, 3, 128, 128))
        self.assertEqual(features.shape, (1, 256))
        self.assertEqual(logits.shape, (1, 30))

    def test_alignment_candidates_include_oracle_under_prior_bound(self) -> None:
        captured = torch.zeros(3, 128, 128)
        observation = partial_observation(captured, retain_ratio=0.5, seed=91)
        views, error, oracle_index = multi_views(observation, seed=91)
        self.assertEqual(views.shape, (9, 3, 128, 128))
        self.assertLessEqual(abs(error[0]), 2)
        self.assertLessEqual(abs(error[1]), 2)
        self.assertGreaterEqual(oracle_index, 0)
        self.assertLess(oracle_index, 9)

    def test_pmfa_accepts_nine_feature_views(self) -> None:
        adapter = PMFA(self.model.Decoder.extractor.linear)
        logits, weights = adapter(torch.zeros(2, 9, 256))
        self.assertEqual(logits.shape, (2, 30))
        self.assertEqual(weights.shape, (2, 9))
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))
        self.assertLess(adapter.fusion_gate, 0.1)

    def test_confidence_sync_prefers_binary_like_candidate(self) -> None:
        logits = torch.full((1, 3, 30), 0.5)
        logits[:, 1] = 0.01
        logits[:, 2] = 0.25
        self.assertEqual(int(confidence_sync_indices(logits)[0]), 1)

    def test_reliable_dcss_falls_back_when_confidence_is_low(self) -> None:
        logits = torch.full((1, 3, 30), 0.5)
        logits[:, 0] = 0.2
        logits[:, 1] = 0.3
        predicted, used = reliable_dcss_logits(logits, threshold=0.01, fallback_mode="average")
        torch.testing.assert_close(predicted, logits.mean(dim=1))
        self.assertFalse(bool(used[0]))


if __name__ == "__main__":
    unittest.main()
