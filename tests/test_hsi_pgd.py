"""CPU checks for full-cube PGD. No CLIP weights or Dassl install required."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from attack.hsi_pgd import attack_cube_pgd, patchify_cube
from trainers.hsi_rgb_adapter import HSIRGBAdapter


class TinyPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = HSIRGBAdapter(4)
        self.head = nn.Linear(3, 2)

    def forward(self, patches, return_features=False):
        features = self.adapter(patches).sigmoid().mean((2, 3))
        features = nn.functional.normalize(features, dim=1)
        text = nn.functional.normalize(self.head.weight, dim=1)
        logits = features @ text.t()
        return (logits, features, text) if return_features else logits


class CubePGDTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        torch.set_num_threads(1)
        self.cube = torch.rand(4, 6, 7) * 0.6 - 0.3
        self.locations = torch.tensor([[0, 0], [2, 2], [2, 3], [5, 6]])
        self.labels = torch.tensor([0, 1, 0, 1])
        self.model = TinyPipeline()

    def attack(self, **kwargs):
        options = dict(patch_size=3, epsilon=0.05 * 255, alpha=0.01 * 255, steps=3,
                       batch_size=2, train_trades=True, lower=-1., upper=1.)
        options.update(kwargs)
        return attack_cube_pgd(self.model, self.cube, self.locations, **options)

    def test_padding_and_shared_voxel_gradients(self):
        cube = self.cube.clone().requires_grad_(True)
        patches = patchify_cube(cube, self.locations, 3)
        padded = nn.functional.pad(cube, (1, 1, 1, 1))
        expected = torch.stack([padded[:, r:r+3, c:c+3] for r, c in self.locations])
        torch.testing.assert_close(patches, expected)
        # The same physical voxel appears in both neighboring patches.
        torch.testing.assert_close(patches[1, :, 1, 2], patches[2, :, 1, 1])
        (patches[1, :, 1, 2].sum() + patches[2, :, 1, 1].sum()).backward()
        torch.testing.assert_close(cube.grad[:, 2, 3], torch.full((4,), 2.))

    def test_linf_bounds_no_parameter_grad_or_bn_updates(self):
        mean = self.model.adapter.proj[1].running_mean.clone()
        self.model.head.weight.requires_grad_(False)
        flags = [p.requires_grad for p in self.model.parameters()]
        delta = self.attack()
        self.assertEqual(delta.shape, self.cube.shape)
        self.assertFalse(delta.requires_grad)
        self.assertLessEqual(delta.abs().max().item(), 0.050001)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertEqual(flags, [p.requires_grad for p in self.model.parameters()])
        self.assertTrue(self.model.training)
        torch.testing.assert_close(mean, self.model.adapter.proj[1].running_mean)

    def test_microbatch_gradient_accumulation_matches_full_split(self):
        torch.manual_seed(123)
        small = self.attack(batch_size=1)
        torch.manual_seed(123)
        full = self.attack(batch_size=4)
        torch.testing.assert_close(small, full)

    def test_l2_is_one_global_cube_budget(self):
        delta = self.attack(norm='l_2')
        self.assertLessEqual(delta.norm().item(), 0.050001)

    def test_cube_bounds_and_zero_budget(self):
        delta = self.attack(lower=-0.31, upper=0.31)
        self.assertLessEqual((self.cube + delta).max().item(), 0.310001)
        self.assertGreaterEqual((self.cube + delta).min().item(), -0.310001)
        self.assertEqual(self.attack(epsilon=0).abs().sum().item(), 0)
        self.assertEqual(self.attack(steps=0).abs().sum().item(), 0)

    def test_trades_kl_increases_and_outer_gradient_reaches_adapter(self):
        self.model.eval()
        with torch.no_grad():
            reference = self.model(patchify_cube(self.cube, self.locations, 3)).softmax(dim=1)
            torch.manual_seed(123)
            initial_delta = torch.empty_like(self.cube).uniform_(-0.05, 0.05)
            initial_logits = self.model(patchify_cube(self.cube + initial_delta, self.locations, 3))
            before = nn.functional.kl_div(initial_logits.log_softmax(dim=1), reference, reduction='batchmean')
        torch.manual_seed(123)
        delta = self.attack()
        logits = self.model(patchify_cube(self.cube + delta, self.locations, 3))
        after = nn.functional.kl_div(logits.log_softmax(dim=1), reference, reduction='batchmean')
        self.assertGreater(after.item(), before.item())
        # Defender classification still sends gradients to the trainable adapter.
        nn.functional.cross_entropy(logits, self.labels).backward()
        self.assertGreater(self.model.adapter.proj[0].weight.grad.abs().sum().item(), 0)

    def test_default_non_trades_uses_ce_and_requires_labels(self):
        with self.assertRaisesRegex(ValueError, 'labels matching locations'):
            attack_cube_pgd(self.model, self.cube, self.locations, patch_size=3, lower=-1., upper=1.)
        self.model.eval()
        with torch.no_grad():
            before = nn.functional.cross_entropy(
                self.model(patchify_cube(self.cube, self.locations, 3)), self.labels)
        delta = self.attack(train_trades=False, labels=self.labels)
        with torch.no_grad():
            after = nn.functional.cross_entropy(
                self.model(patchify_cube(self.cube + delta, self.locations, 3)), self.labels)
        self.assertGreater(after.item(), before.item())

    def test_config_value_one_is_divided_by_255_once(self):
        delta = self.attack(epsilon=1, alpha=1, steps=1,
                            train_trades=False, labels=self.labels)
        self.assertAlmostEqual(delta.abs().max().item(), 1.0 / 255.0, places=6)

    def test_trades_attack_inside_no_grad(self):
        with torch.no_grad():
            delta = self.attack()
        self.assertTrue(torch.isfinite(delta).all())
        self.assertLessEqual(delta.abs().max().item(), 0.050001)


class UnitIntervalCubePGDTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        torch.set_num_threads(1)
        self.raw = torch.rand(4, 6, 7) * 10 + 5
        from hsi_preprocessing import scale_hsi_to_unit_interval
        self.unit = torch.from_numpy(scale_hsi_to_unit_interval(self.raw.numpy()))
        self.locations = torch.tensor([[0, 0], [2, 2], [2, 3], [5, 6]])
        self.labels = torch.tensor([0, 1, 0, 1])
        self.model = TinyPipeline().eval()

    def test_attack_uses_scaled_cube_and_respects_unit_interval_budget(self):
        seen = []
        handle = self.model.register_forward_pre_hook(
            lambda model, inputs: seen.append(inputs[0].detach().clone()))
        try:
            for trades in (False, True):
                seen.clear()
                delta = attack_cube_pgd(
                    self.model, self.unit, self.locations, patch_size=3,
                    epsilon=1, alpha=1, steps=2, batch_size=2,
                    train_trades=trades, labels=self.labels)
                self.assertTrue(seen)
                self.assertTrue(all(patches.min() >= 0 and patches.max() <= 1 for patches in seen))
                self.assertGreater(delta.abs().max().item(), 0)
                self.assertLessEqual(delta.abs().max().item(), 1/255 + 1e-6)
                self.assertGreaterEqual((self.unit + delta).min().item(), 0.)
                self.assertLessEqual((self.unit + delta).max().item(), 1.)
                self.assertFalse(delta.requires_grad)
        finally:
            handle.remove()

    def test_shared_patch_preprocessing_scales_before_patchify(self):
        import tempfile
        import numpy as np
        import scipy.io as sio
        from hsi_preprocessing import preprocess_hsi_dataset, scale_hsi_to_unit_interval
        with tempfile.TemporaryDirectory() as directory:
            cube_path, gt_path = Path(directory) / 'cube.mat', Path(directory) / 'gt.mat'
            raw_hwc = self.raw.permute(1, 2, 0).numpy()
            sio.savemat(cube_path, {'ori_data': raw_hwc})
            sio.savemat(gt_path, {'map': np.ones(raw_hwc.shape[:2], dtype=np.int64)})
            patches, labels, locations = preprocess_hsi_dataset(
                str(cube_path), str(gt_path), patch_size=3)
            index = np.flatnonzero((locations == [2, 3]).all(axis=1))[0]
            expected = scale_hsi_to_unit_interval(raw_hwc)
            np.testing.assert_array_equal(patches[index], expected[1:4, 2:5])
            self.assertGreaterEqual(patches.min(), 0)
            self.assertLessEqual(patches.max(), 1)

    def test_whole_cube_scaling_handles_negative_and_constant_values(self):
        import numpy as np
        from hsi_preprocessing import scale_hsi_to_unit_interval
        raw = np.array([-11., 0., 9207.], dtype=np.float32).reshape(1, 1, 3)
        scaled = scale_hsi_to_unit_interval(raw)
        self.assertEqual(float(scaled.min()), 0.)
        self.assertEqual(float(scaled.max()), 1.)
        self.assertAlmostEqual(float(scaled[0, 0, 1]), 11/9218, places=7)
        np.testing.assert_array_equal(scale_hsi_to_unit_interval(np.ones((2, 3, 4))), 0.)

    def test_scaled_cube_gradient_accumulation_matches_one_full_batch(self):
        options = dict(patch_size=3, epsilon=1, alpha=1, steps=2,
                       labels=self.labels)
        torch.manual_seed(123)
        small = attack_cube_pgd(self.model, self.unit, self.locations, batch_size=1, **options)
        torch.manual_seed(123)
        full = attack_cube_pgd(self.model, self.unit, self.locations, batch_size=4, **options)
        torch.testing.assert_close(small, full, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
