import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from dsdtw3d.features import extract_xyz
from dsdtw3d.data import prepare, load_prepared, verify_sources, read_xyz


class FeatureTests(unittest.TestCase):
    def test_no_mutation_and_z_contributes(self):
        t = np.linspace(0, 1, 30)
        xyz = np.stack([t, t**2, np.sin(t * 4)], axis=1)
        original = xyz.copy()
        features = extract_xyz(xyz)
        np.testing.assert_array_equal(xyz, original)
        self.assertEqual(features.shape, (128, 9))
        self.assertTrue(np.isfinite(features).all())
        flat = xyz.copy(); flat[:, 2] = 0
        self.assertFalse(np.allclose(features, extract_xyz(flat)))

    def test_constant_axes_and_translation_scale(self):
        xyz = np.stack([np.arange(20), np.zeros(20), np.arange(20)**2], axis=1)
        np.testing.assert_allclose(extract_xyz(xyz), extract_xyz(xyz * 3 + [10, 30, -20]), atol=1e-5)
        self.assertTrue(np.isfinite(extract_xyz(np.ones((3, 3)))).all())

    def test_reject_invalid(self):
        for points in (np.ones((10, 2)), np.ones((1, 3)), np.full((10, 3), np.nan)):
            with self.assertRaises(ValueError): extract_xyz(points)


class PreparationTests(unittest.TestCase):
    def test_disjoint_users_and_native_roles_and_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "raw"; output = Path(temp) / "prepared"
            for user in range(1, 9):
                for folder in ("Genuine/train", "Genuine/test", "Forged"):
                    path = root / f"subject-{user}" / folder / "1.txt"
                    path.parent.mkdir(parents=True); path.write_text("0,0,1\n1,2,3\n2,3,5\n")
            manifest = prepare(root, output)
            arrays, loaded = load_prepared(output)
            self.assertFalse(set(manifest["dev_users"]) & set(manifest["eval_users"]))
            self.assertEqual(set(arrays["roles"]), {"reference", "probe", "forgery"})
            self.assertEqual(arrays["features"].shape, (24, 128, 9))
            self.assertTrue(verify_sources(loaded))
            (root / "subject-1/Genuine/train/1.txt").write_text("changed")
            self.assertFalse(verify_sources(loaded))

    def test_bad_row_is_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "sample.txt"; p.write_text("0,0,0\ninvalid\n1,1,1")
            with self.assertRaises(ValueError): read_xyz(p)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("numba"), "training runtime required")
class RuntimeTests(unittest.TestCase):
    def test_training_checkpoint_reloads_for_evaluation(self):
        from dsdtw3d.engine import train, evaluate
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "raw"
            for user in range(1, 7):
                for folder, count in [("Genuine/train", 3), ("Genuine/test", 3), ("Forged", 5)]:
                    for sample in range(1, count + 1):
                        p = root / f"subject-{user}" / folder / f"{sample}.txt"
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_text(f"0,0,0\n1,{user},{sample}\n2,{user + 1},{sample + 1}\n")
            prepared = Path(temp) / "prepared"; trained = Path(temp) / "train"
            prepare(root, prepared, samples=48)
            result = train(prepared, trained, epochs=1, steps_per_epoch=1, device="cpu")
            report = evaluate(prepared, trained / "model.pt", Path(temp) / "evaluation", device="cpu")
            self.assertEqual(result["steps"], 1)
            self.assertTrue(np.isfinite(report["skilled"]["eer_percent"]))

    def test_cpu_forward_backward_and_dynamic_batch(self):
        import torch
        from dsdtw3d.model import DsDTW3D
        model = DsDTW3D("cpu").train()
        output = model(torch.randn(4, 48, 9))
        losses = model.triplet_loss(output, tasks=1, genuine=1, forged=2)
        (losses[1] + .01 * losses[2]).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertEqual(model.eval()(torch.randn(2, 48, 9)).shape, (2, 24, 64))

    def test_eer_and_dtw(self):
        from dsdtw3d.metrics import eer, dtw_l1
        self.assertEqual(eer([0, 1], [2, 3])["eer_percent"], 0)
        self.assertEqual(eer([1, 1], [1, 1])["eer_percent"], 50)
        self.assertEqual(eer([2, 3], [0, 1])["eer_percent"], 100)
        a = np.array([[0., 0.], [1., 1.]])
        self.assertEqual(dtw_l1(a, a), 0)

    def test_sampler_never_uses_evaluation_signers(self):
        from dsdtw3d.engine import sample_batch
        users = np.repeat(np.arange(1, 7), 40)
        roles = np.tile(["reference"] * 5 + ["probe"] * 10 + ["forgery"] * 25, 6)
        arrays = dict(users=users, roles=roles, splits=np.where(users <= 4, "dev", "eval"))
        indices = sample_batch(arrays, np.random.RandomState(111))
        self.assertEqual(len(indices), 64)
        self.assertTrue(np.all(arrays["splits"][indices] == "dev"))
        for group in indices.reshape(4, 16):
            anchor = users[group[0]]
            self.assertTrue(np.all(users[group[:6]] == anchor))
            self.assertTrue(np.all(roles[group[6:11]] == "forgery"))
            self.assertTrue(np.all(users[group[11:]] != anchor))


if __name__ == "__main__": unittest.main()
