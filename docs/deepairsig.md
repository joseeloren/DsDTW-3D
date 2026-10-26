# DeepAirSig

DeepAirSig uses the same XYZ feature extractor, CNN/GRU, sampler and loss as
the 3DAirSig adaptation. The loader understands its different directory layout:

```text
DeepAirSig_Dataset/
  Genuine/subject_1/train/1.txt ... 10.txt
  Genuine/subject_1/test/1.txt ... 10.txt
  Forged/subject_1/1.txt ... 25.txt
  ... subject_40
```

Each text file contains three comma- or whitespace-separated XYZ columns.
The supplied release has 40 signers and 1,800 trajectories: 800 genuine and
1,000 skilled forgeries. The loader also accepts the direct parent of this root.
It rejects missing samples, malformed rows and unmatched genuine/forgery IDs.

## Complete training and evaluation

```bash
python -m pip install -e .
python -m dsdtw3d run --dataset deepairsig --raw-data /path/to/DeepAirSig_Dataset --output runs/deepairsig-111 --epochs 20 --seed 111 --device cuda --softdtw cpu
```

The output directory must be new. `--dataset auto` (the default) recognizes
both supported layouts; specify the dataset to disambiguate mixed directories.
For a CPU-only machine use `--device cpu`.

The default split holds out the last 20% of numerically sorted signer IDs:
1–32 are development users and 33–40 are evaluation users. Development training
pools native genuine train and test samples, plus skilled forgeries. No held-out
signer enters training. Evaluation references come only from native `train`,
and genuine probes only from native `test`. There are 80 genuine, 200 skilled
and 560 random-forgery trials for each reference-count condition (1 and 4).
Random trials use the other seven held-out users' genuine test signatures.
References are the first k files in numeric order.

Training uses seed 111, 20 epochs, 32 steps per epoch, four tasks per batch,
five genuine positives and ten negatives per task (five skilled, five random).
SGD, momentum 0.9, learning rate 0.01 with epoch decay 0.9, and the original
hard-triplet/variance objective are shared with the first 3D adaptation.
The network runs on GPU; differentiable soft-DTW runs on CPU.

Each trajectory produces 128 points with nine channels: normalized XYZ,
velocity and acceleration. See [the feature/protocol notes](3dairsig.md).
Evaluation uses exact L1 DTW divided by the sum of sequence lengths and the
mean distance across references. EER uses a global interpolated FAR/FRR crossing.
This is an explicitly defined functional baseline, **not** a claim to reproduce
an official DeepAirSig benchmark or the original 2D DeepSignDB results.

## Original data preservation

The input tree is read only. Features are written to a new `data/dataset.npz`
inside the run directory; existing researcher pickles are neither read nor
rewritten. `manifest.json` records SHA-256 for every source trajectory and the
prepared NPZ. Training and both evaluations verify those checksums. Results,
checkpoint, epoch losses and scores remain separate from the source data.
Raw trajectories, features, checkpoints and scores are excluded from Git.

Public aggregate results are recorded in [validation](../validation/README.md).
