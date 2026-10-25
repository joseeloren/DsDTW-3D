# First 3D baseline: 3DAirSig

This extension uses the small, locally supplied 3DAirSig dataset: 15 signers,
600 XYZ trajectories. Every signer has five native reference signatures
(`Genuine/train`), ten genuine probes (`Genuine/test`) and 25 skilled forgeries.
Point `--raw-data` to the directory containing `subject-*`, or its direct parent.

```bash
python -m pip install -e .
python -m dsdtw3d run --raw-data /path/to/3DAirSig-Dataset --output runs/pilot --device cuda --softdtw cpu
```

Use `--device cpu` to run the entire model on CPU. The command prepares a copy,
trains seed 111 for 20 epochs and evaluates with one and four references.
Each run uses a new output directory. Source trajectories are read-only;
existing DeepSignDB/3D pickle files are neither read nor modified.

## Data and protocol

Signer IDs 1–12 are development signers; 13–15 are held out. Development uses
all genuine recordings of those development signers. For held-out signers,
the native reference and probe folders stay separate. The first numerically
ordered k reference files are used; the remaining reference files are not
added to the probe set. All ten native genuine probes and 25 skilled
forgeries are evaluated. Random forgeries use the ten native probes from each
other held-out signer, with the correct count of 20 impostor probes per signer.
No evaluation signer participates in optimization. This is an explicitly
defined pilot protocol, not a claim to reproduce an official published split.

Preparation writes a pickle-free NPZ plus a manifest with source-file SHA-256,
signer split, feature names and lengths. It rejects invalid rows rather than
silently dropping them. Files and prepared data are verified after training.
Never commit the dataset, scores or biometric checkpoints; they are ignored.

## Representation and model

The nine features are XYZ, three velocity components and three acceleration
components. The spatial path is centered and isotropically scaled. Linear
resampling to 128 points uses normalized sequence time, not an assumed sensor
sampling rate. Velocity and acceleration each use a single vector-group RMS
scale; constant axes stay finite. No pressure channel is invented from Z.
There is no frequency-based low-pass filter because the supplied README
does not specify a sampling rate.

The CNN widths 64/128, two-layer GRU width 128, output width 64, dropout 0.1,
closed update gates, soft-DTW gamma 5/band 0.1, triplet objective and SGD
settings follow upstream. Input channels change from 12 to 9. Equal-length
resampling removes the need for packed variable-length batches; the hidden
state is created for the actual batch size. The network supports CPU/CUDA;
soft-DTW defaults to CPU. At least 48 resampled points keep the training
soft-DTW sequences out of the short-sequence bandwidth mismatch identified
in the upstream CPU implementation.

Training uses four tasks, five positives, five skilled and five random
negatives per task, learning rate 0.01, momentum 0.9 and decay 0.9 per epoch.
There are twelve steps per epoch for twelve development signers. These
settings retain upstream values but adapt the dataset and epoch size.

Evaluation uses **exact DTW with L1 cost**, normalized by the sum of lengths,
and averages the distance to k references. This deliberately simple 3D
baseline differs from the upstream FastDTW approximation and four-reference
score normalization. EER is interpolated at FAR/FRR crossing. The reported
threshold is a test-set metric, not a deployment threshold calibrated on
independent validation data. This small pilot does not establish generalization.

The upstream 2D scripts, models and attribution remain available. The shared
soft-DTW module has the compatibility changes validated in Isard: 64-register
CUDA kernels and cloned autograd views/saved recurrence matrices. No 2D
checkpoint is reused for a nine-channel network.

## Separate commands

```bash
python -m dsdtw3d prepare --raw-data /path/to/3DAirSig-Dataset --output data/3dairsig
python -m dsdtw3d train --data data/3dairsig --output runs/seed111 --device cuda
python -m dsdtw3d evaluate --data data/3dairsig --checkpoint runs/seed111/model.pt --output runs/eval111 --references 1 --device cuda
python -m unittest discover -s tests -p 'test_3d.py' -v
```

The inherited `requirements.txt` describes the historical 2D environment.
Install the 3D package with `pip install -e .` instead. The original academic
use terms and citation in ReadMe.md still apply to the inherited code.
