# 3DAirSig pilot validation

Validated on 25 October 2026 in Isard with Python 3.9, PyTorch 2.7.1+cu128,
NumPy 1.26.4 and Numba 0.60.0. The network used CUDA and soft-DTW used CPU.

All nine tests pass, including source immutability, signer-disjoint sampling,
XYZ features, CPU gradients, dynamic batch sizes, DTW/EER fixtures and a
train-save-load-evaluate regression. The wheel builds, installs into a separate
target directory, and reproduces identical evaluation scores outside the checkout.

The real-data run used 600 trajectories, with 12 development signers and three
evaluation signers. It completed 20 epochs, 12 steps per epoch, seed 111, and
evaluated 30 genuine, 75 skilled-forgery and 60 random-forgery probes per
reference-count condition. No raw-file SHA-256 occurs in both signer partitions.
All raw source hashes and the prepared-data hash were verified unchanged.

| References | Skilled EER (%) | Random EER (%) |
| ---: | ---: | ---: |
| 1 | 13.3333 | 0.0000 |
| 4 | 0.0000 | 0.0000 |

These are functional validation results on a small, explicitly defined split.
They do not establish accuracy on a published protocol or on unseen populations.
The aggregate machine-readable summary is [3dairsig-pilot.json](3dairsig-pilot.json).
Raw trajectories, prepared features, score arrays and checkpoints are excluded
from the public repository.
