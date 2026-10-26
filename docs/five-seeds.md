# Five-seed comparison: DeepAirSig and 3DAirSig

```bash
python -m dsdtw3d.benchmark --deepairsig /path/to/DeepAirSig_Dataset --3dairsig /path/to/3DAirSig-Dataset --output runs/five-seeds --device cuda
```

The output directory must be new. The command prepares each dataset once,
then trains **five fresh independent models per dataset**, seeds 111, 222,
333, 444 and 555. Every model completes 20 epochs and is evaluated with 1 and
4 references against both skilled and random forgeries. Existing pilot models
are preserved and are not reused. The runner stops and records a failed status
if training, evaluation or integrity checks fail.

The features, splits, model, loss, optimizer, reference selection and evaluation
are the same as the individual [DeepAirSig](deepairsig.md) and
[3DAirSig](3dairsig.md) baselines. The network uses CUDA and soft-DTW uses CPU.
Training size determines epoch length: DeepAirSig has 32 development users
(640 steps/model); 3DAirSig has 12 (240 steps/model). Epoch count and all other
training settings are shared. Signer partitions and reference files are fixed
across seeds; this measures variation due to training, not variation across
different signer splits or reference selections.

`summary.json` records per-seed EER, thresholds, training records, checksums,
runtime versions, source-code hashes and aggregate statistics. `status.json`
records progress. Checkpoints, feature arrays, scores and epoch losses are kept
in separate directories for every dataset and seed. All are private by default.
The source trees and existing pickles are never rewritten. Source/prepared-data
hashes are checked, and duplicate raw hashes across signer partitions are rejected.

For each dataset, reference count and attack type, report the arithmetic mean
of the five **separate per-model EERs**, plus sample standard deviation (`ddof=1`)
in percentage points. The minimum, maximum and population standard deviation
(`ddof=0`) are also recorded. Scores from different models are not pooled and
no seed is selected for the aggregate. A low EER can make relative variation
misleading; inspect absolute percentage-point ranges as well as deviations.

Differences between datasets also reflect their different signers, acquisition
conditions and evaluation sizes. 3DAirSig has only three held-out signers (30
genuine/75 skilled/60 random trials), versus eight in DeepAirSig (80/200/560).
These experiments do not establish statistical equivalence or performance on
an official benchmark protocol. In particular, stability across seeds on one
fixed split is not evidence of stability across populations.

The completed ten-model experiment, all per-seed results and the comparison
are recorded in [validation](../validation/README.md#five-fresh-seeds-on-both-datasets)
and [five-seeds.json](../validation/five-seeds.json).
