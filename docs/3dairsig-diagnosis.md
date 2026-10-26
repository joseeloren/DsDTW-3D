# Why one-reference 3DAirSig varies across seeds

The strongest observed mechanism is **reference choice plus a shared global
threshold**, amplified by the small, particular set of evaluation users.
No preprocessing corruption or nondeterministic feature generation was found.
Increasing the training budget did not remove the seed variation.

These are exploratory diagnostics on the existing fixed test split. Original
pilot/five-seed results and the production feature extractor, model and engine
remain unchanged. Alternative references are not selected as a new best result.

## Separation within users, overlap between users

For the original first reference, every one of users 13, 14 and 15 has **0% local
EER in all five models**, while pooled global EER is 8.80 ± 3.87%. Local EER
fits a separate threshold to each user's test scores; this is diagnostic, not
a deployment threshold calibrated on independent data.

The global-threshold errors consistently concentrate in **false accepts for
user 13** and **false rejects for user 14**. User 15 contributes neither at the
interpolated global EER threshold. For seed 111, user 13's closest skilled
forgery scores 8.9566, while user 14's farthest genuine scores 10.7613. No single
threshold can reject the former and accept the latter. Yet each user's genuine
and skilled ranges are separated internally. Seeds change this cross-user
score alignment and therefore the global EER.

Holding the trained models fixed and changing only the reference index gives:

| Native reference file | Global skilled EER mean ± sample SD (%) |
| ---: | ---: |
| 1.txt (original baseline) | 8.8000 ± 3.8701 |
| 2.txt | 0.0000 ± 0.0000 |
| 3.txt | 0.9333 ± 1.4606 |
| 4.txt | 4.0000 ± 2.0548 |
| 5.txt | 0.0000 ± 0.0000 |

Only one reference is used per verification in each row. Averaging the five
separate single-reference EERs within each seed yields **2.7467 ± 0.9409%**
across seeds. This is not the same as using five simultaneous references.
Exhausting all 5³ independent per-user reference combinations yields an average
of 2.7243 ± 0.8331%; 31/125 combinations have zero global EER in all five seeds.
The original first-reference case is tied for the highest mean and highest
seed SD among these combinations (five ties, differing only in user 15's ref).

The preserved upstream 2D verifier already loops over four single-reference
positions (`verify_stylus_all.py`, lines 105–141), reports global and local
threshold EERs, and normalizes multi-reference scores using reference distances.
The documented initial 3D baseline instead fixes the first reference and uses
raw mean distances. For one reference, a common positive scale factor cannot
repair cross-user score alignment; a robust evaluation should report reference
sensitivity or average predetermined reference trials. Four-reference results
are a different verification condition, not a calibration fix for one reference.

## Effect of the three evaluation users

Removing one user changes the metric markedly without retraining:

| Users retained | Global skilled EER mean ± sample SD (%) |
| --- | ---: |
| 13, 14, 15 | 8.8000 ± 3.8701 |
| 13, 14 | 13.2000 ± 5.8052 |
| 13, 15 | 0.4000 ± 0.8944 |
| 14, 15 | 0.0000 ± 0.0000 |

This isolates the conflict between users 13 and 14; it does not justify dropping
either. There are only 30 genuine probes and 75 skilled forgeries. A genuine
error changes FRR by 3.33 percentage points; one skilled false accept changes
FAR by 1.33 points. EER interpolation does not add independent observations.

As a size/composition control, the fixed DeepAirSig models were evaluated on
all 56 triplets of their eight held-out users, using the same first reference.
The seed SD rises from 1.04 points with all eight users to a median **1.83**
points over triplets, range **0.30–3.87**. One triplet reaches 3DAirSig's 3.87.
Thus small/composition-sensitive evaluation can amplify dispersion, but user
count alone does not explain the specific reference/threshold conflict.
Triplets overlap; this distribution is descriptive, not independent tests or
a statistical p-value. No test user enters training in these subset analyses.

## Controlled training and preprocessing tests

Fifteen new models were trained, five seeds per intervention, 20 epochs each.
The original 3DAirSig signer split, model, optimizer and evaluation stayed
fixed. Derivative interventions use new derived NPZs, preserving XYZ exactly;
the original prepared data and raw files are unchanged. Their batch schedules
match the original corresponding seeds exactly.

| Condition | Steps/model | Skilled EER 1 ref: mean ± SD (%) | Skilled EER 4 refs: mean ± SD (%) |
| --- | ---: | ---: | ---: |
| Original nine channels | 240 | 8.8000 ± 3.8701 | 0.5333 ± 1.1926 |
| 32 steps/epoch, same 20 epochs | 640 | 6.4000 ± 3.8181 | 0.2667 ± 0.5963 |
| Smooth only the path used for derivatives | 240 | 10.8000 ± 5.4650 | 1.2000 ± 1.6600 |
| XYZ only, six derivative channels zeroed | 240 | 8.0000 ± 2.9814 | 3.3333 ± 2.3570 |

The longer budget lowers mean EER but leaves seed SD almost unchanged. It
changes exposure per epoch and the optimizer-step timing of epoch-based LR
decay, so it is a budget treatment, not a perfectly isolated step-count effect.
Smoothing uses a five-point binomial kernel [1,4,6,4,1]/16 with reflected edges,
followed by the original gradient and per-group RMS scaling; XYZ channels are
retained bit for bit. It worsens this condition. Zeroing derivatives slightly
reduces one-reference dispersion but substantially worsens four-reference EER.
These trials do not support attributing the problem to a simple derivative
noise bug or adopting either preprocessing change.

All original 3DAirSig hard-triplet losses decrease from about 0.54–0.56 in
epoch 1 to 0.002–0.019 in epoch 20, with finite losses. The low training loss
does not guarantee cross-user calibration, and no intermediate test checkpoints
were saved, so these logs alone cannot locate an optimal stopping epoch.

## Data audit, verification and limits

All 600 3DAirSig and 1,800 DeepAirSig feature tensors were regenerated from
the raw text and match the stored baseline features **bit for bit**. Neither
dataset has constant trajectories or repeated consecutive XYZ points in this
release. Native point counts are 32–207 (3DAirSig) and 19–207 (DeepAirSig).
There is no conspicuous extra acceleration-energy concentration in 3DAirSig:
the top three resampled points account for 13.33% on average, versus 14.31%
in DeepAirSig. This descriptive measure does not prove feature optimality.

All 300 intervention epoch records, 15 checkpoint hashes and 30 evaluations
were verified; EERs were recomputed from saved scores. Cached baseline distances
reproduce all ten original one-reference skilled EERs and thresholds exactly.
All 14 tests pass, including a fixture for local/global threshold divergence
and checks that derivative interventions preserve XYZ and the source NPZ.
All ten original checkpoint hashes are unchanged. The longer runs' first 240
sampled batches match the corresponding baseline runs.
Source/prepared hashes are unchanged, and all 2,402 original text files in I:
were rechecked. Existing pickles were neither read nor rewritten.

The evidence supports reference/score calibration as the dominant mechanism
in this split, with small evaluation composition amplifying it. It does not
establish a percentage causal contribution or population-level performance.
A subsequent confirmatory protocol should freeze reference averaging and
evaluate several signer-disjoint splits, while reporting both global EER and
local EER as distinct metrics. Any deployable score calibration must be learned
without tuning thresholds or reference choices on these test probes.

## Reproduce

```bash
python -m dsdtw3d.diagnose --baseline /path/to/completed/five-seeds --output runs/diagnosis --device cuda
```

The output must be new. Detailed per-trial distances, checkpoints and derived
features remain private. The aggregate report is
[3dairsig-diagnosis.json](../validation/3dairsig-diagnosis.json).

![Diagnostic comparison](../validation/figures/3dairsig-diagnosis.png)
