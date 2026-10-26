# DsDTW-3D: XYZ in-air signature baseline

Supported datasets: **3DAirSig** (15 signers, 600 trajectories) and
**DeepAirSig** (40 signers, 1,800 trajectories).

```bash
python -m pip install -e .
python -m dsdtw3d run --raw-data /path/to/3DAirSig-Dataset --output runs/pilot --device cuda --softdtw cpu
```

See [the 3D guide](docs/3dairsig.md) for data format, signer-disjoint split,
features, protocol, CPU/GPU choices and separate prepare/train/evaluate commands.
Raw biometric data and generated checkpoints are not included.

For DeepAirSig use `--dataset deepairsig --raw-data /path/to/DeepAirSig_Dataset`.
See [the DeepAirSig guide](docs/deepairsig.md) for its layout, 32/8 signer split
and complete training/evaluation command. Both loaders preserve the raw files.

DeepAirSig validation completed 20 epochs (640 steps), seed 111, with 32
development and eight held-out signers. Skilled-forgery EER is 8.00% / 3.75%
for 1 / 4 references; random-forgery EER is 0.8929% / 0.5357%. All 11 tests pass.
See [the aggregate report](validation/deepairsig-pilot.json) and
[validation details](validation/README.md). These are functional baseline results.

Validated on real 3DAirSig data: 20 epochs, 12 development signers and 3 held-out
signers; nine tests pass. Skilled-forgery EER is 13.33% with one reference and
0% with four; random-forgery EER is 0% in both cases. These are small-pilot
results, not a published-benchmark claim. See [validation](validation/3dairsig-pilot.json).

This repository preserves the history of [KAKAFEI123/DsDTW](https://github.com/KAKAFEI123/DsDTW).
GitHub returned the owner's existing fork when a second fork was requested,
so this is a separate history-preserving derivative. The original upstream
README, citation and academic-use terms follow unchanged below.

---

## The DsDTW method [1] for dynamic signature verification.

This repository implements a soft-DTW-based deep model for learning online signature representations. The soft-DTW was cited from Cuturi et al.'s work [3]. The fast CUDA implementation of soft-DTW for pytorch refers to Maghoumi et al.'s work [5]. The model structure is based on the recurrent adaptation network in our previous work [2]. To use this repository, you should first download the DeepSignDB database [4]. If you find this repository useful, please cite our papers [1,2]. 

![Alt text](./images/DsDTW.PNG)

The proposed DsDTW achieves state-of-the-art performance on several public benchmarks, including DeepSignDB (the largest online signature dataset to date), MCYT-100, SVC-task1 and SVC-task2.

![Alt text](./images/Results_on_DeepSignDB.jpg)
![Alt text](./images/Results_on_several_public_datasets.jpg)

Besides, DsDTW has won first place in the ICDAR 2021 competition for online signature verification with an obvious margin.

![Alt text](./images/Results_on_SVC2021.jpg)

### Pipeline 

- run.sh: Training the DsDTW models.

- evaluate_DsDTW.sh: Extracting signature feature vectors using the trained models, and verifying the signatures according to the protocol of DeepSignDB [4].

### Environment Setup

Python 3.7.5 or a later version is suggested. The package dependencies can be seen in the requirements.txt.

### References

[1] Jiang J, Lai S, Jin L, et al. DsDTW: Local Representation Learning With Deep soft-DTW for Dynamic Signature Verification[J]. IEEE Transactions on Information Forensics and Security, 2022, 17: 2198-2212.

[2] Lai S, Jin L. Recurrent adaptation networks for online signature verification[J]. IEEE Transactions on information forensics and security, 2018, 14(6): 1624-1637.

[3] Cuturi M, Blondel M. Soft-dtw: a differentiable loss function for time-series[C]//International conference on machine learning. PMLR, 2017: 894-903.

[4] Tolosana R, Vera-Rodriguez R, Fierrez J, et al. DeepSign: Deep on-line signature verification[J]. IEEE Transactions on Biometrics, Behavior, and Identity Science, 2021.

[5] https://github.com/Maghoumi/pytorch-softdtw-cuda.git.

### Citation and Contacts

Please consider to cite our paper if you find the code useful.

```
@article{jiang2022dsdtw,
  title={DsDTW: Local Representation Learning With Deep soft-DTW for Dynamic Signature Verification},
  author={Jiang, Jiajia and Lai, Songxuan and Jin, Lianwen and Zhu, Yecheng},
  journal={IEEE Transactions on Information Forensics and Security},
  volume={17},
  pages={2198--2212},
  year={2022},
  publisher={IEEE}
}
```

For any questions about the codes, please contact the authors by sending emails to Prof. Jin (eelwjin@scut.edu.cn), Jiajia Jiang (jiajiajiang123@qq.com), or Songxuan Lai (eesxlai@qq.com).

### Copyright

This code is free to the academic community for research purpose only. For commercial purpose usage, please contact Dr. Lianwen Jin: eelwjin@scut.edu.cn.


