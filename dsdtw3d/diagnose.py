"""Read-only diagnosis of fixed-split 3DAirSig seed and reference sensitivity."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from .benchmark import SEEDS, aggregate
from .data import load_prepared, read_xyz, sha256, verify_sources
from .engine import device_for, train, evaluate
from .features import extract_xyz
from .metrics import dtw_l1, eer
from .model import DsDTW3D


def describe(values):
    x = np.asarray(values, dtype=float)
    return dict(values=x.tolist(), mean=float(x.mean()), sample_std=float(x.std(ddof=1)),
                min=float(x.min()), median=float(np.median(x)), max=float(x.max()))


def audit(data_dir):
    a, m = load_prepared(data_dir)
    lengths, repeats, constant, concentration = [], [], [], []
    for i, f in enumerate(m["files"]):
        xyz = read_xyz(Path(m["raw_root"]) / f["path"])
        rebuilt = extract_xyz(xyz, m["points"])
        if not np.array_equal(rebuilt, a["features"][i]):
            raise ValueError(f"Feature regeneration mismatch at {f['path']}")
        lengths.append(len(xyz))
        repeats.append(float(np.mean(np.all(np.diff(xyz, axis=0) == 0, axis=1))))
        constant.append(bool(np.ptp(xyz, axis=0).max() == 0))
        energy = np.sum(rebuilt[:, 6:] ** 2, axis=1)
        concentration.append(float(np.sort(energy)[-3:].sum() / max(energy.sum(), 1e-12)))
    return dict(files=len(lengths), features_bitwise_regenerated=True,
                raw_sources_unchanged=verify_sources(m), constant_trajectories=sum(constant),
                raw_point_counts=describe(lengths), repeated_point_fraction=describe(repeats),
                top_three_points_acceleration_energy_fraction=describe(concentration))


def tables(data_dir, checkpoint, device, references):
    a, m = load_prepared(data_dir)
    selected = device_for(device)
    saved = torch.load(checkpoint, map_location=selected, weights_only=True)
    assert saved["config"]["dataset_sha256"] == m["dataset_sha256"]
    model = DsDTW3D("cpu").to(selected)
    model.load_state_dict(saved["state_dict"]); model.eval()
    result = {}
    with torch.no_grad():
        for user in m["eval_users"]:
            ids = np.flatnonzero((a["users"] == user) & (a["splits"] == "eval"))
            embeddings = model(torch.from_numpy(a["features"][ids]).to(selected)).cpu().numpy()
            roles = a["roles"][ids]
            refs = np.flatnonzero(roles == "reference")[:references]
            result[user] = {}
            for role in ("probe", "forgery"):
                queries = np.flatnonzero(roles == role)
                result[user][role] = np.array([[dtw_l1(embeddings[q], embeddings[r])
                                               for q in queries] for r in refs])
    return result


def score(table, users, refs=None):
    refs = [0] * len(users) if refs is None else refs
    return eer(np.concatenate([table[u]["probe"][r] for u, r in zip(users, refs)]),
               np.concatenate([table[u]["forgery"][r] for u, r in zip(users, refs)]))


def derived_data(source, output, variant):
    a, m = load_prepared(source)
    original = a["features"]
    f = original.copy()
    if variant == "xyz_only":
        f[:, :, 3:] = 0
    elif variant == "smoothed_derivatives":
        # Keep XYZ exactly fixed; smooth only the path used to derive v/a.
        xyz = f[:, :, :3].astype(np.float64)
        padded = np.pad(xyz, ((0, 0), (2, 2), (0, 0)), mode="reflect")
        smoothed = sum(weight * padded[:, offset:offset + xyz.shape[1]]
                       for offset, weight in enumerate(np.array([1, 4, 6, 4, 1]) / 16))
        time = np.linspace(0, 1, xyz.shape[1])
        velocity = np.gradient(smoothed, time, axis=1)
        acceleration = np.gradient(velocity, time, axis=1)
        velocity /= np.maximum(np.sqrt(np.mean(velocity ** 2, axis=(1, 2), keepdims=True)), 1e-8)
        acceleration /= np.maximum(np.sqrt(np.mean(acceleration ** 2, axis=(1, 2), keepdims=True)), 1e-8)
        f[:, :, 3:6] = velocity
        f[:, :, 6:] = acceleration
    else:
        raise ValueError(variant)
    assert np.isfinite(f).all() and np.array_equal(f[:, :, :3], original[:, :, :3])
    output.mkdir(parents=True, exist_ok=False)
    a["features"] = f
    np.savez_compressed(output / "dataset.npz", **a)
    m.update(dataset_sha256=sha256(output / "dataset.npz"), diagnostic_variant=variant,
             parent_dataset_sha256=sha256(Path(source) / "dataset.npz"))
    (output / "manifest.json").write_text(json.dumps(m, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    status = dict(phase="audit")
    def save_status():
        (args.output / "status.json").write_text(json.dumps(status, indent=2))
    save_status()
    report = dict(seeds=list(SEEDS), audit={}, evaluation={}, interventions={})
    baseline = json.loads((args.baseline / "summary.json").read_text())
    try:
        all_tables = {}
        for dataset in ("3dairsig", "deepairsig"):
            data_dir = args.baseline / dataset / "data"
            report["audit"][dataset] = audit(data_dir)
            _, manifest = load_prepared(data_dir)
            users = manifest["eval_users"]
            status.update(phase="distance_tables", dataset=dataset); save_status()
            seed_tables = {}
            for seed in SEEDS:
                table = tables(data_dir, args.baseline / dataset / f"seed{seed}/train/model.pt",
                               args.device, 5 if dataset == "3dairsig" else 1)
                seed_tables[seed] = table
                old = next(r for r in baseline["datasets"][dataset]["evaluation"]
                           if r["seed"] == seed and r["references"] == 1)
                assert score(table, users) == old["skilled"]
                np.savez_compressed(args.output / f"{dataset}-distances-{seed}.npz",
                                    **{f"user{u}_{role}": v for u, t in table.items() for role, v in t.items()})
            all_tables[dataset] = seed_tables
            per_user = {str(u): describe([score(seed_tables[s], [u])["eer_percent"] for s in SEEDS]) for u in users}
            per_user_errors = []
            for seed in SEEDS:
                threshold = score(seed_tables[seed], users)["eer_threshold"]
                for user in users:
                    t = seed_tables[seed][user]
                    per_user_errors.append(dict(seed=seed, user=user, threshold=threshold,
                                                false_rejections=int(np.sum(t["probe"][0] > threshold)),
                                                false_acceptances=int(np.sum(t["forgery"][0] <= threshold))))
            macro = describe([np.mean([score(seed_tables[s], [u])["eer_percent"] for u in users]) for s in SEEDS])
            subsets = [dict(users=list(group), statistics=describe([score(seed_tables[s], group)["eer_percent"]
                                                                    for s in SEEDS]))
                       for group in itertools.combinations(users, 2 if dataset == "3dairsig" else 3)]
            report["evaluation"][dataset] = dict(per_user_eer=per_user, errors_at_global_eer_threshold=per_user_errors,
                                                 macro_user_eer=macro, user_subsets=subsets,
                                                 subset_sample_std=describe([g["statistics"]["sample_std"] for g in subsets]))
        three = all_tables["3dairsig"]
        users = load_prepared(args.baseline / "3dairsig/data")[1]["eval_users"]
        combinations = [dict(reference_indices=[r + 1 for r in refs],
                             statistics=describe([score(three[s], users, refs)["eer_percent"] for s in SEEDS]))
                        for refs in itertools.product(range(5), repeat=len(users))]
        report["evaluation"]["3dairsig"]["reference_combinations"] = combinations
        report["evaluation"]["3dairsig"]["reference_mean_distribution"] = describe([g["statistics"]["mean"] for g in combinations])
        report["evaluation"]["3dairsig"]["reference_seed_std_distribution"] = describe([g["statistics"]["sample_std"] for g in combinations])
        (args.output / "evaluation-analysis.json").write_text(json.dumps(report, indent=2))
        for variant in ("more_steps", "smoothed_derivatives", "xyz_only"):
            root = args.output / variant
            data_dir = args.baseline / "3dairsig/data"
            if variant != "more_steps":
                data_dir = root / "data"
                derived_data(args.baseline / "3dairsig/data", data_dir, variant)
            evaluations, training = [], []
            for seed in SEEDS:
                status.update(phase="training_intervention", dataset="3dairsig", variant=variant, seed=seed); save_status()
                run_dir = root / f"seed{seed}"
                result = train(data_dir, run_dir / "train", seed=seed, epochs=20, device=args.device,
                               softdtw="cpu", steps_per_epoch=32 if variant == "more_steps" else 12)
                training.append(dict(seed=seed, **result))
                if variant != "more_steps":
                    old = next(t for t in baseline["datasets"]["3dairsig"]["training"] if t["seed"] == seed)
                    assert result["batch_schedule_sha256"] == old["batch_schedule_sha256"]
                for count in (1, 4):
                    result = evaluate(data_dir, run_dir / "train/model.pt", run_dir / "evaluation", count, args.device)
                    evaluations.append(dict(seed=seed, **result))
            report["interventions"][variant] = dict(training=training, evaluation=evaluations, statistics=aggregate(evaluations))
        for dataset in ("3dairsig", "deepairsig"):
            _, manifest = load_prepared(args.baseline / dataset / "data")
            assert verify_sources(manifest)
        report["baseline_inputs_verified_unchanged"] = True
        report["source_code_sha256"] = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
        (args.output / "diagnosis.json").write_text(json.dumps(report, indent=2))
        status["phase"] = "complete"; save_status()
    except Exception as error:
        status.update(phase="failed", error=str(error)); save_status()
        raise


if __name__ == "__main__":
    main()
