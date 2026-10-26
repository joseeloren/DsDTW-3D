"""Five independent seeds on both XYZ datasets; never pool test scores across seeds."""
import argparse
import json
from pathlib import Path
import platform
import numpy as np
import torch
import numba
from .data import prepare, load_prepared, verify_sources, sha256
from .engine import train, evaluate


SEEDS = (111, 222, 333, 444, 555)


def aggregate(evaluations):
    rows = []
    for references in (1, 4):
        selected = sorted([r for r in evaluations if r["references"] == references], key=lambda r: r["seed"])
        if len(selected) != len(SEEDS) or {r["seed"] for r in selected} != set(SEEDS):
            raise ValueError("Expected exactly five distinct seeds for each reference count")
        for attack in ("skilled", "random"):
            values = np.asarray([r[attack]["eer_percent"] for r in selected])
            if not np.isfinite(values).all():
                raise ValueError("Non-finite EER")
            rows.append(dict(references=references, attack=attack,
                             seeds=list(SEEDS), eer_percent=values.tolist(),
                             mean_percent=float(values.mean()),
                             sample_std_pp=float(values.std(ddof=1)),
                             population_std_pp=float(values.std(ddof=0)),
                             min_percent=float(values.min()), max_percent=float(values.max())))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deepairsig", type=Path, required=True)
    parser.add_argument("--3dairsig", dest="air3d", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    status = dict(phase="preparing", completed=[])
    def save_status():
        (args.output / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    save_status()
    summary = dict(seeds=list(SEEDS), epochs=20, points=128, softdtw="cpu",
                   statistics="Arithmetic mean and sample standard deviation (ddof=1) of five per-seed EERs; pp=percentage points",
                   runtime=dict(python=platform.python_version(), torch=str(torch.__version__),
                                numpy=np.__version__, numba=numba.__version__,
                                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                   code_sha256={p.name: sha256(p) for p in [*Path(__file__).parent.glob("*.py"),
                                                          Path(__file__).parent.parent / "soft_dtw_cuda.py"]},
                   datasets={})
    manifests = []
    try:
        for name, raw in [("deepairsig", args.deepairsig), ("3dairsig", args.air3d)]:
            root = args.output / name
            data_dir = root / "data"
            manifest = prepare(raw, data_dir, dataset=name)
            manifests.append(manifest)
            evaluations = []
            training = []
            def signer(path):
                return int(path.split("/")[1].split("_")[1] if name == "deepairsig"
                           else path.split("/")[0].split("-")[1])
            dev_hashes = {f["sha256"] for f in manifest["files"] if signer(f["path"]) in manifest["dev_users"]}
            actual_eval_hashes = {f["sha256"] for f in manifest["files"] if signer(f["path"]) in manifest["eval_users"]}
            if dev_hashes & actual_eval_hashes:
                raise ValueError("Identical source files occur across signer partitions")
            for seed in SEEDS:
                status.update(phase="training", dataset=name, seed=seed); save_status()
                run_dir = root / f"seed{seed}"
                result = train(data_dir, run_dir / "train", seed=seed, epochs=20,
                               device=args.device, softdtw="cpu")
                training.append(dict(seed=seed, **result))
                status["phase"] = "evaluating"; save_status()
                for count in (1, 4):
                    result = evaluate(data_dir, run_dir / "train/model.pt", run_dir / "evaluation",
                                      references=count, device=args.device)
                    evaluations.append(dict(seed=seed, **result))
                status["completed"].append(dict(dataset=name, seed=seed)); save_status()
            _, checked = load_prepared(data_dir)
            if not verify_sources(checked):
                raise ValueError("Source files changed")
            summary["datasets"][name] = dict(dataset=manifest["dataset"], trajectories=manifest["samples"],
                                            dev_users=manifest["dev_users"], eval_users=manifest["eval_users"],
                                            dataset_sha256=manifest["dataset_sha256"],
                                            steps_per_epoch=len(manifest["dev_users"]),
                                            cross_split_duplicate_hashes=len(dev_hashes & actual_eval_hashes),
                                            prepared_data_unchanged=True, raw_sources_unchanged=True,
                                            training=training, evaluation=evaluations,
                                            statistics=aggregate(evaluations))
        if not all(verify_sources(m) for m in manifests):
            raise ValueError("Source changed before completion")
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        status.update(phase="complete"); save_status()
    except Exception as error:
        status.update(phase="failed", error=str(error)); save_status()
        raise


if __name__ == "__main__":
    main()
