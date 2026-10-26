"""Read-only XYZ dataset preparation with signer-disjoint development/evaluation."""
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
from .features import extract_xyz, FEATURE_NAMES


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_xyz(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = re.split(r"[,\s]+", line.strip())
        if len(fields) != 3:
            raise ValueError(f"{path}:{number}: expected exactly three XYZ columns")
        try:
            rows.append([float(value) for value in fields])
        except ValueError as error:
            raise ValueError(f"{path}:{number}: invalid XYZ row") from error
    xyz = np.asarray(rows, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2 or not np.isfinite(xyz).all():
        raise ValueError(f"{path}: invalid trajectory")
    return xyz


def identify(root, dataset="auto"):
    layouts = []
    if dataset in ("auto", "3dairsig") and list(root.glob("subject-*/Genuine/train")):
        layouts.append("3DAirSig")
    if dataset in ("auto", "deepairsig") and list(root.glob("Genuine/subject_*/train")) and (root / "Forged").is_dir():
        layouts.append("DeepAirSig")
    if len(layouts) > 1:
        raise ValueError("Ambiguous dataset root: specify --dataset")
    return layouts[0] if layouts else None


def find_root(root, dataset="auto"):
    if dataset not in ("auto", "3dairsig", "deepairsig"):
        raise ValueError("Unknown dataset; choose auto, 3dairsig or deepairsig")
    root = Path(root).resolve()
    if identify(root, dataset):
        return root
    children = [p for p in root.iterdir() if p.is_dir() and identify(p, dataset)]
    if len(children) != 1:
        raise ValueError("Point --raw-data to one dataset root or its direct parent")
    return children[0]


def prepare(root, output, samples=128, eval_fraction=0.2, dataset="auto"):
    root = find_root(root, dataset)
    name = identify(root, dataset)
    output = Path(output)
    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be between zero and one")
    if name == "DeepAirSig":
        genuine = {int(p.name.split("_")[1]): p for p in (root / "Genuine").glob("subject_*") if p.is_dir()}
        forged = {int(p.name.split("_")[1]): p for p in (root / "Forged").glob("subject_*") if p.is_dir()}
        if genuine.keys() != forged.keys():
            raise ValueError("DeepAirSig genuine/forgery signer IDs do not match")
        subjects = [(uid, genuine[uid] / "train", genuine[uid] / "test", forged[uid]) for uid in sorted(genuine)]
    else:
        paths = sorted([p for p in root.glob("subject-*") if p.is_dir()], key=lambda p: int(p.name.split("-")[1]))
        subjects = [(int(p.name.split("-")[1]), p / "Genuine/train", p / "Genuine/test", p / "Forged") for p in paths]
    count_eval = max(2, math.ceil(len(subjects) * eval_fraction))
    if len(subjects) - count_eval < 4:
        raise ValueError("Need >= 4 development and >= 2 evaluation signers")
    eval_users = {item[0] for item in subjects[-count_eval:]}
    features, user_ids, roles, splits, source_files = [], [], [], [], []
    for uid, reference_dir, probe_dir, forgery_dir in subjects:
        for folder, role in [(reference_dir, "reference"), (probe_dir, "probe"), (forgery_dir, "forgery")]:
            files = sorted(folder.glob("*.txt"), key=lambda p: int(p.stem))
            if not files:
                raise ValueError(f"Missing samples in {folder}")
            for path in files:
                before = sha256(path)
                xyz = read_xyz(path)
                features.append(extract_xyz(xyz, samples))
                if sha256(path) != before:
                    raise ValueError(f"Source changed during preparation: {path}")
                user_ids.append(uid); roles.append(role)
                splits.append("eval" if uid in eval_users else "dev")
                source_files.append(dict(path=path.relative_to(root).as_posix(), sha256=before, points=len(xyz)))
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "dataset.npz", features=np.stack(features),
                        users=np.asarray(user_ids, dtype=np.int64), roles=np.asarray(roles), splits=np.asarray(splits))
    manifest = dict(dataset=name, samples=len(features), points=samples, feature_names=FEATURE_NAMES,
                    dev_users=sorted(set(user_ids) - eval_users), eval_users=sorted(eval_users),
                    split_policy=f"last ceil({eval_fraction:g} * signers), minimum two, held out; native train=references, test=probes",
                    raw_root=str(root), files=source_files, dataset_sha256=sha256(output / "dataset.npz"))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_prepared(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if sha256(directory / "dataset.npz") != manifest["dataset_sha256"]:
        raise ValueError("Prepared data checksum mismatch")
    with np.load(directory / "dataset.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return arrays, manifest


def verify_sources(manifest):
    root = Path(manifest["raw_root"])
    return all(sha256(root / item["path"]) == item["sha256"] for item in manifest["files"])
