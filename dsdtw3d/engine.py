import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from .data import load_prepared, sha256, verify_sources
from .model import DsDTW3D
from .metrics import dtw_l1, eer


def device_for(name):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose --device cpu")
    return torch.device(name)


def sample_batch(arrays, rng, tasks=4, genuine=5, forged=10):
    users = np.unique(arrays["users"][arrays["splits"] == "dev"])
    if len(users) < tasks or forged < 2 or forged % 2:
        raise ValueError("Need >= tasks development users and an even forged count >= 2")
    batch = []
    for user in rng.choice(users, tasks, replace=False):
        same = (arrays["users"] == user) & (arrays["splits"] == "dev")
        pos = np.flatnonzero(same & (arrays["roles"] != "forgery"))
        neg = np.flatnonzero(same & (arrays["roles"] == "forgery"))
        if len(pos) < genuine + 1 or len(neg) < forged // 2:
            raise ValueError(f"User {user} has too few training signatures")
        batch.extend(rng.choice(pos, genuine + 1, replace=False))
        batch.extend(rng.choice(neg, forged // 2, replace=False))
        for other in rng.choice(users[users != user], forged // 2):
            candidates = np.flatnonzero((arrays["users"] == other) & (arrays["splits"] == "dev") & (arrays["roles"] != "forgery"))
            batch.append(rng.choice(candidates))
    return np.asarray(batch, dtype=np.int64)


def train(data_dir, output, seed=111, epochs=20, device="auto", softdtw="cpu", steps_per_epoch=None):
    arrays, manifest = load_prepared(data_dir)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    selected = device_for(device)
    if softdtw == "cuda" and selected.type != "cuda":
        raise ValueError("CUDA soft-DTW needs a CUDA network device")
    model = DsDTW3D(softdtw).to(selected).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    rng = np.random.RandomState(seed)
    steps = len(np.unique(arrays["users"][arrays["splits"] == "dev"])) if steps_per_epoch is None else steps_per_epoch
    if epochs < 1 or steps < 1:
        raise ValueError("epochs and steps_per_epoch must be positive")
    batch_hash = hashlib.sha256(); started = time.time()
    config = dict(seed=seed, epochs=epochs, steps_per_epoch=steps, tasks=4, genuine=5, forged=10,
                  lr=0.01, momentum=0.9, lr_decay=0.9, gamma=5, bandwidth=0.1,
                  device=str(selected), softdtw=softdtw, feature_names=manifest["feature_names"],
                  dataset_sha256=manifest["dataset_sha256"], points=manifest["points"], torch=str(torch.__version__))
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    with (output / "epochs.jsonl").open("w", encoding="utf-8", buffering=1) as log:
        for epoch in range(epochs):
            totals = np.zeros(3)
            for _ in range(steps):
                indices = sample_batch(arrays, rng); batch_hash.update(indices.tobytes())
                batch = torch.from_numpy(arrays["features"][indices]).to(selected)
                optimizer.zero_grad()
                standard, hard, variance = model.triplet_loss(model(batch))
                loss = hard + 0.01 * variance
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                loss.backward(); optimizer.step()
                totals += [standard.item(), hard.item(), variance.item()]
            row = dict(epoch=epoch + 1, lr=optimizer.param_groups[0]["lr"],
                       standard=totals[0] / steps, hard=totals[1] / steps, variance=totals[2] / steps)
            log.write(json.dumps(row) + "\n"); print(json.dumps(row), flush=True)
            scheduler.step()
    torch.save(dict(state_dict=model.state_dict(), config=config), output / "model.pt")
    result = dict(complete=True, epochs=epochs, steps=epochs * steps, batch_schedule_sha256=batch_hash.hexdigest(),
                  model_sha256=sha256(output / "model.pt"), seconds=time.time() - started,
                  prepared_data_unchanged=sha256(Path(data_dir) / "dataset.npz") == manifest["dataset_sha256"],
                  raw_sources_unchanged=verify_sources(manifest))
    assert result["prepared_data_unchanged"] and result["raw_sources_unchanged"]
    (output / "training.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def evaluate(data_dir, checkpoint, output, references=1, device="auto"):
    arrays, manifest = load_prepared(data_dir)
    selected = device_for(device)
    saved = torch.load(checkpoint, map_location=selected, weights_only=True)
    if saved["config"]["dataset_sha256"] != manifest["dataset_sha256"]:
        raise ValueError("Model and prepared dataset do not match")
    model = DsDTW3D("cpu").to(selected); model.load_state_dict(saved["state_dict"]); model.eval()
    all_users = np.unique(arrays["users"][arrays["splits"] == "eval"])
    if len(all_users) < 2:
        raise ValueError("Need two evaluation users for random forgeries")
    embeddings = {}
    with torch.no_grad():
        for user in all_users:
            indices = np.flatnonzero((arrays["users"] == user) & (arrays["splits"] == "eval"))
            output_features = model(torch.from_numpy(arrays["features"][indices]).to(selected)).cpu().numpy()
            embeddings.update(zip(indices.tolist(), output_features))
    genuine, skilled, random_scores = [], [], []
    for user in all_users:
        same = (arrays["users"] == user) & (arrays["splits"] == "eval")
        refs = np.flatnonzero(same & (arrays["roles"] == "reference"))
        if references < 1 or len(refs) < references:
            raise ValueError(f"User {user} has too few native reference signatures")
        refs = refs[:references]
        groups = [(genuine, np.flatnonzero(same & (arrays["roles"] == "probe"))),
                  (skilled, np.flatnonzero(same & (arrays["roles"] == "forgery"))),
                  (random_scores, np.flatnonzero((arrays["splits"] == "eval") & (arrays["users"] != user) & (arrays["roles"] == "probe")))]
        for target, queries in groups:
            for query in queries:
                target.append(float(np.mean([dtw_l1(embeddings[query], embeddings[ref]) for ref in refs])))
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f"scores-ref{references}.npz", genuine=genuine, skilled=skilled, random=random_scores)
    report = dict(dataset="3DAirSig", evaluation_users=len(all_users), references=references,
                  reference_policy="first k numerically ordered files from native Genuine/train",
                  probe_policy="all native Genuine/test; random uses all other held-out signers' Genuine/test",
                  distance="exact DTW L1 / (sequence lengths sum); mean across references",
                  skilled=eer(genuine, skilled), random=eer(genuine, random_scores),
                  model_sha256=sha256(checkpoint), dataset_sha256=manifest["dataset_sha256"],
                  raw_sources_unchanged=verify_sources(manifest))
    assert report["raw_sources_unchanged"]
    (output / f"evaluation-ref{references}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report
