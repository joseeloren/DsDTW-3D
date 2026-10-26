import argparse
import json
from pathlib import Path
from .data import prepare


def main():
    parser = argparse.ArgumentParser(description="DsDTW baseline for XYZ in-air signatures")
    sub = parser.add_subparsers(dest="command", required=True)
    preparation = sub.add_parser("prepare", help="Prepare 3DAirSig or DeepAirSig with signer-disjoint partitions")
    preparation.add_argument("--raw-data", type=Path, required=True)
    preparation.add_argument("--output", type=Path, required=True)
    preparation.add_argument("--points", type=int, default=128)
    training = sub.add_parser("train")
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--data", type=Path, required=True)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--references", type=int, default=1)
    pipeline = sub.add_parser("run", help="Prepare, train and evaluate with 1 and 4 references")
    pipeline.add_argument("--raw-data", type=Path, required=True)
    pipeline.add_argument("--output", type=Path, required=True)
    pipeline.add_argument("--points", type=int, default=128)
    for command in (preparation, pipeline):
        command.add_argument("--dataset", choices=("auto", "3dairsig", "deepairsig"), default="auto")
    for command in (training, pipeline):
        command.add_argument("--seed", type=int, default=111)
        command.add_argument("--epochs", type=int, default=20)
        command.add_argument("--softdtw", choices=("cpu", "cuda"), default="cpu")
        command.add_argument("--steps-per-epoch", type=int)
    for command in (training, evaluation, pipeline):
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare(args.raw_data, args.output, args.points, dataset=args.dataset)
        print(json.dumps({key: value for key, value in manifest.items() if key != "files"}, indent=2))
        return
    from .engine import train, evaluate
    if args.command == "train":
        train(args.data, args.output, args.seed, args.epochs, args.device, args.softdtw, args.steps_per_epoch)
    elif args.command == "evaluate":
        evaluate(args.data, args.checkpoint, args.output, args.references, args.device)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        status = dict(phase="preparing", seed=args.seed)
        def save():
            (args.output / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        save()
        try:
            prepare(args.raw_data, args.output / "data", args.points, dataset=args.dataset)
            status["phase"] = "training"; save()
            training_result = train(args.output / "data", args.output / "train", args.seed,
                                    args.epochs, args.device, args.softdtw, args.steps_per_epoch)
            status["phase"] = "evaluating"; save()
            results = [evaluate(args.output / "data", args.output / "train/model.pt",
                                args.output / "evaluation", count, args.device) for count in (1, 4)]
            summary = dict(training=training_result, evaluation=results)
            (args.output / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            status["phase"] = "complete"; save()
        except Exception as error:
            status.update(phase="failed", error=str(error)); save()
            raise
