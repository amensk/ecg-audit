"""Command-line interface for ecg-audit.

Examples
--------
    ecg-audit audit predictions.csv --conditions MI,STTC,NORM \\
        --output report.json --plot reliability/
    ecg-audit recalibrate --method isotonic \\
        --cal-data cal.csv --test-data test.csv --output recal.csv
    ecg-audit submit --model ST-MEM --dataset PTB-XL --auroc 0.838 --ece 0.124 \\
        --leaderboard leaderboard.json

CSV format: for each condition COND, a probability column ``COND_prob`` and a
binary label column ``COND_true``.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np


def _read_probs_labels(csv_path, conditions):
    import pandas as pd
    df = pd.read_csv(csv_path)
    probs, labels = [], []
    for c in conditions:
        pcol, lcol = f"{c}_prob", f"{c}_true"
        if pcol not in df.columns:
            raise SystemExit(f"missing column '{pcol}' in {csv_path}")
        probs.append(df[pcol].to_numpy(float))
        labels.append(df[lcol].to_numpy(float) if lcol in df.columns
                      else np.full(len(df), np.nan))
    return np.array(probs).T, np.array(labels).T


def cmd_audit(args):
    from .metrics import CalibrationAuditor
    from .plotting import reliability_diagram
    conds = [c.strip() for c in args.conditions.split(",")]
    probs, labels = _read_probs_labels(args.predictions, conds)
    report = CalibrationAuditor(classes=conds, n_bins=args.bins).audit(probs, labels)
    print(repr(report))
    if args.output:
        Path(args.output).write_text(json.dumps(report.to_dict(), indent=2))
        print(f"wrote {args.output}")
    if args.plot:
        outdir = Path(args.plot); outdir.mkdir(parents=True, exist_ok=True)
        for ci, c in enumerate(conds):
            if report.per_class[c].get("included"):
                reliability_diagram(labels[:, ci], probs[:, ci], title=c,
                                    out_path=str(outdir / f"reliability_{c}.png"), label=c)
        print(f"wrote reliability plots to {outdir}/")
    return 0


def cmd_recalibrate(args):
    from .recalibration import recalibrate_multilabel
    import pandas as pd
    conds = [c.strip() for c in args.conditions.split(",")] if args.conditions else None
    if conds is None:
        df = pd.read_csv(args.cal_data)
        conds = [c[:-5] for c in df.columns if c.endswith("_prob")]
    p_cal, y_cal = _read_probs_labels(args.cal_data, conds)
    p_test, _ = _read_probs_labels(args.test_data, conds)
    recal = recalibrate_multilabel(args.method, p_cal, y_cal, p_test)
    out = pd.DataFrame({f"{c}_prob": recal[:, i] for i, c in enumerate(conds)})
    out.to_csv(args.output, index=False)
    print(f"recalibrated ({args.method}) -> {args.output}")
    return 0


def cmd_submit(args):
    from .leaderboard import make_entry, submit
    entry = make_entry(args.model, args.dataset, args.auroc, args.ece,
                       brier=args.brier, sce=args.sce)
    res = submit(entry, args.leaderboard)
    print(f"submitted {args.model} on {args.dataset}: {res['n_entries']} entries in {res['path']}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="ecg-audit",
                                description="Calibration auditing for ECG foundation models")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", help="audit calibration of a predictions CSV")
    a.add_argument("predictions")
    a.add_argument("--conditions", required=True, help="comma-separated condition names")
    a.add_argument("--output", help="output JSON report path")
    a.add_argument("--plot", help="directory for reliability diagrams")
    a.add_argument("--bins", type=int, default=15)
    a.set_defaults(func=cmd_audit)

    r = sub.add_parser("recalibrate", help="recalibrate probabilities")
    r.add_argument("--method", default="isotonic",
                   choices=["platt", "isotonic", "temperature"])
    r.add_argument("--cal-data", required=True)
    r.add_argument("--test-data", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--conditions", help="comma-separated; inferred from cal-data if omitted")
    r.set_defaults(func=cmd_recalibrate)

    s = sub.add_parser("submit", help="append an entry to a JSON leaderboard")
    s.add_argument("--model", required=True)
    s.add_argument("--dataset", required=True)
    s.add_argument("--auroc", type=float, required=True)
    s.add_argument("--ece", type=float, required=True)
    s.add_argument("--brier", type=float, default=None)
    s.add_argument("--sce", type=float, default=None)
    s.add_argument("--leaderboard", default="leaderboard.json")
    s.set_defaults(func=cmd_submit)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
