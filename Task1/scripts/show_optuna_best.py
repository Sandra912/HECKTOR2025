#!/usr/bin/env python3
"""
Show best Optuna trial and top-k trial summary for HECKTOR Task 1.

Examples:
python scripts/show_optuna_best.py \
    --storage sqlite:///optuna_unet3d_multifold.db \
    --study-name hecktor_task1

python scripts/show_optuna_best.py \
    --storage sqlite:///optuna_hecktor_unet3d.db \
    --study-name hecktor_task1 \
    --top-k 10 \
    --export-csv outputs/optuna_trials_summary.csv
"""

import argparse
import json
import os
from typing import Any, Dict, List

import optuna
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect Optuna study results")
    parser.add_argument(
        "--storage",
        type=str,
        required=True,
        help="Optuna storage, e.g. sqlite:///optuna_hecktor_unet3d.db",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="hecktor_task1",
        help="Optuna study name",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Show top-k completed trials ranked by objective value",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="Optional path to export all trials as CSV",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default=None,
        help="Optional path to export best trial as JSON",
    )
    return parser.parse_args()


def safe_state_name(trial) -> str:
    try:
        return trial.state.name
    except Exception:
        return str(trial.state)


def flatten_trial(trial) -> Dict[str, Any]:
    row = {
        "trial_number": trial.number,
        "state": safe_state_name(trial),
        "value": trial.value,
    }

    for k, v in trial.params.items():
        row[f"param__{k}"] = v

    for k, v in trial.user_attrs.items():
        row[f"user_attr__{k}"] = v

    return row


def print_best_trial(study: optuna.Study):
    best = study.best_trial

    print("=" * 80)
    print("Best trial")
    print("=" * 80)
    print(f"Trial number: {best.number}")
    print(f"Objective value: {best.value:.6f}")
    print(f"State: {safe_state_name(best)}")

    print("\nParameters:")
    if len(best.params) == 0:
        print("  (none)")
    else:
        for k, v in best.params.items():
            print(f"  {k}: {v}")

    print("\nExtra metrics:")
    if len(best.user_attrs) == 0:
        print("  (none)")
    else:
        for k, v in best.user_attrs.items():
            print(f"  {k}: {v}")
    print()


def build_trials_dataframe(study: optuna.Study) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for trial in study.trials:
        rows.append(flatten_trial(trial))

    if len(rows) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if "value" in df.columns:
        # completed with numeric value first, then descending by value
        df["_has_value"] = df["value"].notna().astype(int)
        df = df.sort_values(["_has_value", "value"], ascending=[False, False]).drop(columns=["_has_value"])

    return df


def print_top_k(df: pd.DataFrame, top_k: int):
    if df.empty:
        print("No trials found.")
        return

    completed_df = df[df["value"].notna()].copy()
    if completed_df.empty:
        print("No completed trials with objective values found.")
        return

    show_df = completed_df.head(top_k)

    print("=" * 80)
    print(f"Top {min(top_k, len(show_df))} trials")
    print("=" * 80)

    display_cols = ["trial_number", "state", "value"]

    preferred_param_cols = [
        "param__learning_rate",
        "param__weight_decay",
        "param__min_gtvn_size",
        "param__loss_name",
        "param__sw_overlap",
    ]
    preferred_user_cols = [
        "user_attr__gtvn_tp",
        "user_attr__gtvn_fp",
        "user_attr__gtvn_fn",
        "user_attr__gtvn_f1agg",
        "user_attr__gtvn_agg_dsc",
        "user_attr__gtvp_mean_dsc",
        "user_attr__task1_proxy_score",
    ]

    for col in preferred_param_cols + preferred_user_cols:
        if col in show_df.columns:
            display_cols.append(col)

    print(show_df[display_cols].to_string(index=False))
    print()


def export_best_json(study: optuna.Study, path: str):
    best = study.best_trial
    payload = {
        "trial_number": best.number,
        "value": best.value,
        "state": safe_state_name(best),
        "params": best.params,
        "user_attrs": best.user_attrs,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Best trial JSON saved to: {path}")


def main():
    args = parse_args()

    study = optuna.load_study(
        study_name=args.study_name,
        storage=args.storage,
    )

    print_best_trial(study)

    df = build_trials_dataframe(study)
    print_top_k(df, args.top_k)

    if args.export_csv:
        os.makedirs(os.path.dirname(args.export_csv) or ".", exist_ok=True)
        df.to_csv(args.export_csv, index=False)
        print(f"Trial summary CSV saved to: {args.export_csv}")

    if args.export_json:
        export_best_json(study, args.export_json)


if __name__ == "__main__":
    main()