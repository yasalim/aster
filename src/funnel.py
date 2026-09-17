
import argparse
import os
import pandas as pd

STAGE_COLUMNS = [
    "initial_inspection", "on_complete_for_init_inspection",
    "first_check_eval_rgv", "initial_evaluation_queue", "initial_evaluation",
    "first_eval_approve", "rework_first_check_eval_rgv",
    "initial_evaluation_completed", "client_thinking", "queue_for_mp",
    "test_drive_mp", "on_diagnostics", "cost_estimatation",
    "final_evaluation_mov", "final_evaluation_rgv",
    "queue_for_final_evaluation", "at_the_final_evaluation",
    "final_evaluation_completed", "final_evaluation_on_approval",
    "deal_to_approve_by_rgv", "deal_to_approve_by_rov",
    "sign_by_head_manager", "on_oss", "deal_completed", "criminalist_check",
]

def load(data_dir: str):
    ev = pd.read_csv(os.path.join(data_dir, "evaluations.csv.gz"), low_memory=False)
    st = pd.read_csv(os.path.join(data_dir, "stage_times.csv.gz"), low_memory=False)
    return ev, st


def funnel_table(st: pd.DataFrame) -> pd.DataFrame:
    n = len(st)
    rows = []
    for c in STAGE_COLUMNS:
        reached = st[c].notna().sum()
        rows.append({
            "stage": c,
            "reached": reached,
            "reached_share": reached / n,
            "median_minutes": st[c].median(),
            "p95_minutes": st[c].quantile(0.95),
        })
    return pd.DataFrame(rows)


def total_duration_by_outcome(ev: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    st = st.copy()
    st["total_minutes"] = st[STAGE_COLUMNS].sum(axis=1, skipna=True)
    merged = st.merge(ev[["evaluation_id", "purchase_date"]], on="evaluation_id", how="inner")
    merged["purchased"] = merged["purchase_date"].notna()

    out = []
    for label, sub in [
        ("all", merged),
        ("purchased", merged[merged.purchased]),
        ("not_purchased", merged[~merged.purchased]),
    ]:
        q = sub["total_minutes"].quantile([0.5, 0.95])
        out.append({
            "group": label, "n": len(sub),
            "median_minutes": q[0.5], "median_days": q[0.5] / 60 / 24,
            "p95_minutes": q[0.95], "p95_days": q[0.95] / 60 / 24,
        })
    return pd.DataFrame(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset_task")
    args = parser.parse_args()

    ev, st = load(args.data_dir)
    print(f"evaluations: {len(ev)}  stage_times: {len(st)}  "
          f"пересечение по evaluation_id: {len(set(ev.evaluation_id) & set(st.evaluation_id))}")
    print()
    print(funnel_table(st).to_string(index=False))
    print()
    print(total_duration_by_outcome(ev, st).to_string(index=False))
