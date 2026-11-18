#!/usr/bin/env python3
"""analyze_eval.py - everything the assignment asks for, from eval_log.csv (+ the recorded rollout datasets).
Outputs: results.md (tables + STEP), eval_manifest.csv (every rollout), step_result.json, violin.png
    pip install sequentialized_barnard_tests matplotlib pandas
    python analyze_eval.py
"""
import glob, json, math, os
from importlib.metadata import version
import numpy as np, pandas as pd

N_MAX, ALPHA_TOTAL, Z90 = 25, 0.10, 1.6449
POLICIES = ["act", "diffusion"]                      # policy A = ACT (first), policy B = Diffusion (second)
LABEL = {"act": "ACT (40k, temporal ensembling)", "diffusion": "Diffusion v2 (40k, 240x320, DDIM-10)"}

def wilson(s, n, z=Z90):
    if n == 0: return (float("nan"),) * 2
    p = s / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def load_episodes(repo):
    """{episode_index: (gripper action array, fps)} read straight from the rollout dataset parquet files."""
    root = os.path.expanduser(f"~/.cache/huggingface/lerobot/{repo}")
    files = glob.glob(f"{root}/data/**/*.parquet", recursive=True)
    if not files: return {}
    info = json.load(open(f"{root}/meta/info.json")); fps = info["fps"]
    names = info["features"]["action"]["names"]; names = names if isinstance(names, list) else next(iter(names.values()))
    gi = next(i for i, n in enumerate(names) if "gripper" in n)
    df = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", "action"]) for f in files)
    out = {}
    for e, g in df.sort_values(["episode_index", "frame_index"]).groupby("episode_index"):
        out[int(e)] = (np.stack(g["action"].to_numpy())[:, gi].astype(float), fps)
    return out

def release_time(grip, fps):
    """Seconds from policy start to the release over the tray = first gripper re-open after the first grasp
    (gripper: low = closed). Hysteresis 30%/50% of the episode's gripper range."""
    lo, hi = np.percentile(grip, 2), np.percentile(grip, 98)
    g = (grip - lo) / (hi - lo + 1e-9)
    state_open, grasped = g[0] > 0.5, False
    for i in range(1, len(g)):
        if state_open and g[i] < 0.30: state_open, grasped = False, True
        elif not state_open and g[i] > 0.50:
            state_open = True
            if grasped: return i / fps
    return float("nan")

log = pd.read_csv("eval_log.csv", dtype={"trial": int})
valid = log[log["valid"] == 1].copy()
valid["success"] = valid["success"].astype(int)

# --- completion / termination time from the recorded data ---
eps = {repo: load_episodes(repo) for repo in valid["dataset"].unique()}
def times(r):
    e = pd.to_numeric(r["episode_index"], errors="coerce")
    ep = None if pd.isna(e) else eps.get(r["dataset"], {}).get(int(e))
    if ep is None: return pd.Series({"completion_s": np.nan, "episode_len_s": np.nan})
    grip, fps = ep
    return pd.Series({"completion_s": release_time(grip, fps) if r["success"] == 1 else np.nan,
                      "episode_len_s": len(grip) / fps})
valid[["completion_s", "episode_len_s"]] = valid.apply(times, axis=1)

# --- per-policy table ---
rows = []
for p in POLICIES:
    d = valid[valid["policy"] == p]; n, s = len(d), int(d["success"].sum()); lo, hi = wilson(s, n)
    ct = d.loc[d["success"] == 1, "completion_s"].dropna()
    seen, unseen = d[d["seen"] == 1], d[d["seen"] == 0]
    stages = d["score"].astype(float).value_counts().reindex([0, 0.25, 0.5, 0.75, 1.0], fill_value=0)
    rows.append(dict(policy=LABEL[p], trials=n, successes=s, failures=n - s,
        success_rate=f"{s/n:.0%}" if n else "-", wilson90=f"[{lo:.0%}, {hi:.0%}]" if n else "-",
        completion_median_s=f"{ct.median():.1f}" if len(ct) else "-",
        completion_IQR_s=f"{ct.quantile(.25):.1f}–{ct.quantile(.75):.1f}" if len(ct) else "-",
        n_timed=len(ct), mean_progress=f"{d['score'].astype(float).mean():.2f}" if n else "-",
        seen=f"{int(seen['success'].sum())}/{len(seen)}", unseen=f"{int(unseen['success'].sum())}/{len(unseen)}",
        **{f"stage_{k}": int(v) for k, v in stages.items()}))
table = pd.DataFrame(rows)

# --- STEP (mirrored, ranking mode), chronological trial order, configs aligned ---
from sequentialized_barnard_tests import MirroredStepTest, Hypothesis, Decision
A = valid[valid["policy"] == "act"].sort_values("trial"); B = valid[valid["policy"] == "diffusion"].sort_values("trial")
assert list(A["trial"]) == list(B["trial"]), "ACT and Diffusion trials are not aligned"
a = [bool(x) for x in A["success"]]; b = [bool(x) for x in B["success"]]
k = len(POLICIES); alpha_pair = ALPHA_TOTAL / (k * (k - 1) / 2)
test = MirroredStepTest(alternative=Hypothesis.P0LessThanP1, n_max=N_MAX, alpha=alpha_pair, inference_mode="ranking")
res = test.run_on_sequence(a, b)
first_n = res.info.get("Time") if res.decision != Decision.FailToDecide else None
meaning = {Decision.AcceptAlternative: "Diffusion has the higher success rate",
           Decision.AcceptNull: "ACT has the higher success rate",
           Decision.FailToDecide: "inconclusive: difference not established (NOT evidence that they are equal)"}[res.decision]
step = dict(decision=res.decision.name, meaning=meaning, first_decision_trials_per_policy=first_n,
            n_used=len(a), n_max=N_MAX, alpha_pair=alpha_pair, alternative="P0LessThanP1 (A=ACT, B=Diffusion)",
            test="MirroredStepTest(inference_mode='ranking'), default linear risk budget",
            package_version=version("sequentialized_barnard_tests"))
json.dump(step, open("step_result.json", "w"), indent=2)

# --- CLD letters from the STEP decision ---
cld = {"act": "a", "diffusion": "a"}
if res.decision == Decision.AcceptAlternative: cld = {"diffusion": "a", "act": "b"}
if res.decision == Decision.AcceptNull: cld = {"act": "a", "diffusion": "b"}

# --- posterior violins: Beta(s+1, n-s+1) ---
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng = np.random.default_rng(0); fig, ax = plt.subplots(figsize=(5, 4.5))
for i, p in enumerate(POLICIES):
    d = valid[valid["policy"] == p]; n, s = len(d), int(d["success"].sum())
    parts = ax.violinplot(rng.beta(s + 1, n - s + 1, 20000), positions=[i], showmedians=True, widths=0.8)
    ax.text(i, 1.03, cld[p], ha="center", fontsize=14, fontweight="bold")
    ax.text(i, -0.12, f"{s}/{n}", ha="center", fontsize=10)
ax.set_xticks(range(len(POLICIES))); ax.set_xticklabels([LABEL[p].split(" (")[0] for p in POLICIES])
ax.set_ylim(-0.17, 1.1); ax.set_ylabel("success probability (posterior, Beta(s+1, n−s+1))")
ax.set_title("Cube placement: posterior success rate\n(shared letter = not statistically separated, STEP α=0.10)", fontsize=9)
fig.tight_layout(); fig.savefig("violin.png", dpi=150)

# --- manifest + report ---
man = log.merge(valid[["trial", "policy", "completion_s", "episode_len_s"]], on=["trial", "policy"], how="left")
man.to_csv("eval_manifest.csv", index=False)
with open("results.md", "w") as f:
    f.write("# Task 1 evaluation results\n\n## Per-policy summary\n\n" + table.to_markdown(index=False) + "\n\n")
    f.write("stage_x = number of trials whose furthest stage was x (0 none, 0.25 reached, 0.5 grasped, 0.75 transported, 1 placed).\n")
    f.write("Completion time = policy start → gripper release over the tray (from recorded gripper command); failures ran to the 60 s timeout.\n\n")
    f.write("## STEP\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in step.items()) + "\n\n")
    f.write("## Failures (from operator notes)\n\n" + valid[valid["success"] == 0][["trial", "grid_id", "seen", "policy", "score", "note"]].to_markdown(index=False) + "\n")
    inv = log[log["valid"] == 0]
    f.write(f"\n## Invalid (repeated) runs: {len(inv)}\n\n" + (inv[["trial", "grid_id", "policy", "note"]].to_markdown(index=False) if len(inv) else "none") + "\n")
print(table.to_string(index=False)); print(json.dumps(step, indent=2))
print("saved results.md, eval_manifest.csv, step_result.json, violin.png")