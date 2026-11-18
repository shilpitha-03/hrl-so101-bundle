#!/usr/bin/env python3
"""Check that the kept episodes cover the Task 2 schedule exactly.
Usage: python3 check_coverage.py [schedule.csv] [log.csv] [dataset_root]
"""
import csv, json, sys, os, collections

sched_p = sys.argv[1] if len(sys.argv) > 1 else "task2_collection_schedule_v3.csv"
log_p = sys.argv[2] if len(sys.argv) > 2 else "task2_demos_log.csv"
root = sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser(
    "~/.cache/huggingface/lerobot/shilpitha-03/so101_stack_rpy")

sched = list(csv.DictReader(open(sched_p)))
log = list(csv.DictReader(open(log_p)))
kept = [r for r in log if r["kept"] == "1"]
disc = [r for r in log if r["kept"] == "0"]
problems = []

# 1. Dataset vs log
try:
    info = json.load(open(os.path.join(root, "meta", "info.json")))
    n_ds, f_ds = info["total_episodes"], info["total_frames"]
    print(f"Dataset: {n_ds} episodes, {f_ds} frames | Log: {len(log)} rows ({len(kept)} kept, {len(disc)} discarded)")
    if n_ds != len(log):
        problems.append(f"dataset has {n_ds} episodes but log has {len(log)} rows")
    f_log = sum(int(r["frames"]) for r in log)
    if f_log != f_ds:
        problems.append(f"frame totals differ: log {f_log} vs dataset {f_ds}")
except Exception as e:
    problems.append(f"could not read dataset info.json at {root}: {e}")

ep_ids = [int(r["dataset_episode"]) for r in log]
if ep_ids != list(range(len(log))):
    problems.append("log dataset_episode numbers are not 0..N-1 in order (missing/duplicate rows)")

# 2. Every schedule row kept exactly once, with matching config
by_row = collections.defaultdict(list)
for r in kept:
    by_row[int(r["schedule_row"])].append(r)
cols = ["type", "lighting", "T1", "T2", "T3", "placement", "pre_stacked", "pre_stack_offset"]
for i, s in enumerate(sched):
    got = by_row.get(i, [])
    if len(got) == 0:
        problems.append(f"schedule row {i} has NO kept episode ({s['type']} {s['T1']}/{s['T2']}/{s['T3']})")
    elif len(got) > 1:
        problems.append(f"schedule row {i} kept {len(got)} times: episodes {[g['dataset_episode'] for g in got]}")
    for g in got:
        bad = [c for c in cols if g[c] != s[c]]
        if bad:
            problems.append(f"episode {g['dataset_episode']} config differs from schedule row {i} in {bad}")
extra = sorted(set(by_row) - set(range(len(sched))))
if extra:
    problems.append(f"kept episodes point at schedule rows that don't exist: {extra}")

# 3. Coverage tables
print("\nKept full stacks per color arrangement (T1/T2/T3), target 7 each:")
c = collections.Counter((r["T1"], r["T2"], r["T3"]) for r in kept if r["type"] == "full")
for k, v in sorted(c.items()):
    print(f"  {'/'.join(k):24s} {v}")
print("\nKept by type:", dict(collections.Counter(r["type"] for r in kept)), " target {'full': 42, 'pre1': 12, 'pre2': 9}")
print("Kept by lighting:", dict(sorted(collections.Counter(r["lighting"] for r in kept).items())))
pre1 = collections.Counter((r["T1"], r["T2"], r["T3"]) for r in kept if r["type"] == "pre1")
pre2 = collections.Counter((r["T1"], r["T2"], r["T3"]) for r in kept if r["type"] == "pre2")
print(f"pre1 distinct states: {len(pre1)}/6   pre2 distinct states: {len(pre2)}/3")

# 4. Duration sanity
print("\nDuration (s) of kept episodes by type: min / median / max")
for t in ["full", "pre1", "pre2"]:
    d = sorted(int(r["seconds"]) for r in kept if r["type"] == t)
    if d:
        print(f"  {t:5s} {d[0]:4d} / {d[len(d)//2]:4d} / {d[-1]:4d}")
short = [r["dataset_episode"] for r in kept if r["type"] == "full" and int(r["seconds"]) < 40]
if short:
    problems.append(f"kept full stacks under 40 s (check video, may be bad discards): episodes {short}")
capped = [r["dataset_episode"] for r in kept if int(r["seconds"]) >= 99]
if capped:
    problems.append(f"kept episodes that hit the 100 s cap (check they returned home): {capped}")

# 5. Held-out tail
tail = kept[-6:]
tail_cfg = {(r["T1"], r["T2"], r["T3"]) for r in tail if r["type"] == "full"}
print(f"\nLast 6 kept episodes (validation split): {[r['dataset_episode'] for r in tail]} -> {len(tail_cfg)}/6 arrangements")

print("\nDiscarded:", [(r["dataset_episode"], r["note"]) for r in disc])
print("\n" + ("ALL CHECKS PASSED" if not problems else "PROBLEMS:\n  - " + "\n  - ".join(problems)))
print("\nTraining episode list:")
print("  --dataset.episodes=[" + ",".join(r["dataset_episode"] for r in kept) + "]")