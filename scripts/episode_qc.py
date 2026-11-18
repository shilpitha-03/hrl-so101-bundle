#!/usr/bin/env python3
"""
episode_qc.py - flag suspicious teleop demos in a LeRobot v3 dataset.

Reads the parquet files directly (no video decoding), so it is fast and does not
depend on a specific LeRobot Python API version. These are the same per-frame
`action` / `observation.state` columns that the LeRobot visualizer plots.

Usage:
  python episode_qc.py --root ~/.cache/huggingface/lerobot/<user>/<dataset>
  python episode_qc.py --root <path> --manifest demos.csv --gripper-closed low

Optional manifest CSV columns: episode_index, grid_id[, keep, operator]  (demos.csv from record_loop.sh)
  -> keep=0 episodes are skipped
  -> enables "approach looks different from other demos at the same grid node".

Outputs (next to --out prefix): <out>_metrics.csv, <out>_overlay.png
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

ROBUST_Z = 3.0  # |robust z| above this = flagged


def load(root):
    info = json.load(open(os.path.join(root, "meta", "info.json")))
    files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"No parquet files under {root}/data")
    cols = ["episode_index", "frame_index", "timestamp", "action", "observation.state"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    names = info["features"]["action"].get("names")
    if isinstance(names, dict):  # older formats: {"motors": [...]}
        names = next(iter(names.values()))
    return df.sort_values(["episode_index", "frame_index"]), info["fps"], list(names)


def robust_z(x):
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    return (x - med) / (mad + 1e-9)


def smooth_speed(a, jstd, arm, fps, win_s=0.5):
    """Arm joint speed (scale-normalized). Positions are smoothed first so encoder /
    hand jitter doesn't look like motion during pauses."""
    k = max(1, int(win_s * fps))
    x = a[:, arm] / jstd[arm]
    xp = np.pad(x, ((k // 2, k - 1 - k // 2), (0, 0)), mode="edge")
    xs = np.stack([np.convolve(xp[:, j], np.ones(k) / k, mode="valid") for j in range(x.shape[1])], 1)
    return np.linalg.norm(np.diff(xs, axis=0), axis=1) * fps


def runs(mask):
    """Return list of (start, end) index pairs of True runs."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            out.append((start, i)); start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def resample(x, n=50):
    t = np.linspace(0, 1, len(x))
    tn = np.linspace(0, 1, n)
    return np.stack([np.interp(tn, t, x[:, j]) for j in range(x.shape[1])], 1)


def gripper_events(g, closed_low, lo=0.30, hi=0.50):
    """Hysteresis on normalized gripper signal. Returns list of close frames, open frames."""
    closed = g < lo if closed_low else g > hi
    opened = g > hi if closed_low else g < lo
    state = bool(closed[0])
    closes, opens = [], []
    for i in range(1, len(g)):
        if not state and closed[i]:
            state = True; closes.append(i)
        elif state and opened[i]:
            state = False; opens.append(i)
    return closes, opens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--gripper-closed", choices=["low", "high"], default="low",
                    help="Is the gripper value LOW or HIGH when closed? Check one episode first.")
    ap.add_argument("--max-pause", type=float, default=1.0, help="flag mid-episode pauses longer than this (s)")
    ap.add_argument("--max-lead", type=float, default=2.0, help="flag idle time at episode start longer than this (s)")
    ap.add_argument("--out", default="qc")
    args = ap.parse_args()

    df, fps, names = load(args.root)
    gi = next((i for i, n in enumerate(names) if "gripper" in n.lower()), len(names) - 1)
    ri = next((i for i, n in enumerate(names) if "wrist_roll" in n.lower()), None)
    arm = [i for i in range(len(names)) if i != gi]
    closed_low = args.gripper_closed == "low"

    eps = {e: (np.stack(g["action"].to_numpy()).astype(float),
               np.stack(g["observation.state"].to_numpy()).astype(float),
               g["timestamp"].to_numpy())
           for e, g in df.groupby("episode_index")}

    man = None
    if args.manifest:
        man = pd.read_csv(args.manifest).set_index("episode_index")
        if "keep" in man:
            drop = set(man.index[man["keep"] == 0])
            eps = {e: v for e, v in eps.items() if e not in drop}
            print(f"Skipping {len(drop)} discarded episodes: {sorted(drop)}")

    all_act = np.concatenate([a for a, _, _ in eps.values()])
    # Per-joint scale = 5-95 percentile range, floored so near-constant joints
    # (e.g. wrist roll) don't blow up sensor jitter into "motion".
    jstd = np.maximum(np.percentile(all_act, 95, 0) - np.percentile(all_act, 5, 0), 5.0)
    g_lo, g_hi = np.percentile(all_act[:, gi], [2, 98])
    print(f"fps={fps}  episodes={len(eps)}  joints={names}")
    print(f"gripper '{names[gi]}' range ~ [{g_lo:.1f}, {g_hi:.1f}]  (closed assumed {args.gripper_closed})")
    # Idle = arm still AND gripper still (squeezing the gripper with a still arm is not a pause).
    # Separate thresholds so fast gripper snaps don't inflate the arm threshold.
    arm_sp = np.concatenate([smooth_speed(a, jstd, arm, fps) for a, _, _ in eps.values()])
    grip_sp = np.concatenate([smooth_speed(a, jstd, [gi], fps) for a, _, _ in eps.values()])
    arm_thr, grip_thr = 0.15 * np.percentile(arm_sp, 95), 0.15 * np.percentile(grip_sp, 95)

    rows, carry, approach = [], {}, {}
    for e, (a, s, ts) in eps.items():
        n = len(a)
        g = (a[:, gi] - g_lo) / (g_hi - g_lo + 1e-9)
        closes, opens = gripper_events(g, closed_low)
        first_close = closes[0] if closes else None
        after = [o for o in opens if first_close is not None and o > first_close]
        release = after[-1] if after else None
        grasp_closes = [c for c in closes if release is None or c < release]  # closes after release = homing
        grasp = grasp_closes[0] if grasp_closes else None
        idle = (smooth_speed(a, jstd, arm, fps) < arm_thr) & (smooth_speed(a, jstd, [gi], fps) < grip_thr)
        r = []
        for a_, b in runs(idle):  # merge idle runs split by brief jitter (< 0.3 s)
            if r and a_ - r[-1][1] < 0.3 * fps:
                r[-1] = (r[-1][0], b)
            else:
                r.append((a_, b))
        lead = (r[0][1] - r[0][0]) / fps if r and r[0][0] == 0 else 0.0
        mids = [(a_, b) for a_, b in r if a_ > 0 and b < len(idle)]
        mid = [(b - a_) / fps for a_, b in mids]
        phase = ""
        if mids:
            a_, b = max(mids, key=lambda x: x[1] - x[0])
            c = (a_ + b) / 2
            phase = ("start" if grasp is None and release is None else
                     "before grasp" if grasp is not None and c < grasp else
                     "holding cube" if release is not None and c < release else "after release")
        dts = np.diff(ts)
        row = dict(
            episode=e, frames=n, duration_s=n / fps,
            dropped_frames=int((dts > 1.5 / fps).sum()),
            lead_idle_s=lead, longest_pause_s=max(mid) if mid else 0.0, pause_phase=phase,
            jerk=float(np.abs(np.diff(a[:, arm] / jstd[arm], 2, axis=0)).mean() * fps ** 2),
            track_err=float(np.abs((a[:, arm] - s[:, arm]) / jstd[arm]).mean()),
            n_grasps=len(grasp_closes), released=release is not None,
            reopen_after_grasp=(float((g[grasp:] if closed_low else 1 - g[grasp:]).max()) if grasp is not None else np.nan),
            start_state=s[0, arm],
            release_state=s[release, arm] if release is not None else None,
            roll_at_grasp=s[grasp, ri] if (grasp is not None and ri is not None) else np.nan,
        )
        if grasp is not None and release is not None and release - grasp > 5:
            carry[e] = resample(s[grasp:release, arm] / jstd[arm])
        if grasp is not None and grasp > 5:
            approach[e] = resample(s[:grasp, arm] / jstd[arm])
        rows.append(row)
    m = pd.DataFrame(rows).set_index("episode")

    # Consistency vs. the rest of the dataset
    med_start = np.median(np.stack(m["start_state"]), 0)
    m["start_dev"] = [np.linalg.norm((x - med_start) / jstd[arm]) for x in m["start_state"]]
    rel = m["release_state"].dropna()
    if len(rel):
        med_rel = np.median(np.stack(rel), 0)
        m["release_dev"] = [np.linalg.norm((x - med_rel) / jstd[arm]) if x is not None else np.nan
                            for x in m["release_state"]]
    if carry:
        med_c = np.median(np.stack(list(carry.values())), 0)
        m["carry_dev"] = pd.Series({e: np.sqrt(((c - med_c) ** 2).mean()) for e, c in carry.items()})
    if man is not None and approach:
        m = m.join(man, how="left")
        dev = {}
        for gid, grp in man.groupby("grid_id"):
            ids = [e for e in grp.index if e in approach]
            if len(ids) < 2:
                continue
            med_a = np.median(np.stack([approach[e] for e in ids]), 0)
            for e in ids:
                dev[e] = np.sqrt(((approach[e] - med_a) ** 2).mean())
        m["approach_dev"] = pd.Series(dev)

    # Flags
    checks = ["duration_s", "jerk", "track_err",
              "start_dev", "release_dev", "carry_dev", "approach_dev"]
    reasons = {e: [] for e in m.index}
    for c in checks:
        if c not in m or m[c].notna().sum() < 4:
            continue
        z = pd.Series(robust_z(m[c]), index=m.index)
        for e in m.index[(z > ROBUST_Z) | ((z < -ROBUST_Z) & (c == "duration_s"))]:
            reasons[e].append(f"{c} (z={z[e]:.1f})" if abs(z[e]) < 10 else f"{c} (z>10)")
    if m["roll_at_grasp"].notna().sum() >= 4:
        z = pd.Series(robust_z(m["roll_at_grasp"]), index=m.index)
        for e in m.index[z.abs() > ROBUST_Z]:
            reasons[e].append(f"wrist roll at grasp (z={z[e]:.1f})" if abs(z[e]) < 10 else "wrist roll at grasp (z>10)")
    for e, r in m.iterrows():
        if r.n_grasps == 0: reasons[e].append("no grasp detected")
        if r.n_grasps > 1: reasons[e].append(f"{r.n_grasps} closes before release (regrasp?)")
        if not r.released:
            extra = (f" (gripper reopened to only {100 * r.reopen_after_grasp:.0f}% of its range after grasp)"
                     if r.n_grasps and not np.isnan(r.reopen_after_grasp) else "")
            reasons[e].append("no release detected" + extra)
        if r.dropped_frames: reasons[e].append(f"{r.dropped_frames} timestamp gaps")
        if r.longest_pause_s > args.max_pause: reasons[e].append(f"mid-episode pause {r.longest_pause_s:.1f}s ({r.pause_phase})")
        if r.lead_idle_s > args.max_lead: reasons[e].append(f"idle start {r.lead_idle_s:.1f}s")
    m["flags"] = ["; ".join(reasons[e]) for e in m.index]

    out = m.drop(columns=["start_state", "release_state"])
    out.to_csv(f"{args.out}_metrics.csv")
    flagged = out[out["flags"] != ""]
    print(f"\n{len(flagged)}/{len(out)} episodes flagged -> review these videos:")
    for e, r in flagged.iterrows():
        print(f"  ep {e:4d}: {r["flags"]}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(len(names), 1, figsize=(9, 2 * len(names)), sharex=True)
        for e, (a, s, _) in eps.items():
            t = np.linspace(0, 1, len(a))
            bad = bool(reasons[e])
            for j, ax in enumerate(axs):
                ax.plot(t, a[:, j], color="tab:red" if bad else "0.6",
                        lw=1.2 if bad else 0.6, alpha=0.9 if bad else 0.5, zorder=3 if bad else 1)
        for j, ax in enumerate(axs):
            ax.set_ylabel(names[j], fontsize=8)
        axs[-1].set_xlabel("normalized episode time (red = flagged)")
        fig.tight_layout(); fig.savefig(f"{args.out}_overlay.png", dpi=120)
        print(f"\nSaved {args.out}_metrics.csv and {args.out}_overlay.png")
    except ImportError:
        print(f"\nSaved {args.out}_metrics.csv (install matplotlib for the overlay plot)")


if __name__ == "__main__":
    main()
    
    
    
    # #!/usr/bin/env python3
# """
# episode_qc.py - flag suspicious teleop demos in a LeRobot v3 dataset.

# Reads the parquet files directly (no video decoding), so it is fast and does not
# depend on a specific LeRobot Python API version. These are the same per-frame
# `action` / `observation.state` columns that the LeRobot visualizer plots.

# Usage:
#   python episode_qc.py --root ~/.cache/huggingface/lerobot/<user>/<dataset>
#   python episode_qc.py --root <path> --manifest demos.csv --gripper-closed low

# Optional manifest CSV columns: episode_index, grid_id[, keep, operator]  (demos.csv from record_loop.sh)
#   -> keep=0 episodes are skipped
#   -> enables "approach looks different from other demos at the same grid node".

# Outputs (next to --out prefix): <out>_metrics.csv, <out>_overlay.png
# """
# import argparse, glob, json, os
# import numpy as np
# import pandas as pd

# ROBUST_Z = 3.0  # |robust z| above this = flagged


# def load(root):
#     info = json.load(open(os.path.join(root, "meta", "info.json")))
#     files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
#     if not files:
#         raise SystemExit(f"No parquet files under {root}/data")
#     cols = ["episode_index", "frame_index", "timestamp", "action", "observation.state"]
#     df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
#     names = info["features"]["action"].get("names")
#     if isinstance(names, dict):  # older formats: {"motors": [...]}
#         names = next(iter(names.values()))
#     return df.sort_values(["episode_index", "frame_index"]), info["fps"], list(names)


# def robust_z(x):
#     x = np.asarray(x, float)
#     med = np.nanmedian(x)
#     mad = np.nanmedian(np.abs(x - med)) * 1.4826
#     return (x - med) / (mad + 1e-9)


# def smooth_speed(a, jstd, arm, fps, win_s=0.5):
#     """Arm joint speed (scale-normalized). Positions are smoothed first so encoder /
#     hand jitter doesn't look like motion during pauses."""
#     k = max(1, int(win_s * fps))
#     x = a[:, arm] / jstd[arm]
#     xp = np.pad(x, ((k // 2, k - 1 - k // 2), (0, 0)), mode="edge")
#     xs = np.stack([np.convolve(xp[:, j], np.ones(k) / k, mode="valid") for j in range(x.shape[1])], 1)
#     return np.linalg.norm(np.diff(xs, axis=0), axis=1) * fps


# def runs(mask):
#     """Return list of (start, end) index pairs of True runs."""
#     out, start = [], None
#     for i, m in enumerate(mask):
#         if m and start is None:
#             start = i
#         if not m and start is not None:
#             out.append((start, i)); start = None
#     if start is not None:
#         out.append((start, len(mask)))
#     return out


# def resample(x, n=50):
#     t = np.linspace(0, 1, len(x))
#     tn = np.linspace(0, 1, n)
#     return np.stack([np.interp(tn, t, x[:, j]) for j in range(x.shape[1])], 1)


# def gripper_events(g, closed_low, lo=0.30, hi=0.50):
#     """Hysteresis on normalized gripper signal. Returns list of close frames, open frames."""
#     closed = g < lo if closed_low else g > hi
#     opened = g > hi if closed_low else g < lo
#     state = bool(closed[0])
#     closes, opens = [], []
#     for i in range(1, len(g)):
#         if not state and closed[i]:
#             state = True; closes.append(i)
#         elif state and opened[i]:
#             state = False; opens.append(i)
#     return closes, opens


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--root", required=True)
#     ap.add_argument("--manifest", default=None)
#     ap.add_argument("--gripper-closed", choices=["low", "high"], default="low",
#                     help="Is the gripper value LOW or HIGH when closed? Check one episode first.")
#     ap.add_argument("--max-pause", type=float, default=1.0, help="flag mid-episode pauses longer than this (s)")
#     ap.add_argument("--max-lead", type=float, default=2.0, help="flag idle time at episode start longer than this (s)")
#     ap.add_argument("--out", default="qc")
#     args = ap.parse_args()

#     df, fps, names = load(args.root)
#     gi = next((i for i, n in enumerate(names) if "gripper" in n.lower()), len(names) - 1)
#     ri = next((i for i, n in enumerate(names) if "wrist_roll" in n.lower()), None)
#     arm = [i for i in range(len(names)) if i != gi]
#     closed_low = args.gripper_closed == "low"

#     eps = {e: (np.stack(g["action"].to_numpy()).astype(float),
#                np.stack(g["observation.state"].to_numpy()).astype(float),
#                g["timestamp"].to_numpy())
#            for e, g in df.groupby("episode_index")}

#     man = None
#     if args.manifest:
#         man = pd.read_csv(args.manifest).set_index("episode_index")
#         if "keep" in man:
#             drop = set(man.index[man["keep"] == 0])
#             eps = {e: v for e, v in eps.items() if e not in drop}
#             print(f"Skipping {len(drop)} discarded episodes: {sorted(drop)}")

#     all_act = np.concatenate([a for a, _, _ in eps.values()])
#     # Per-joint scale = 5-95 percentile range, floored so near-constant joints
#     # (e.g. wrist roll) don't blow up sensor jitter into "motion".
#     jstd = np.maximum(np.percentile(all_act, 95, 0) - np.percentile(all_act, 5, 0), 5.0)
#     g_lo, g_hi = np.percentile(all_act[:, gi], [2, 98])
#     print(f"fps={fps}  episodes={len(eps)}  joints={names}")
#     print(f"gripper '{names[gi]}' range ~ [{g_lo:.1f}, {g_hi:.1f}]  (closed assumed {args.gripper_closed})")
#     speeds = np.concatenate([smooth_speed(a, jstd, arm, fps) for a, _, _ in eps.values()])
#     idle_thr = 0.15 * np.percentile(speeds, 95)

#     rows, carry, approach = [], {}, {}
#     for e, (a, s, ts) in eps.items():
#         n = len(a)
#         idle = smooth_speed(a, jstd, arm, fps) < idle_thr
#         r = []
#         for a_, b in runs(idle):  # merge idle runs split by brief jitter (< 0.3 s)
#             if r and a_ - r[-1][1] < 0.3 * fps:
#                 r[-1] = (r[-1][0], b)
#             else:
#                 r.append((a_, b))
#         lead = (r[0][1] - r[0][0]) / fps if r and r[0][0] == 0 else 0.0
#         mid = [(b - a_) / fps for a_, b in r if a_ > 0 and b < len(idle)]
#         g = (a[:, gi] - g_lo) / (g_hi - g_lo + 1e-9)
#         closes, opens = gripper_events(g, closed_low)
#         grasp = closes[0] if closes else None
#         release = next((o for o in opens if grasp is not None and o > grasp), None)
#         dts = np.diff(ts)
#         row = dict(
#             episode=e, frames=n, duration_s=n / fps,
#             dropped_frames=int((dts > 1.5 / fps).sum()),
#             lead_idle_s=lead, longest_pause_s=max(mid) if mid else 0.0,
#             jerk=float(np.abs(np.diff(a[:, arm] / jstd[arm], 2, axis=0)).mean() * fps ** 2),
#             track_err=float(np.abs((a[:, arm] - s[:, arm]) / jstd[arm]).mean()),
#             n_grasps=len(closes), released=release is not None,
#             reopen_after_grasp=(float((g[grasp:] if closed_low else 1 - g[grasp:]).max()) if grasp is not None else np.nan),
#             start_state=s[0, arm],
#             release_state=s[release, arm] if release is not None else None,
#             roll_at_grasp=s[grasp, ri] if (grasp is not None and ri is not None) else np.nan,
#         )
#         if grasp is not None and release is not None and release - grasp > 5:
#             carry[e] = resample(s[grasp:release, arm] / jstd[arm])
#         if grasp is not None and grasp > 5:
#             approach[e] = resample(s[:grasp, arm] / jstd[arm])
#         rows.append(row)
#     m = pd.DataFrame(rows).set_index("episode")

#     # Consistency vs. the rest of the dataset
#     med_start = np.median(np.stack(m["start_state"]), 0)
#     m["start_dev"] = [np.linalg.norm((x - med_start) / jstd[arm]) for x in m["start_state"]]
#     rel = m["release_state"].dropna()
#     if len(rel):
#         med_rel = np.median(np.stack(rel), 0)
#         m["release_dev"] = [np.linalg.norm((x - med_rel) / jstd[arm]) if x is not None else np.nan
#                             for x in m["release_state"]]
#     if carry:
#         med_c = np.median(np.stack(list(carry.values())), 0)
#         m["carry_dev"] = pd.Series({e: np.sqrt(((c - med_c) ** 2).mean()) for e, c in carry.items()})
#     if man is not None and approach:
#         m = m.join(man, how="left")
#         dev = {}
#         for gid, grp in man.groupby("grid_id"):
#             ids = [e for e in grp.index if e in approach]
#             if len(ids) < 2:
#                 continue
#             med_a = np.median(np.stack([approach[e] for e in ids]), 0)
#             for e in ids:
#                 dev[e] = np.sqrt(((approach[e] - med_a) ** 2).mean())
#         m["approach_dev"] = pd.Series(dev)

#     # Flags
#     checks = ["duration_s", "jerk", "track_err",
#               "start_dev", "release_dev", "carry_dev", "approach_dev"]
#     reasons = {e: [] for e in m.index}
#     for c in checks:
#         if c not in m or m[c].notna().sum() < 4:
#             continue
#         z = pd.Series(robust_z(m[c]), index=m.index)
#         for e in m.index[(z > ROBUST_Z) | ((z < -ROBUST_Z) & (c == "duration_s"))]:
#             reasons[e].append(f"{c} (z={z[e]:.1f})" if abs(z[e]) < 10 else f"{c} (z>10)")
#     if m["roll_at_grasp"].notna().sum() >= 4:
#         z = pd.Series(robust_z(m["roll_at_grasp"]), index=m.index)
#         for e in m.index[z.abs() > ROBUST_Z]:
#             reasons[e].append(f"wrist roll at grasp (z={z[e]:.1f})" if abs(z[e]) < 10 else "wrist roll at grasp (z>10)")
#     for e, r in m.iterrows():
#         if r.n_grasps == 0: reasons[e].append("no grasp detected")
#         if r.n_grasps > 1: reasons[e].append(f"{r.n_grasps} grasps (regrasp?)")
#         if not r.released:
#             extra = (f" (gripper reopened to only {100 * r.reopen_after_grasp:.0f}% of its range after grasp)"
#                      if r.n_grasps and not np.isnan(r.reopen_after_grasp) else "")
#             reasons[e].append("no release detected" + extra)
#         if r.dropped_frames: reasons[e].append(f"{r.dropped_frames} timestamp gaps")
#         if r.longest_pause_s > args.max_pause: reasons[e].append(f"mid-episode pause {r.longest_pause_s:.1f}s")
#         if r.lead_idle_s > args.max_lead: reasons[e].append(f"idle start {r.lead_idle_s:.1f}s")
#     m["flags"] = ["; ".join(reasons[e]) for e in m.index]

#     out = m.drop(columns=["start_state", "release_state"])
#     out.to_csv(f"{args.out}_metrics.csv")
#     flagged = out[out["flags"] != ""]
#     print(f"\n{len(flagged)}/{len(out)} episodes flagged -> review these videos:")
#     for e, r in flagged.iterrows():
#         print(f"  ep {e:4d}: {r["flags"]}")

#     try:
#         import matplotlib; matplotlib.use("Agg")
#         import matplotlib.pyplot as plt
#         fig, axs = plt.subplots(len(names), 1, figsize=(9, 2 * len(names)), sharex=True)
#         for e, (a, s, _) in eps.items():
#             t = np.linspace(0, 1, len(a))
#             bad = bool(reasons[e])
#             for j, ax in enumerate(axs):
#                 ax.plot(t, a[:, j], color="tab:red" if bad else "0.6",
#                         lw=1.2 if bad else 0.6, alpha=0.9 if bad else 0.5, zorder=3 if bad else 1)
#         for j, ax in enumerate(axs):
#             ax.set_ylabel(names[j], fontsize=8)
#         axs[-1].set_xlabel("normalized episode time (red = flagged)")
#         fig.tight_layout(); fig.savefig(f"{args.out}_overlay.png", dpi=120)
#         print(f"\nSaved {args.out}_metrics.csv and {args.out}_overlay.png")
#     except ImportError:
#         print(f"\nSaved {args.out}_metrics.csv (install matplotlib for the overlay plot)")


# if __name__ == "__main__":
#     main()



























# #!/usr/bin/env python3
# """
# episode_qc.py - flag suspicious teleop demos in a LeRobot v3 dataset.

# Reads the parquet files directly (no video decoding), so it is fast and does not
# depend on a specific LeRobot Python API version. These are the same per-frame
# `action` / `observation.state` columns that the LeRobot visualizer plots.

# Usage:
#   python episode_qc.py --root ~/.cache/huggingface/lerobot/<user>/<dataset>
#   python episode_qc.py --root <path> --manifest demos.csv --gripper-closed low

# Optional manifest CSV columns: episode_index, grid_id[, operator]
#   -> enables "approach looks different from other demos at the same grid node".

# Outputs (next to --out prefix): <out>_metrics.csv, <out>_overlay.png
# """
# import argparse, glob, json, os
# import numpy as np
# import pandas as pd

# ROBUST_Z = 3.0  # |robust z| above this = flagged


# def load(root):
#     info = json.load(open(os.path.join(root, "meta", "info.json")))
#     files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
#     if not files:
#         raise SystemExit(f"No parquet files under {root}/data")
#     cols = ["episode_index", "frame_index", "timestamp", "action", "observation.state"]
#     df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
#     names = info["features"]["action"].get("names")
#     if isinstance(names, dict):  # older formats: {"motors": [...]}
#         names = next(iter(names.values()))
#     return df.sort_values(["episode_index", "frame_index"]), info["fps"], list(names)


# def robust_z(x):
#     x = np.asarray(x, float)
#     med = np.nanmedian(x)
#     mad = np.nanmedian(np.abs(x - med)) * 1.4826
#     return (x - med) / (mad + 1e-9)


# def smooth_speed(a, jstd, arm, fps, win_s=0.5):
#     """Arm joint speed (scale-normalized). Positions are smoothed first so encoder /
#     hand jitter doesn't look like motion during pauses."""
#     k = max(1, int(win_s * fps))
#     x = a[:, arm] / jstd[arm]
#     xp = np.pad(x, ((k // 2, k - 1 - k // 2), (0, 0)), mode="edge")
#     xs = np.stack([np.convolve(xp[:, j], np.ones(k) / k, mode="valid") for j in range(x.shape[1])], 1)
#     return np.linalg.norm(np.diff(xs, axis=0), axis=1) * fps


# def runs(mask):
#     """Return list of (start, end) index pairs of True runs."""
#     out, start = [], None
#     for i, m in enumerate(mask):
#         if m and start is None:
#             start = i
#         if not m and start is not None:
#             out.append((start, i)); start = None
#     if start is not None:
#         out.append((start, len(mask)))
#     return out


# def resample(x, n=50):
#     t = np.linspace(0, 1, len(x))
#     tn = np.linspace(0, 1, n)
#     return np.stack([np.interp(tn, t, x[:, j]) for j in range(x.shape[1])], 1)


# def gripper_events(g, closed_low, lo=0.35, hi=0.65):
#     """Hysteresis on normalized gripper signal. Returns list of close frames, open frames."""
#     closed = g < lo if closed_low else g > hi
#     opened = g > hi if closed_low else g < lo
#     state = bool(closed[0])
#     closes, opens = [], []
#     for i in range(1, len(g)):
#         if not state and closed[i]:
#             state = True; closes.append(i)
#         elif state and opened[i]:
#             state = False; opens.append(i)
#     return closes, opens


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--root", required=True)
#     ap.add_argument("--manifest", default=None)
#     ap.add_argument("--gripper-closed", choices=["low", "high"], default="low",
#                     help="Is the gripper value LOW or HIGH when closed? Check one episode first.")
#     ap.add_argument("--max-pause", type=float, default=1.0, help="flag mid-episode pauses longer than this (s)")
#     ap.add_argument("--max-lead", type=float, default=2.0, help="flag idle time at episode start longer than this (s)")
#     ap.add_argument("--out", default="qc")
#     args = ap.parse_args()

#     df, fps, names = load(args.root)
#     gi = next((i for i, n in enumerate(names) if "gripper" in n.lower()), len(names) - 1)
#     ri = next((i for i, n in enumerate(names) if "wrist_roll" in n.lower()), None)
#     arm = [i for i in range(len(names)) if i != gi]
#     closed_low = args.gripper_closed == "low"

#     eps = {e: (np.stack(g["action"].to_numpy()).astype(float),
#                np.stack(g["observation.state"].to_numpy()).astype(float),
#                g["timestamp"].to_numpy())
#            for e, g in df.groupby("episode_index")}

#     all_act = np.concatenate([a for a, _, _ in eps.values()])
#     # Per-joint scale = 5-95 percentile range, floored so near-constant joints
#     # (e.g. wrist roll) don't blow up sensor jitter into "motion".
#     jstd = np.maximum(np.percentile(all_act, 95, 0) - np.percentile(all_act, 5, 0), 5.0)
#     g_lo, g_hi = np.percentile(all_act[:, gi], [2, 98])
#     print(f"fps={fps}  episodes={len(eps)}  joints={names}")
#     print(f"gripper '{names[gi]}' range ~ [{g_lo:.1f}, {g_hi:.1f}]  (closed assumed {args.gripper_closed})")
#     speeds = np.concatenate([smooth_speed(a, jstd, arm, fps) for a, _, _ in eps.values()])
#     idle_thr = 0.15 * np.percentile(speeds, 95)

#     rows, carry, approach = [], {}, {}
#     for e, (a, s, ts) in eps.items():
#         n = len(a)
#         idle = smooth_speed(a, jstd, arm, fps) < idle_thr
#         r = []
#         for a_, b in runs(idle):  # merge idle runs split by brief jitter (< 0.3 s)
#             if r and a_ - r[-1][1] < 0.3 * fps:
#                 r[-1] = (r[-1][0], b)
#             else:
#                 r.append((a_, b))
#         lead = (r[0][1] - r[0][0]) / fps if r and r[0][0] == 0 else 0.0
#         mid = [(b - a_) / fps for a_, b in r if a_ > 0 and b < len(idle)]
#         g = (a[:, gi] - g_lo) / (g_hi - g_lo + 1e-9)
#         closes, opens = gripper_events(g, closed_low)
#         grasp = closes[0] if closes else None
#         release = next((o for o in opens if grasp is not None and o > grasp), None)
#         dts = np.diff(ts)
#         row = dict(
#             episode=e, frames=n, duration_s=n / fps,
#             dropped_frames=int((dts > 1.5 / fps).sum()),
#             lead_idle_s=lead, longest_pause_s=max(mid) if mid else 0.0,
#             jerk=float(np.abs(np.diff(a[:, arm] / jstd[arm], 2, axis=0)).mean() * fps ** 2),
#             track_err=float(np.abs((a[:, arm] - s[:, arm]) / jstd[arm]).mean()),
#             n_grasps=len(closes), released=release is not None,
#             start_state=s[0, arm],
#             release_state=s[release, arm] if release is not None else None,
#             roll_at_grasp=s[grasp, ri] if (grasp is not None and ri is not None) else np.nan,
#         )
#         if grasp is not None and release is not None and release - grasp > 5:
#             carry[e] = resample(s[grasp:release, arm] / jstd[arm])
#         if grasp is not None and grasp > 5:
#             approach[e] = resample(s[:grasp, arm] / jstd[arm])
#         rows.append(row)
#     m = pd.DataFrame(rows).set_index("episode")

#     # Consistency vs. the rest of the dataset
#     med_start = np.median(np.stack(m["start_state"]), 0)
#     m["start_dev"] = [np.linalg.norm((x - med_start) / jstd[arm]) for x in m["start_state"]]
#     rel = m["release_state"].dropna()
#     if len(rel):
#         med_rel = np.median(np.stack(rel), 0)
#         m["release_dev"] = [np.linalg.norm((x - med_rel) / jstd[arm]) if x is not None else np.nan
#                             for x in m["release_state"]]
#     if carry:
#         med_c = np.median(np.stack(list(carry.values())), 0)
#         m["carry_dev"] = pd.Series({e: np.sqrt(((c - med_c) ** 2).mean()) for e, c in carry.items()})
#     if args.manifest and approach:
#         man = pd.read_csv(args.manifest).set_index("episode_index")
#         m = m.join(man, how="left")
#         dev = {}
#         for gid, grp in man.groupby("grid_id"):
#             ids = [e for e in grp.index if e in approach]
#             if len(ids) < 2:
#                 continue
#             med_a = np.median(np.stack([approach[e] for e in ids]), 0)
#             for e in ids:
#                 dev[e] = np.sqrt(((approach[e] - med_a) ** 2).mean())
#         m["approach_dev"] = pd.Series(dev)

#     # Flags
#     checks = ["duration_s", "jerk", "track_err",
#               "start_dev", "release_dev", "carry_dev", "approach_dev"]
#     reasons = {e: [] for e in m.index}
#     for c in checks:
#         if c not in m or m[c].notna().sum() < 4:
#             continue
#         z = pd.Series(robust_z(m[c]), index=m.index)
#         for e in m.index[(z > ROBUST_Z) | ((z < -ROBUST_Z) & (c == "duration_s"))]:
#             reasons[e].append(f"{c} (z={z[e]:.1f})" if abs(z[e]) < 10 else f"{c} (z>10)")
#     if m["roll_at_grasp"].notna().sum() >= 4:
#         z = pd.Series(robust_z(m["roll_at_grasp"]), index=m.index)
#         for e in m.index[z.abs() > ROBUST_Z]:
#             reasons[e].append(f"wrist roll at grasp (z={z[e]:.1f})" if abs(z[e]) < 10 else "wrist roll at grasp (z>10)")
#     for e, r in m.iterrows():
#         if r.n_grasps == 0: reasons[e].append("no grasp detected")
#         if r.n_grasps > 1: reasons[e].append(f"{r.n_grasps} grasps (regrasp?)")
#         if not r.released: reasons[e].append("no release detected")
#         if r.dropped_frames: reasons[e].append(f"{r.dropped_frames} timestamp gaps")
#         if r.longest_pause_s > args.max_pause: reasons[e].append(f"mid-episode pause {r.longest_pause_s:.1f}s")
#         if r.lead_idle_s > args.max_lead: reasons[e].append(f"idle start {r.lead_idle_s:.1f}s")
#     m["flags"] = ["; ".join(reasons[e]) for e in m.index]

#     out = m.drop(columns=["start_state", "release_state"])
#     out.to_csv(f"{args.out}_metrics.csv")
#     flagged = out[out["flags"] != ""]
#     print(f"\n{len(flagged)}/{len(out)} episodes flagged -> review these videos:")
#     for e, r in flagged.iterrows():
#         print(f"  ep {e:4d}: {r["flags"]}")

#     try:
#         import matplotlib; matplotlib.use("Agg")
#         import matplotlib.pyplot as plt
#         fig, axs = plt.subplots(len(names), 1, figsize=(9, 2 * len(names)), sharex=True)
#         for e, (a, s, _) in eps.items():
#             t = np.linspace(0, 1, len(a))
#             bad = bool(reasons[e])
#             for j, ax in enumerate(axs):
#                 ax.plot(t, a[:, j], color="tab:red" if bad else "0.6",
#                         lw=1.2 if bad else 0.6, alpha=0.9 if bad else 0.5, zorder=3 if bad else 1)
#         for j, ax in enumerate(axs):
#             ax.set_ylabel(names[j], fontsize=8)
#         axs[-1].set_xlabel("normalized episode time (red = flagged)")
#         fig.tight_layout(); fig.savefig(f"{args.out}_overlay.png", dpi=120)
#         print(f"\nSaved {args.out}_metrics.csv and {args.out}_overlay.png")
#     except ImportError:
#         print(f"\nSaved {args.out}_metrics.csv (install matplotlib for the overlay plot)")


# if __name__ == "__main__":
#     main()