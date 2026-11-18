#!/usr/bin/env python3
"""
Task 1 evaluation runner: 25 cube positions x 2 policies = 50 rollouts.

What it does, per trial:
  1. Tells you which cube position (C01..C25) and which policy is next.
  2. Waits for you to place the cube and press Enter.
  3. Launches lerobot-rollout for that policy and starts the clock the moment
     the policy's control loop starts (read from the rollout log).
  4. You press Enter the instant the task is complete (cube in tray, gripper
     open, arm lifted) -> success + completion time.
     Type f + Enter for a failure, a + Enter to abort for safety.
     If nothing is pressed within TIMEOUT_S, it is a timeout failure.
  5. Stops the rollout, asks how far it got (for the partial score), and
     appends one row to results.csv immediately.

It is resume-safe: rerun the script and it skips trials already in results.csv.
If a rollout crashes before the policy starts (camera/servo error), that is
not counted as a policy attempt; the trial is offered again.

Outputs (in OUT_DIR):
  schedule.csv   randomized run order (made once, never changed)
  results.csv    one row per trial  -> input for analyze_eval.py / STEP
  logs/          full lerobot-rollout output for every trial
"""
import csv
import os
import random
import select
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# ============================ EDIT THESE ============================
POLICIES = {
    # name used in the logs : checkpoint + execution horizon
    "A": {"path": "shilpitha-03/act_cube_chunkXX", "n_action_steps": 30},
    "B": {"path": "shilpitha-03/act_cube_chunkYY", "n_action_steps": 30},
}
N_CONFIGS = 25            # cube positions C01..C25 (= n_max for STEP)
SEED = 0                  # fixes the random order; do not change after starting
TIMEOUT_S = 60            # seconds of policy execution before it counts as a failure

FOLLOWER_PORT = "/dev/ttyACM0"
ROBOT_ID = "so101_follower_vasco"
FRONT_CAM = "/dev/video4"     # first node under the 720P camera
WRIST_CAM = "/dev/video6"     # first node under the 1080p camera
TASK = "Pick up the cube and place it in the box"

# Extra rollout flags, e.g. to record each trial as an episode once you have
# checked `lerobot-rollout --help` for the episodic-strategy flags.
EXTRA_ARGS = []

OUT_DIR = Path("eval_task1")
DRY_RUN = False           # True = fake rollouts, to rehearse the workflow without the robot
# ====================================================================

START_MARKERS = ("control loop started", "starting rollout")
STAGES = ["reach", "grasp", "transport", "place", "release"]
FIELDS = ["trial", "config", "policy", "policy_path", "n_action_steps", "success",
          "partial", "time_s", "end_reason", "failure_stage", "start_time",
          "log_file", "notes"]


def make_schedule(path):
    rng = random.Random(SEED)
    configs = [f"C{i:02d}" for i in range(1, N_CONFIGS + 1)]
    rng.shuffle(configs)
    rows, t = [], 1
    for c in configs:
        for p in rng.sample(sorted(POLICIES), len(POLICIES)):
            rows.append({"trial": t, "config": c, "policy": p})
            t += 1
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trial", "config", "policy"])
        w.writeheader()
        w.writerows(rows)
    return rows


def load_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_cmd(policy):
    p = POLICIES[policy]
    if DRY_RUN:
        return [sys.executable, "-c",
                "import time;print('Base strategy control loop started',flush=True);time.sleep(600)"]
    cams = (
        "{ front: {type: opencv, index_or_path: %s, width: 480, height: 640, fps: 30, "
        "fourcc: MJPG, rotation: 90}, wrist: {type: opencv, index_or_path: %s, "
        "width: 640, height: 480, fps: 30, fourcc: MJPG}}" % (FRONT_CAM, WRIST_CAM)
    )
    return [
        "lerobot-rollout",
        "--strategy.type=base",
        f"--policy.path={p['path']}",
        f"--policy.n_action_steps={p['n_action_steps']}",
        "--policy.device=cuda",
        "--robot.type=so101_follower",
        f"--robot.port={FOLLOWER_PORT}",
        f"--robot.id={ROBOT_ID}",
        f"--robot.cameras={cams}",
        f"--task={TASK}",
        f"--duration={TIMEOUT_S + 30}",
    ] + EXTRA_ARGS


def stop_process(proc):
    """Ctrl+C first (lets LeRobot disconnect cleanly), then escalate."""
    if proc.poll() is not None:
        return
    for sig, wait in ((signal.SIGINT, 10), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def read_line_nonblocking(timeout):
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.readline().strip().lower() if r else None


def ask_int(prompt, lo, hi):
    while True:
        s = input(prompt).strip()
        if s.isdigit() and lo <= int(s) <= hi:
            return int(s)
        print(f"  enter a number {lo}-{hi}")


def run_trial(row, log_dir):
    policy = row["policy"]
    log_file = log_dir / f"trial{int(row['trial']):02d}_{row['config']}_{policy}.log"
    proc = subprocess.Popen(build_cmd(policy), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            start_new_session=True)
    started = threading.Event()

    def pump():
        with open(log_file, "w") as lf:
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                if not started.is_set() and any(m in line.lower() for m in START_MARKERS):
                    started.set()

    threading.Thread(target=pump, daemon=True).start()

    print("  loading policy and connecting robot ...")
    while not started.is_set():
        if proc.poll() is not None:
            print(f"  !! rollout exited before the policy started. See {log_file}")
            return None, log_file
        time.sleep(0.1)

    t0 = time.monotonic()
    start_iso = datetime.now().isoformat(timespec="seconds")
    print("  >>> POLICY RUNNING.  Enter = success the moment it is complete | "
          "f + Enter = failure | a + Enter = abort")
    end_reason, elapsed = None, None
    while end_reason is None:
        elapsed = time.monotonic() - t0
        if elapsed >= TIMEOUT_S:
            end_reason = "timeout"
            break
        if proc.poll() is not None:
            end_reason = "rollout_exited"
            break
        key = read_line_nonblocking(0.1)
        if key is None:
            continue
        if key == "":
            end_reason = "success"
        elif key.startswith("f"):
            end_reason = "operator_fail"
        elif key.startswith("a"):
            end_reason = "abort"
    elapsed = round(time.monotonic() - t0 if end_reason != "timeout" else TIMEOUT_S, 1)
    print(f"  stopping rollout ({end_reason}, {elapsed}s) - support the arm, torque may release ...")
    stop_process(proc)
    return {"end_reason": end_reason, "time_s": elapsed, "start_time": start_iso}, log_file


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = OUT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    sched_path, res_path = OUT_DIR / "schedule.csv", OUT_DIR / "results.csv"

    schedule = load_csv(sched_path) or make_schedule(sched_path)
    done = {int(r["trial"]) for r in load_csv(res_path)}
    new_file = not res_path.exists()
    print(f"Schedule: {len(schedule)} trials, {len(done)} already done. Results -> {res_path}")
    if DRY_RUN:
        print("DRY RUN: no robot is used.")

    for row in schedule:
        t = int(row["trial"])
        if t in done:
            continue
        print("\n" + "=" * 70)
        print(f"Trial {t}/{len(schedule)}   cube position {row['config']}   policy {row['policy']}")
        print("=" * 70)
        input("Place the cube on the mark, arm at rest, then press Enter to start ...")

        while True:
            outcome, log_file = run_trial(row, log_dir)
            if outcome is not None:
                break
            if input("  Infrastructure failure (not counted). Retry this trial? [Y/n] ").strip().lower() == "n":
                print("Stopping. Rerun the script to continue from this trial.")
                return

        success = outcome["end_reason"] == "success"
        if success:
            partial, failure_stage = 1.0, ""
        else:
            print("  How far did it get?  0 nothing  1 reached cube  2 grasped  "
                  "3 carried to tray  4 placed but no release/lift")
            n = ask_int("  stage reached [0-4]: ", 0, 4)
            partial, failure_stage = n * 0.25, STAGES[n]
        notes = input("  notes (optional): ").strip()

        rec = {
            "trial": t, "config": row["config"], "policy": row["policy"],
            "policy_path": POLICIES[row["policy"]]["path"],
            "n_action_steps": POLICIES[row["policy"]]["n_action_steps"],
            "success": int(success), "partial": partial, "time_s": outcome["time_s"],
            "end_reason": outcome["end_reason"], "failure_stage": failure_stage,
            "start_time": outcome["start_time"], "log_file": str(log_file), "notes": notes,
        }
        with open(res_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                w.writeheader()
                new_file = False
            w.writerow(rec)
        print(f"  saved: {'SUCCESS' if success else 'FAIL'}  partial={partial}  time={outcome['time_s']}s")

    print("\nAll trials done. Run: python analyze_eval.py")


if __name__ == "__main__":
    main()