#!/usr/bin/env bash
# eval_loop.sh - the real evaluation: 25 configs x 2 policies, in the order fixed by eval_schedule.csv.
# For each rollout: overlay (place cube, press q) -> policy runs 60 s and is recorded -> you score it.
# Resumable: rerun any time; it continues with the next rollout without a valid row in eval_log.csv.
#   ./eval_loop.sh
set -u

# ---------------- fixed settings (decided in the pilot) ----------------
ACT_PATH="/home/shilpitha/Downloads/HRL/so101-act-3/outputs/train/act_cube_v3/checkpoints/040000/pretrained_model_te"
DIFF_PATH="./diffusion_v2"
FRONT="/dev/video4"; WRIST="/dev/video6"; FOLLOWER_PORT="/dev/ttyACM0"
EPISODE_TIME_S=60
OVERLAY_PY="$HOME/overlay-env/bin/python"
SCHEDULE="eval_schedule.csv"; LOG="eval_log.csv"; CALIB="overlay_calib.json"
# ------------------------------------------------------------------------

[ -f "$LOG" ] || echo "trial,grid_id,seen,policy,order,dataset,episode_index,score,success,valid,note,timestamp" > "$LOG"
[ -d "$ACT_PATH" ] || { echo "!! ACT checkpoint not found: $ACT_PATH (run ./make_te.sh on 040000 first)"; exit 1; }
[ -d "$DIFF_PATH" ] || { echo "!! Diffusion checkpoint not found: $DIFF_PATH"; exit 1; }

n_eps() { python3 -c "import json;print(json.load(open('$1/meta/info.json'))['total_episodes'])" 2>/dev/null || echo 0; }

while true; do
  read -r TRIAL GRID SEEN POLICY ORDER < <(python3 eval_state.py "$SCHEDULE" "$LOG")
  if [ "$TRIAL" = "DONE" ]; then echo "All 50 rollouts logged. Run: python analyze_eval.py"; break; fi
  if [ "$POLICY" = "act" ]; then PPATH="$ACT_PATH"; NAME="act40k_te"; else PPATH="$DIFF_PATH"; NAME="diff_v2"; fi
  REPO="shilpitha-03/rollout_eval_$NAME"; ROOT="$HOME/.cache/huggingface/lerobot/$REPO"

  echo
  echo "=== Trial $TRIAL/25  position $GRID  policy $POLICY ($ORDER of 2)  - place cube, press q ==="
  printf 'episode_index,grid_id,round\n%s,%s,0\n' "$TRIAL" "$GRID" > .eval_target.csv
  "$OVERLAY_PY" grid_overlay.py live --cam "$FRONT" --calib "$CALIB" --schedule .eval_target.csv --index 0 || exit 1
  read -r -p "Cube on $GRID, arm at home? [Enter] run   [s] stop: " a; [ "$a" = "s" ] && break

  if [ -d "$ROOT" ]; then RESUME=true; else RESUME=false; fi
  N0=$(n_eps "$ROOT")
  lerobot-rollout \
    --strategy.type=episodic --policy.path="$PPATH" --device=cuda \
    --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id=so101_follower_vasco \
    --robot.cameras="{ front: {type: opencv, index_or_path: $FRONT, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
    --task="Pick up the cube and place it in the tray" --display_data=true \
    --dataset.repo_id="$REPO" --dataset.root="$ROOT" \
    --dataset.single_task="Pick up the cube and place it in the tray" \
    --dataset.num_episodes=1 --dataset.episode_time_s="$EPISODE_TIME_S" --dataset.reset_time_s=5 \
    --dataset.push_to_hub=false --resume="$RESUME"
  N1=$(n_eps "$ROOT")
  EP=$([ "$N1" -gt "$N0" ] && echo $((N1 - 1)) || echo "NA")

  echo "Score: 0=never reached  0.25=reached  0.5=grasped  0.75=carried to tray  1=placed+released+lifted"
  echo "       x = INVALID run (setup/operator error, e.g. cube misplaced, crash) -> same rollout is repeated"
  while true; do read -r -p "Score for trial $TRIAL ($POLICY at $GRID): " S
    case "$S" in 0|0.25|0.5|0.75|1|x) break ;; *) echo "enter 0, 0.25, 0.5, 0.75, 1 or x" ;; esac; done
  read -r -p "Short note (where/how it failed; no commas): " NOTE
  if [ "$S" = "x" ]; then VALID=0; S=""; SUCC=""; else VALID=1; SUCC=$([ "$S" = "1" ] && echo 1 || echo 0); fi
  echo "$TRIAL,$GRID,$SEEN,$POLICY,$ORDER,$REPO,$EP,$S,$SUCC,$VALID,${NOTE//,/;},$(date -Iseconds)" >> "$LOG"
  echo ">> logged trial $TRIAL $POLICY: score=${S:-invalid} (dataset episode $EP)"
done