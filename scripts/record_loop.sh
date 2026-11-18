#!/usr/bin/env bash
# record_loop.sh - overlay (place cube) -> lerobot-record (1 episode) -> keep/discard -> repeat.
#
#   conda activate lerobot
#   ./record_loop.sh 16        # stop once 16 GOOD episodes are kept (round 1)
#   ./record_loop.sh           # continue until every schedule slot has a good episode
#
# Bad episode? Finish it with the RIGHT ARROW as usual, then answer "d" (discard) at the
# keep/discard prompt. The overlay then shows the SAME target again.
# Kept one by mistake? Change its keep value from 1 to 0 in demos.csv; the loop returns
# to that position next (lowest unfilled slot first). Do NOT use the left
# arrow (LeRobot's built-in re-record): the overlay can't open while lerobot-record runs.
#
# Discarded episodes stay in the dataset (the assignment wants a log of every demo) and are
# marked keep=0 in demos.csv. Training uses only keep=1 episodes (--dataset.episodes).

set -u

# ---------------- edit these ----------------
REPO="shilpitha-03/so101_cube_grid"
TASK="Pick up the cube and place it in the tray"
FRONT="/dev/video4"                           # replace with /dev/v4l/by-id/...-video-index0
WRIST="/dev/video6"                           # replace with /dev/v4l/by-id/...-video-index0
FOLLOWER_PORT="/dev/ttyACM0"
LEADER_PORT="/dev/ttyACM1"
EPISODE_TIME_S=60                             # cap; end each episode with the RIGHT ARROW key
DISPLAY_DATA=true
OVERLAY_PY="$HOME/overlay-env/bin/python"
SCHEDULE="schedule.csv"
CALIB="overlay_calib.json"
LOG="demos.csv"                               # episode_index,slot,grid_id,round,keep,note
# --------------------------------------------

ROOT="$HOME/.cache/huggingface/lerobot/$REPO"
TOTAL=$(( $(wc -l < "$SCHEDULE") - 1 ))
STOP_AT="${1:-$TOTAL}"

if [ -f "$LOG" ]; then python3 loop_state.py migrate "$LOG"; else echo "episode_index,slot,grid_id,round,keep,note" > "$LOG"; fi

count_episodes() { python3 -c "import json; print(json.load(open('$ROOT/meta/info.json'))['total_episodes'])" 2>/dev/null || echo 0; }
count_frames()   { python3 -c "import json; print(json.load(open('$ROOT/meta/info.json'))['total_frames'])" 2>/dev/null || echo 0; }
logged_count()   { awk 'NR > 1' "$LOG" | wc -l; }

# Dataset and log must agree before we start, or the episode -> position mapping is lost.
if [ "$(count_episodes)" -ne "$(logged_count)" ]; then
  echo "!! Dataset has $(count_episodes) episodes but $LOG lists $(logged_count)."
  echo "   Delete both and start fresh, or fix $LOG by hand before recording more."
  exit 1
fi

while true; do
  read -r SLOT TARGET ROUND < <(python3 loop_state.py next "$LOG" "$SCHEDULE" "$STOP_AT")
  K=$(python3 loop_state.py filled "$LOG")
  if [ "$SLOT" -lt 0 ]; then
    echo "All $STOP_AT slots have a kept episode ($K kept). Dataset total: $(count_episodes). Done."
    break
  fi
  N=$(count_episodes)

  echo
  echo "=== Slot $SLOT ($K/$STOP_AT kept)  (dataset episode $N)  ->  place cube at $TARGET  (q in overlay when placed) ==="
  if ! "$OVERLAY_PY" grid_overlay.py live --cam "$FRONT" --calib "$CALIB" --schedule "$SCHEDULE" --index "$SLOT"; then
    echo "!! Overlay failed to start (see error above). Stopping before recording."
    exit 1
  fi

  read -r -p "Arm at home, cube on $TARGET? [Enter] record   [s] stop: " ans
  [ "$ans" = "s" ] && break

  if [ -d "$ROOT" ]; then RESUME=true; else RESUME=false; fi
  F0=$(count_frames)

  lerobot-record \
    --robot.type=so101_follower \
    --robot.port="$FOLLOWER_PORT" \
    --robot.id=so101_follower_vasco \
    --robot.cameras="{ front: {type: opencv, index_or_path: $FRONT, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
    --teleop.type=so101_leader \
    --teleop.port="$LEADER_PORT" \
    --teleop.id=so101_leader_vasco \
    --display_data="$DISPLAY_DATA" \
    --dataset.repo_id="$REPO" \
    --dataset.root="$ROOT" \
    --dataset.single_task="$TASK" \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s="$EPISODE_TIME_S" \
    --dataset.reset_time_s=5 \
    --dataset.push_to_hub=false \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --dataset.no_stamp=true \
    --resume="$RESUME"
  RC=$?
  if [ "$RC" -ne 0 ]; then
    echo "!! lerobot-record exited with an error (code $RC). Stopping the loop - see the traceback above."
    exit 1
  fi

  NEW=$(count_episodes)
  if [ "$NEW" -le "$N" ]; then
    echo "!! No new episode saved. Same target will be shown again."
    continue
  fi
  F1=$(count_frames)
  echo ">> Saved dataset episode $N at $TARGET: $((F1 - F0)) frames (~$(( (F1 - F0) / 30 )) s)"
  read -r -p "Keep it? [Enter] keep   [d] discard: " k
  case "$k" in
    d*|D*)
      read -r -p "Short reason (optional): " note
      echo "$N,$SLOT,$TARGET,$ROUND,0,${note//,/;}" >> "$LOG"
      echo "   DISCARDED episode $N. $TARGET will be shown again." ;;
    *)
      echo "$N,$SLOT,$TARGET,$ROUND,1," >> "$LOG"
      echo "   KEPT episode $N at $TARGET." ;;
  esac
done
















# #!/usr/bin/env bash
# # record_loop.sh - alternate: overlay (place cube) -> lerobot-record (1 episode) -> repeat.
# #
# # Run from the folder containing grid_overlay.py, schedule.csv, overlay_calib.json,
# # inside the lerobot conda env:
# #   conda activate lerobot
# #   ./record_loop.sh 16        # stop once the dataset has 16 episodes (round 1)
# #   ./record_loop.sh           # continue until every schedule row is recorded
# #
# # The next target is always schedule row = number of episodes already in the dataset,
# # so a crash or a run that saved nothing never misaligns schedule and dataset.

# set -u

# # ---------------- edit these ----------------
# REPO="shilpitha-03/so101_cube_grid"          # use shilpitha-03/so101_cube_grid_test for the 2-episode test
# TASK="Pick up the cube and place it in the tray"
# FRONT="/dev/video4"                           # replace with /dev/v4l/by-id/...-video-index0
# WRIST="/dev/video6"                           # replace with /dev/v4l/by-id/...-video-index0
# FOLLOWER_PORT="/dev/ttyACM0"
# LEADER_PORT="/dev/ttyACM1"
# EPISODE_TIME_S=60                             # cap; end each episode with the RIGHT ARROW key
# DISPLAY_DATA=true                             # Rerun viewer with live camera feeds during recording
# OVERLAY_PY="$HOME/overlay-env/bin/python"
# SCHEDULE="schedule.csv"
# CALIB="overlay_calib.json"
# # --------------------------------------------

# ROOT="$HOME/.cache/huggingface/lerobot/$REPO"

# # ---- pre-flight checks: stop loudly instead of silently skipping the overlay ----
# for f in grid_overlay.py "$SCHEDULE" "$CALIB"; do
#   [ -f "$f" ] || { echo "ERROR: $f not found in $(pwd). Run this script from the folder that has it."; exit 1; }
# done
# [ -x "$OVERLAY_PY" ] || { echo "ERROR: overlay python not found at $OVERLAY_PY"; exit 1; }
# [ -e "$FRONT" ] || { echo "ERROR: front camera $FRONT does not exist (camera numbers may have changed: run v4l2-ctl --list-devices)"; exit 1; }
# command -v lerobot-record >/dev/null || { echo "ERROR: lerobot-record not found. Run: conda activate lerobot"; exit 1; }
# TOTAL=$(( $(wc -l < "$SCHEDULE") - 1 ))
# STOP_AT="${1:-$TOTAL}"
# echo "Pre-flight OK: dataset=$REPO  front=$FRONT  wrist=$WRIST  target=$STOP_AT episodes"

# count_episodes() {
#   python3 -c "import json; print(json.load(open('$ROOT/meta/info.json'))['total_episodes'])" 2>/dev/null || echo 0
# }
# count_frames() {
#   python3 -c "import json; print(json.load(open('$ROOT/meta/info.json'))['total_frames'])" 2>/dev/null || echo 0
# }

# while true; do
#   N=$(count_episodes)
#   if [ "$N" -ge "$STOP_AT" ]; then
#     echo "Dataset has $N episodes (target $STOP_AT). Done."
#     break
#   fi
#   TARGET=$(awk -F, -v n="$N" 'NR == n + 2 {print $2}' "$SCHEDULE")
#   echo
#   echo "=== Episode $N  ->  place cube at $TARGET  (press q in the overlay when placed) ==="
#   if ! "$OVERLAY_PY" grid_overlay.py live --cam "$FRONT" --calib "$CALIB" --schedule "$SCHEDULE" --index "$N"; then
#     echo "ERROR: overlay failed (see message above). Not recording."
#     exit 1
#   fi

#   read -r -p "Arm at home, cube on $TARGET? [Enter] record   [s] stop: " ans
#   [ "$ans" = "s" ] && break

#   if [ -d "$ROOT" ]; then RESUME=true; else RESUME=false; fi
#   F0=$(count_frames)

#   lerobot-record \
#     --robot.type=so101_follower \
#     --robot.port="$FOLLOWER_PORT" \
#     --robot.id=so101_follower_vasco \
#     --robot.cameras="{ front: {type: opencv, index_or_path: $FRONT, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
#     --teleop.type=so101_leader \
#     --teleop.port="$LEADER_PORT" \
#     --teleop.id=so101_leader_vasco \
#     --display_data="$DISPLAY_DATA" \
#     --dataset.repo_id="$REPO" \
#     --dataset.root="$ROOT" \
#     --dataset.single_task="$TASK" \
#     --dataset.num_episodes=1 \
#     --dataset.episode_time_s="$EPISODE_TIME_S" \
#     --dataset.reset_time_s=20 \
#     --dataset.push_to_hub=false \
#     --dataset.streaming_encoding=true \
#     --dataset.encoder_threads=2 \
#     --dataset.no_stamp=true \
#     --resume="$RESUME"
#   RC=$?
#   if [ "$RC" -ne 0 ]; then
#     echo "!! lerobot-record exited with an error (code $RC). Stopping the loop - see the traceback above."
#     exit 1
#   fi

#   NEW=$(count_episodes)
#   if [ "$NEW" -le "$N" ]; then
#     echo "!! No new episode saved (still $NEW). Same target will be shown again."
#   else
#     F1=$(count_frames)
#     echo ">> Saved episode $N: $((F1 - F0)) frames (~$(( (F1 - F0) / 30 )) s at 30 fps)"
#   fi
# done