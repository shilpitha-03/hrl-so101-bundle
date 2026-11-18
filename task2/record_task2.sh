#!/usr/bin/env bash
# Task 2 block-stacking data collection loop (wrist camera only).
# One lerobot-record call per episode. Shows the next configuration from the
# schedule, records, then asks keep/discard and logs it. Safe to stop and restart:
# it resumes from the log.
#
# Usage:   bash record_task2.sh
# Stop:    answer "s" at the prompt (or Ctrl+C between episodes)

# ======================= EDIT THESE (verify before running) =======================
FOLLOWER_PORT=/dev/ttyACM1          # lerobot-find-port if unsure (these have swapped before)
LEADER_PORT=/dev/ttyACM0
FOLLOWER_ID=so101_follower_vasco
LEADER_ID=so101_leader_vasco
WRIST=/dev/video4                   # check with: v4l2-ctl --list-devices
HF_USER=shilpitha-03
REPO="$HF_USER/so101_stack_rpy"
ROOT="$HOME/.cache/huggingface/lerobot/$REPO"
TASK="Stack the red, yellow and purple cubes on the table mark"
SCHEDULE="./task2_collection_schedule_v3.csv"
LOG="./task2_demos_log.csv"
EPISODE_TIME_S=100                  # hard cap; press RIGHT ARROW to end once arm is home
DISPLAY_DATA=true                   # rerun viewer of the wrist feed
PILOT_N=5                           # pause after this many kept episodes for the replay check
# ==================================================================================

set -u
command -v lerobot-record >/dev/null || { echo "!! lerobot-record not found. Activate the env: conda activate lerobot"; exit 1; }
[ -f "$SCHEDULE" ] || { echo "!! Schedule not found: $SCHEDULE"; exit 1; }
for d in "$FOLLOWER_PORT" "$LEADER_PORT" "$WRIST"; do
  [ -e "$d" ] || { echo "!! $d does not exist. Check ports/camera and edit the top of this script."; exit 1; }
done

meta() {  # meta total_episodes | total_frames
  python3 - "$ROOT/meta/info.json" "$1" <<'PY'
import json,sys
try: print(json.load(open(sys.argv[1]))[sys.argv[2]])
except Exception: print(0)
PY
}

row() {  # row N -> prints shell assignments for schedule row N (0-based), or END
  python3 - "$SCHEDULE" "$1" <<'PY'
import csv,sys,shlex
rows=list(csv.DictReader(open(sys.argv[1])))
n=int(sys.argv[2])
if n>=len(rows): print("END"); sys.exit()
r=rows[n]
prev=rows[n-1]["lighting"] if n>0 else ""
keys=["episode_index","type","lighting","T1","T2","T3","placement","pre_stacked","pre_stack_offset"]
for k in keys: print(f"S_{k}={shlex.quote(r[k])}")
print(f"S_prevlight={shlex.quote(prev)}")
print(f"S_total={len(rows)}")
PY
}

[ -f "$LOG" ] || echo "dataset_episode,schedule_row,type,lighting,T1,T2,T3,placement,pre_stacked,pre_stack_offset,kept,frames,seconds,note,timestamp" > "$LOG"

if [ -d "$ROOT" ] && [ "$(meta total_episodes)" = "0" ]; then
  echo "!! $ROOT exists but has no saved episodes (a failed earlier start)."
  read -r -p "Delete it and start fresh? [y/N]: " a || exit 0
  [ "$a" = "y" ] && rm -rf "$ROOT" || exit 1
fi

# Sanity: dataset episode count must match the log
N_DS=$(meta total_episodes)
N_LOG=$(( $(wc -l < "$LOG") - 1 ))
if [ "$N_DS" -ne "$N_LOG" ]; then
  echo "!! Dataset has $N_DS episodes but the log has $N_LOG rows. Fix this before continuing"
  echo "   (an episode was saved without being logged, or the log belongs to another dataset)."
  exit 1
fi

flush_tty() { python3 -c "import sys,termios; termios.tcflush(sys.stdin, termios.TCIFLUSH)" 2>/dev/null; }

LIGHT_DESC() { case "$1" in
  L1) echo "L1 = enclosure lamp NORMAL";;
  L2) echo "L2 = lamp DIMMED/moved, room lights OFF";;
  L3) echo "L3 = lamp normal + EXTRA SIDE LIGHT from open side";;
  *) echo "$1";; esac; }

while true; do
  KEPT=$(awk -F, 'NR>1 && $11==1' "$LOG" | wc -l)
  NEXT=$(awk -F, 'NR>1 && $11==1 {k[$2]=1} END{i=0; while(i in k) i++; print i}' "$LOG")  # first schedule row with no kept episode
  OUT=$(row "$NEXT")
  [ "$OUT" = "END" ] && { echo; echo "== Schedule complete: $KEPT kept episodes. =="; break; }
  eval "$OUT"
  N=$(meta total_episodes)

  clear
  echo "================================================================"
  echo "  Schedule row $NEXT / $((S_total-1))      next dataset episode: $N"
  echo "================================================================"
  if [ "$S_lighting" != "$S_prevlight" ]; then
    echo "  >>> LIGHTING CHANGE: $(LIGHT_DESC "$S_lighting") <<<"
  else
    echo "  Lighting: $(LIGHT_DESC "$S_lighting")"
  fi
  echo
  echo "  Type: $S_type"
  printf "  Tray:   T1 = %-8s  T2 = %-8s  T3 = %-8s\n" "$S_T1" "$S_T2" "$S_T3"
  echo "  Placement: $S_placement"
  if [ "$S_type" != "full" ]; then
    echo "  PRE-STACK on table mark: $S_pre_stacked  ($S_pre_stack_offset)"
  fi
  echo
  echo "  Pattern: start -> LOOK -> red -> LOOK -> yellow -> LOOK -> purple -> LOOK -> home"
  echo "  (pre-stacked: start -> LOOK -> continue from current stack)"
  echo "  Look = brief 0.5-1 s pass. Lift straight up 3-4 cm after each release."
  echo "  When the arm is back at start pose: press RIGHT ARROW (keep this terminal focused)."
  echo "----------------------------------------------------------------"
  read -r -p "  Scene set, leader + follower at START POSE? [Enter] record   [s] stop: " ans || exit 0
  [ "$ans" = "s" ] && break

  if [ -d "$ROOT" ]; then RESUME=true; else RESUME=false; fi
  F0=$(meta total_frames)

  lerobot-record \
    --robot.type=so101_follower \
    --robot.port="$FOLLOWER_PORT" \
    --robot.id="$FOLLOWER_ID" \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: $WRIST, width: 480, height: 640, fps: 30, fourcc: MJPG, rotation: -90}}" \
    --teleop.type=so101_leader \
    --teleop.port="$LEADER_PORT" \
    --teleop.id="$LEADER_ID" \
    --display_data="$DISPLAY_DATA" \
    --dataset.repo_id="$REPO" \
    --dataset.root="$ROOT" \
    --dataset.single_task="$TASK" \
    --dataset.num_episodes=1 \
    --dataset.fps=30 \
    --dataset.episode_time_s="$EPISODE_TIME_S" \
    --dataset.reset_time_s=0 \
    --dataset.push_to_hub=false \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --dataset.no_stamp=true \
    --resume="$RESUME"
  RC=$?
  sleep 0.3; flush_tty   # drop arrow-key bytes typed during recording
  if [ "$RC" -ne 0 ]; then
    echo "!! lerobot-record exited with an error (code $RC). Stopping - see the traceback above."
    exit 1
  fi

  NEW=$(meta total_episodes)
  if [ "$NEW" -le "$N" ]; then
    echo "!! No new episode saved. Same configuration will be shown again."
    read -r -p "   [Enter] to continue" _ || exit 0; continue
  fi
  F1=$(meta total_frames); FR=$((F1 - F0)); SEC=$((FR / 30))
  echo
  echo ">> Saved dataset episode $N: $FR frames (~${SEC} s)"
  if [ "$S_type" = "full" ] && [ "$SEC" -lt 40 ]; then
    echo "   WARNING: only ${SEC} s for a full stack - ended early by accident?"
  fi
  if [ "$SEC" -ge $((EPISODE_TIME_S - 1)) ]; then
    echo "   WARNING: hit the ${EPISODE_TIME_S} s cap - did it finish and return home?"
  fi
  echo "   Discard if: stack fell, regrasp/fumble, pause > 1 s, look pose held too long, didn't return home."
  while true; do
    read -r -p "Keep it? [k] keep   [d] discard: " k || exit 0
    k=$(printf '%s' "$k" | tr -cd '[:alpha:]' | tr 'A-Z' 'a-z')
    [ "$k" = "k" ] || [ "$k" = "d" ] && break
    echo "   Type exactly k or d."
  done
  TS=$(date +%F_%T)
  COMMON="$N,$S_episode_index,$S_type,$S_lighting,$S_T1,$S_T2,$S_T3,$S_placement,$S_pre_stacked,$S_pre_stack_offset"
  if [[ "$k" == *d || "$k" == *D ]]; then
    echo "   >> DISCARDING"
    read -r -p "Short reason: " note || exit 0
    echo "$COMMON,0,$FR,$SEC,${note//,/;},$TS" >> "$LOG"
    echo "   => DISCARDED episode $N (stays in dataset, excluded from training). Same configuration again."
    sleep 1
  else
    echo "   >> KEEPING"
    read -r -p "Optional note (Enter to skip): " note || exit 0
    echo "$COMMON,1,$FR,$SEC,${note//,/;},$TS" >> "$LOG"
    echo "   => KEPT episode $N"
    KEPT_NOW=$((KEPT + 1))
    if [ "$KEPT_NOW" -eq "$PILOT_N" ]; then
      echo
      echo "== PILOT DONE ($PILOT_N kept). Replay-check one episode before continuing: =="
      echo "lerobot-replay --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$FOLLOWER_ID \\"
      echo "  --dataset.repo_id=$REPO --dataset.root=$ROOT --dataset.episode=0"
      echo "Also scrub the wrist video: at every LOOK are all tray cubes + the stack mark visible,"
      echo "and do red vs purple look clearly different?"
      read -r -p "[Enter] when checked to continue recording, [s] stop: " a || exit 0
      [ "$a" = "s" ] && break
    fi
  fi
done

KEEP_LIST=$(awk -F, 'NR>1 && $11==1 {printf "%s%s", sep, $1; sep=","}' "$LOG")
echo
echo "Kept dataset episodes (use for training):"
echo "  --dataset.episodes=[$KEEP_LIST]"
echo
echo "When finished, push to the Hub:"
echo "  python -c \"from lerobot.datasets.lerobot_dataset import LeRobotDataset; LeRobotDataset('$REPO', root='$ROOT').push_to_hub()\""
echo "  huggingface-cli upload $REPO $LOG task2_demos_log.csv --repo-type dataset"















# #!/usr/bin/env bash
# # Task 2 block-stacking data collection loop (wrist camera only).
# # One lerobot-record call per episode. Shows the next configuration from the
# # schedule, records, then asks keep/discard and logs it. Safe to stop and restart:
# # it resumes from the log.
# #
# # Usage:   bash record_task2.sh
# # Stop:    answer "s" at the prompt (or Ctrl+C between episodes)

# # ======================= EDIT THESE (verify before running) =======================
# FOLLOWER_PORT=/dev/ttyACM1          # lerobot-find-port if unsure (these have swapped before)
# LEADER_PORT=/dev/ttyACM0
# FOLLOWER_ID=so101_follower_vasco
# LEADER_ID=so101_leader_vasco
# WRIST=/dev/video4                  # check with: v4l2-ctl --list-devices
# HF_USER=shilpitha-03
# REPO="$HF_USER/so101_stack_rpy"
# ROOT="$HOME/.cache/huggingface/lerobot/$REPO"
# TASK="Stack the red, yellow and purple cubes on the table mark"
# SCHEDULE="./task2_collection_schedule_v3.csv"
# LOG="./task2_demos_log.csv"
# EPISODE_TIME_S=100                  # hard cap; press RIGHT ARROW to end once arm is home
# DISPLAY_DATA=true                   # rerun viewer of the wrist feed
# PILOT_N=5                         # pause after this many kept episodes for the replay check
# # ==================================================================================

# set -u
# command -v lerobot-record >/dev/null || { echo "!! lerobot-record not found. Activate the env: conda activate lerobot"; exit 1; }
# [ -f "$SCHEDULE" ] || { echo "!! Schedule not found: $SCHEDULE"; exit 1; }
# for d in "$FOLLOWER_PORT" "$LEADER_PORT" "$WRIST"; do
#   [ -e "$d" ] || { echo "!! $d does not exist. Check ports/camera and edit the top of this script."; exit 1; }
# done

# meta() {  # meta total_episodes | total_frames
#   python3 - "$ROOT/meta/info.json" "$1" <<'PY'
# import json,sys
# try: print(json.load(open(sys.argv[1]))[sys.argv[2]])
# except Exception: print(0)
# PY
# }

# row() {  # row N -> prints shell assignments for schedule row N (0-based), or END
#   python3 - "$SCHEDULE" "$1" <<'PY'
# import csv,sys,shlex
# rows=list(csv.DictReader(open(sys.argv[1])))
# n=int(sys.argv[2])
# if n>=len(rows): print("END"); sys.exit()
# r=rows[n]
# prev=rows[n-1]["lighting"] if n>0 else ""
# keys=["episode_index","type","lighting","T1","T2","T3","placement","pre_stacked","pre_stack_offset"]
# for k in keys: print(f"S_{k}={shlex.quote(r[k])}")
# print(f"S_prevlight={shlex.quote(prev)}")
# print(f"S_total={len(rows)}")
# PY
# }

# [ -f "$LOG" ] || echo "dataset_episode,schedule_row,type,lighting,T1,T2,T3,placement,pre_stacked,pre_stack_offset,kept,frames,seconds,note,timestamp" > "$LOG"

# if [ -d "$ROOT" ] && [ "$(meta total_episodes)" = "0" ]; then
#   echo "!! $ROOT exists but has no saved episodes (a failed earlier start)."
#   read -r -p "Delete it and start fresh? [y/N]: " a || exit 0
#   [ "$a" = "y" ] && rm -rf "$ROOT" || exit 1
# fi

# # Sanity: dataset episode count must match the log
# N_DS=$(meta total_episodes)
# N_LOG=$(( $(wc -l < "$LOG") - 1 ))
# if [ "$N_DS" -ne "$N_LOG" ]; then
#   echo "!! Dataset has $N_DS episodes but the log has $N_LOG rows. Fix this before continuing"
#   echo "   (an episode was saved without being logged, or the log belongs to another dataset)."
#   exit 1
# fi

# flush_tty() { python3 -c "import sys,termios; termios.tcflush(sys.stdin, termios.TCIFLUSH)" 2>/dev/null; }

# LIGHT_DESC() { case "$1" in
#   L1) echo "L1 = enclosure lamp NORMAL";;
#   L2) echo "L2 = lamp DIMMED/moved, room lights OFF";;
#   L3) echo "L3 = lamp normal + EXTRA SIDE LIGHT from open side";;
#   *) echo "$1";; esac; }

# while true; do
#   KEPT=$(awk -F, 'NR>1 && $11==1' "$LOG" | wc -l)
#   OUT=$(row "$KEPT")
#   [ "$OUT" = "END" ] && { echo; echo "== Schedule complete: $KEPT kept episodes. =="; break; }
#   eval "$OUT"
#   N=$(meta total_episodes)

#   clear
#   echo "================================================================"
#   echo "  Schedule row $KEPT / $((S_total-1))      next dataset episode: $N"
#   echo "================================================================"
#   if [ "$S_lighting" != "$S_prevlight" ]; then
#     echo "  >>> LIGHTING CHANGE: $(LIGHT_DESC "$S_lighting") <<<"
#   else
#     echo "  Lighting: $(LIGHT_DESC "$S_lighting")"
#   fi
#   echo
#   echo "  Type: $S_type"
#   printf "  Tray:   T1 = %-8s  T2 = %-8s  T3 = %-8s\n" "$S_T1" "$S_T2" "$S_T3"
#   echo "  Placement: $S_placement"
#   if [ "$S_type" != "full" ]; then
#     echo "  PRE-STACK on table mark: $S_pre_stacked  ($S_pre_stack_offset)"
#   fi
#   echo
#   echo "  Pattern: start -> LOOK -> red -> LOOK -> yellow -> LOOK -> purple -> LOOK -> home"
#   echo "  (pre-stacked: start -> LOOK -> continue from current stack)"
#   echo "  Look = brief 0.5-1 s pass. Lift straight up 3-4 cm after each release."
#   echo "  When the arm is back at start pose: press RIGHT ARROW (keep this terminal focused)."
#   echo "----------------------------------------------------------------"
#   read -r -p "  Scene set, leader + follower at START POSE? [Enter] record   [s] stop: " ans || exit 0
#   [ "$ans" = "s" ] && break

#   if [ -d "$ROOT" ]; then RESUME=true; else RESUME=false; fi
#   F0=$(meta total_frames)

#   lerobot-record \
#     --robot.type=so101_follower \
#     --robot.port="$FOLLOWER_PORT" \
#     --robot.id="$FOLLOWER_ID" \
#     --robot.cameras="{ wrist: {type: opencv, index_or_path: $WRIST, width: 480, height: 640, fps: 30, fourcc: MJPG, rotation: -90}}" \
#     --teleop.type=so101_leader \
#     --teleop.port="$LEADER_PORT" \
#     --teleop.id="$LEADER_ID" \
#     --display_data="$DISPLAY_DATA" \
#     --dataset.repo_id="$REPO" \
#     --dataset.root="$ROOT" \
#     --dataset.single_task="$TASK" \
#     --dataset.num_episodes=1 \
#     --dataset.fps=30 \
#     --dataset.episode_time_s="$EPISODE_TIME_S" \
#     --dataset.reset_time_s=0 \
#     --dataset.push_to_hub=false \
#     --dataset.streaming_encoding=true \
#     --dataset.encoder_threads=2 \
#     --dataset.no_stamp=true \
#     --resume="$RESUME"
#   RC=$?
#   sleep 0.3; flush_tty   # drop arrow-key bytes typed during recording
#   if [ "$RC" -ne 0 ]; then
#     echo "!! lerobot-record exited with an error (code $RC). Stopping - see the traceback above."
#     exit 1
#   fi

#   NEW=$(meta total_episodes)
#   if [ "$NEW" -le "$N" ]; then
#     echo "!! No new episode saved. Same configuration will be shown again."
#     read -r -p "   [Enter] to continue" _ || exit 0; continue
#   fi
#   F1=$(meta total_frames); FR=$((F1 - F0)); SEC=$((FR / 30))
#   echo
#   echo ">> Saved dataset episode $N: $FR frames (~${SEC} s)"
#   if [ "$S_type" = "full" ] && [ "$SEC" -lt 40 ]; then
#     echo "   WARNING: only ${SEC} s for a full stack - ended early by accident?"
#   fi
#   if [ "$SEC" -ge $((EPISODE_TIME_S - 1)) ]; then
#     echo "   WARNING: hit the ${EPISODE_TIME_S} s cap - did it finish and return home?"
#   fi
#   echo "   Discard if: stack fell, regrasp/fumble, pause > 1 s, look pose held too long, didn't return home."
#   while true; do
#     read -r -p "Keep it? [k] keep   [d] discard: " k || exit 0
#     k=$(printf '%s' "$k" | tr -cd '[:alpha:]' | tr 'A-Z' 'a-z')
#     [ "$k" = "k" ] || [ "$k" = "d" ] && break
#     echo "   Type exactly k or d."
#   done
#   TS=$(date +%F_%T)
#   COMMON="$N,$S_episode_index,$S_type,$S_lighting,$S_T1,$S_T2,$S_T3,$S_placement,$S_pre_stacked,$S_pre_stack_offset"
#   if [[ "$k" == *d || "$k" == *D ]]; then
#     echo "   >> DISCARDING"
#     read -r -p "Short reason: " note || exit 0
#     echo "$COMMON,0,$FR,$SEC,${note//,/;},$TS" >> "$LOG"
#     echo "   => DISCARDED episode $N (stays in dataset, excluded from training). Same configuration again."
#     sleep 1
#   else
#     echo "   >> KEEPING"
#     read -r -p "Optional note (Enter to skip): " note || exit 0
#     echo "$COMMON,1,$FR,$SEC,${note//,/;},$TS" >> "$LOG"
#     echo "   => KEPT episode $N"
#     KEPT_NOW=$((KEPT + 1))
#     if [ "$KEPT_NOW" -eq "$PILOT_N" ]; then
#       echo
#       echo "== PILOT DONE ($PILOT_N kept). Replay-check one episode before continuing: =="
#       echo "lerobot-replay --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$FOLLOWER_ID \\"
#       echo "  --dataset.repo_id=$REPO --dataset.root=$ROOT --dataset.episode=0"
#       echo "Also scrub the wrist video: at every LOOK are all tray cubes + the stack mark visible,"
#       echo "and do red vs purple look clearly different?"
#       read -r -p "[Enter] when checked to continue recording, [s] stop: " a || exit 0
#       [ "$a" = "s" ] && break
#     fi
#   fi
# done

# KEEP_LIST=$(awk -F, 'NR>1 && $11==1 {printf "%s%s", sep, $1; sep=","}' "$LOG")
# echo
# echo "Kept dataset episodes (use for training):"
# echo "  --dataset.episodes=[$KEEP_LIST]"
# echo
# echo "When finished, push to the Hub:"
# echo "  python -c \"from lerobot.datasets.lerobot_dataset import LeRobotDataset; LeRobotDataset('$REPO', root='$ROOT').push_to_hub()\""
# echo "  huggingface-cli upload $REPO $LOG task2_demos_log.csv --repo-type dataset"




























# #!/usr/bin/env bash
# # Task 2 block-stacking data collection loop (wrist camera only).
# # One lerobot-record call per episode. Shows the next configuration from the
# # schedule, records, then asks keep/discard and logs it. Safe to stop and restart:
# # it resumes from the log.
# #
# # Usage:   bash record_task2.sh
# # Stop:    answer "s" at the prompt (or Ctrl+C between episodes)

# # ======================= EDIT THESE (verify before running) =======================
# FOLLOWER_PORT=/dev/ttyACM1          # lerobot-find-port if unsure (these have swapped before)
# LEADER_PORT=/dev/ttyACM0
# FOLLOWER_ID=so101_follower_vasco
# LEADER_ID=so101_leader_vasco
# WRIST=/dev/video4                  # check with: v4l2-ctl --list-devices
# HF_USER=shilpitha-03
# REPO="$HF_USER/so101_stack_rpy"
# ROOT="$HOME/.cache/huggingface/lerobot/$REPO"
# TASK="Stack the red, yellow and purple cubes on the table mark"
# SCHEDULE="./task2_collection_schedule_v3.csv"
# LOG="./task2_demos_log.csv"
# EPISODE_TIME_S=100                  # hard cap; press RIGHT ARROW to end once arm is home
# DISPLAY_DATA=true                   # rerun viewer of the wrist feed
# PILOT_N=5                           # pause after this many kept episodes for the replay check
# # ==================================================================================

# set -u
# command -v lerobot-record >/dev/null || { echo "!! lerobot-record not found. Activate the env: conda activate lerobot"; exit 1; }
# [ -f "$SCHEDULE" ] || { echo "!! Schedule not found: $SCHEDULE"; exit 1; }
# for d in "$FOLLOWER_PORT" "$LEADER_PORT" "$WRIST"; do
#   [ -e "$d" ] || { echo "!! $d does not exist. Check ports/camera and edit the top of this script."; exit 1; }
# done

# meta() {  # meta total_episodes | total_frames
#   python3 - "$ROOT/meta/info.json" "$1" <<'PY'
# import json,sys
# try: print(json.load(open(sys.argv[1]))[sys.argv[2]])
# except Exception: print(0)
# PY
# }

# row() {  # row N -> prints shell assignments for schedule row N (0-based), or END
#   python3 - "$SCHEDULE" "$1" <<'PY'
# import csv,sys,shlex
# rows=list(csv.DictReader(open(sys.argv[1])))
# n=int(sys.argv[2])
# if n>=len(rows): print("END"); sys.exit()
# r=rows[n]
# prev=rows[n-1]["lighting"] if n>0 else ""
# keys=["episode_index","type","lighting","T1","T2","T3","placement","pre_stacked","pre_stack_offset"]
# for k in keys: print(f"S_{k}={shlex.quote(r[k])}")
# print(f"S_prevlight={shlex.quote(prev)}")
# print(f"S_total={len(rows)}")
# PY
# }

# [ -f "$LOG" ] || echo "dataset_episode,schedule_row,type,lighting,T1,T2,T3,placement,pre_stacked,pre_stack_offset,kept,frames,seconds,note,timestamp" > "$LOG"

# if [ -d "$ROOT" ] && [ "$(meta total_episodes)" = "0" ]; then
#   echo "!! $ROOT exists but has no saved episodes (a failed earlier start)."
#   read -r -p "Delete it and start fresh? [y/N]: " a || exit 0
#   [ "$a" = "y" ] && rm -rf "$ROOT" || exit 1
# fi

# # Sanity: dataset episode count must match the log
# N_DS=$(meta total_episodes)
# N_LOG=$(( $(wc -l < "$LOG") - 1 ))
# if [ "$N_DS" -ne "$N_LOG" ]; then
#   echo "!! Dataset has $N_DS episodes but the log has $N_LOG rows. Fix this before continuing"
#   echo "   (an episode was saved without being logged, or the log belongs to another dataset)."
#   exit 1
# fi

# LIGHT_DESC() { case "$1" in
#   L1) echo "L1 = enclosure lamp NORMAL";;
#   L2) echo "L2 = lamp DIMMED/moved, room lights OFF";;
#   L3) echo "L3 = lamp normal + EXTRA SIDE LIGHT from open side";;
#   *) echo "$1";; esac; }

# while true; do
#   KEPT=$(awk -F, 'NR>1 && $11==1' "$LOG" | wc -l)
#   OUT=$(row "$KEPT")
#   [ "$OUT" = "END" ] && { echo; echo "== Schedule complete: $KEPT kept episodes. =="; break; }
#   eval "$OUT"
#   N=$(meta total_episodes)

#   clear
#   echo "================================================================"
#   echo "  Schedule row $KEPT / $((S_total-1))      next dataset episode: $N"
#   echo "================================================================"
#   if [ "$S_lighting" != "$S_prevlight" ]; then
#     echo "  >>> LIGHTING CHANGE: $(LIGHT_DESC "$S_lighting") <<<"
#   else
#     echo "  Lighting: $(LIGHT_DESC "$S_lighting")"
#   fi
#   echo
#   echo "  Type: $S_type"
#   printf "  Tray:   T1 = %-8s  T2 = %-8s  T3 = %-8s\n" "$S_T1" "$S_T2" "$S_T3"
#   echo "  Placement: $S_placement"
#   if [ "$S_type" != "full" ]; then
#     echo "  PRE-STACK on table mark: $S_pre_stacked  ($S_pre_stack_offset)"
#   fi
#   echo
#   echo "  Pattern: start -> LOOK -> red -> LOOK -> yellow -> LOOK -> purple -> LOOK -> home"
#   echo "  (pre-stacked: start -> LOOK -> continue from current stack)"
#   echo "  Look = brief 0.5-1 s pass. Lift straight up 3-4 cm after each release."
#   echo "  When the arm is back at start pose: press RIGHT ARROW (keep this terminal focused)."
#   echo "----------------------------------------------------------------"
#   read -r -p "  Scene set, leader + follower at START POSE? [Enter] record   [s] stop: " ans || exit 0
#   [ "$ans" = "s" ] && break

#   if [ -d "$ROOT" ]; then RESUME=true; else RESUME=false; fi
#   F0=$(meta total_frames)

#   lerobot-record \
#     --robot.type=so101_follower \
#     --robot.port="$FOLLOWER_PORT" \
#     --robot.id="$FOLLOWER_ID" \
#     --robot.cameras="{ wrist: {type: opencv, index_or_path: $WRIST, width: 480, height: 640, fps: 30, fourcc: MJPG, rotation: -90}}" \
#     --teleop.type=so101_leader \
#     --teleop.port="$LEADER_PORT" \
#     --teleop.id="$LEADER_ID" \
#     --display_data="$DISPLAY_DATA" \
#     --dataset.repo_id="$REPO" \
#     --dataset.root="$ROOT" \
#     --dataset.single_task="$TASK" \
#     --dataset.num_episodes=1 \
#     --dataset.fps=30 \
#     --dataset.episode_time_s="$EPISODE_TIME_S" \
#     --dataset.reset_time_s=0 \
#     --dataset.push_to_hub=false \
#     --dataset.streaming_encoding=true \
#     --dataset.encoder_threads=2 \
#     --dataset.no_stamp=true \
#     --resume="$RESUME"
#   RC=$?
#   if [ "$RC" -ne 0 ]; then
#     echo "!! lerobot-record exited with an error (code $RC). Stopping - see the traceback above."
#     exit 1
#   fi

#   NEW=$(meta total_episodes)
#   if [ "$NEW" -le "$N" ]; then
#     echo "!! No new episode saved. Same configuration will be shown again."
#     read -r -p "   [Enter] to continue" _ || exit 0; continue
#   fi
#   F1=$(meta total_frames); FR=$((F1 - F0)); SEC=$((FR / 30))
#   echo
#   echo ">> Saved dataset episode $N: $FR frames (~${SEC} s)"
#   if [ "$S_type" = "full" ] && [ "$SEC" -lt 40 ]; then
#     echo "   WARNING: only ${SEC} s for a full stack - ended early by accident?"
#   fi
#   if [ "$SEC" -ge $((EPISODE_TIME_S - 1)) ]; then
#     echo "   WARNING: hit the ${EPISODE_TIME_S} s cap - did it finish and return home?"
#   fi
#   echo "   Discard if: stack fell, regrasp/fumble, pause > 1 s, look pose held too long, didn't return home."
#   read -r -p "Keep it? [Enter] keep   [d] discard: " k || exit 0
#   TS=$(date +%F_%T)
#   COMMON="$N,$S_episode_index,$S_type,$S_lighting,$S_T1,$S_T2,$S_T3,$S_placement,$S_pre_stacked,$S_pre_stack_offset"
#   if [ "$k" = "d" ]; then
#     read -r -p "Short reason: " note || exit 0
#     echo "$COMMON,0,$FR,$SEC,${note//,/;},$TS" >> "$LOG"
#     echo "   Discarded (stays in dataset, excluded from training). Same configuration again."
#     sleep 1
#   else
#     read -r -p "Optional note (Enter to skip): " note || exit 0
#     echo "$COMMON,1,$FR,$SEC,${note//,/;},$TS" >> "$LOG"
#     KEPT_NOW=$((KEPT + 1))
#     if [ "$KEPT_NOW" -eq "$PILOT_N" ]; then
#       echo
#       echo "== PILOT DONE ($PILOT_N kept). Replay-check one episode before continuing: =="
#       echo "lerobot-replay --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$FOLLOWER_ID \\"
#       echo "  --dataset.repo_id=$REPO --dataset.root=$ROOT --dataset.episode=0"
#       echo "Also scrub the wrist video: at every LOOK are all tray cubes + the stack mark visible,"
#       echo "and do red vs purple look clearly different?"
#       read -r -p "[Enter] when checked to continue recording, [s] stop: " a || exit 0
#       [ "$a" = "s" ] && break
#     fi
#   fi
# done

# KEEP_LIST=$(awk -F, 'NR>1 && $11==1 {printf "%s%s", sep, $1; sep=","}' "$LOG")
# echo
# echo "Kept dataset episodes (use for training):"
# echo "  --dataset.episodes=[$KEEP_LIST]"
# echo
# echo "When finished, push to the Hub:"
# echo "  python -c \"from lerobot.datasets.lerobot_dataset import LeRobotDataset; LeRobotDataset('$REPO', root='$ROOT').push_to_hub()\""
# echo "  huggingface-cli upload $REPO $LOG task2_demos_log.csv --repo-type dataset"
