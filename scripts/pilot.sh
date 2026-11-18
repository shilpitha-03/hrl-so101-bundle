#!/usr/bin/env bash
# pilot.sh - run ONE policy rollout on the robot with lerobot-rollout and record it.
#   ./pilot.sh <name> <policy_path> [extra lerobot-rollout args]
#   e.g. ./pilot.sh act20k /path/to/checkpoints/020000/pretrained_model
# Each <name> gets its own dataset: shilpitha-03/rollout_pilot_<name>. Runs with the same name append.
# Time limit 60 s per rollout.
set -u
NAME="$1"; POLICY="$2"; shift 2

# CAMERAS: "front" must be the overhead camera, as in training (recorded with front=/dev/video4).
# Only change these if the device numbers changed since recording - check with the overlay first.
FRONT="/dev/video4"; WRIST="/dev/video6"
FOLLOWER_PORT="/dev/ttyACM0"

REPO="shilpitha-03/rollout_pilot_$NAME"
ROOT="$HOME/.cache/huggingface/lerobot/$REPO"
if [ -d "$ROOT" ]; then RESUME=true; else RESUME=false; fi

lerobot-rollout \
  --strategy.type=episodic \
  --policy.path="$POLICY" \
  --device=cuda \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id=so101_follower_vasco \
  --robot.cameras="{ front: {type: opencv, index_or_path: $FRONT, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --task="Pick up the cube and place it in the tray" \
  --display_data=true \
  --dataset.repo_id="$REPO" \
  --dataset.root="$ROOT" \
  --dataset.single_task="Pick up the cube and place it in the tray" \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=5 \
  --dataset.push_to_hub=false \
  --resume="$RESUME" \
  "$@"