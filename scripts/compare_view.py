#!/usr/bin/env python3
"""compare_view.py - has the external camera moved, or does the scene look different, since recording?

Saves view_compare.png: [training frame | live frame | 50/50 blend].
In the blend, anything fixed (robot base, tray, walls) should look sharp. Double edges = camera moved.
Run in the lerobot env, with no other program using the camera:
    python compare_view.py --cam /dev/video4 --episode 0
"""
import argparse, os
import cv2, numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ap = argparse.ArgumentParser()
ap.add_argument("--cam", default="/dev/video4")
ap.add_argument("--episode", type=int, default=0)
ap.add_argument("--repo", default="shilpitha-03/so101_cube_grid")
a = ap.parse_args()

root = os.path.expanduser(f"~/.cache/huggingface/lerobot/{a.repo}")
ds = LeRobotDataset(a.repo, root=root, episodes=[a.episode])
img = ds[0]["observation.images.front"]                       # first frame of that episode, C,H,W in [0,1]
train = np.ascontiguousarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1])   # RGB -> BGR

src = int(a.cam) if str(a.cam).isdigit() else a.cam
cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
for _ in range(30):                                           # let auto exposure settle
    ok, live = cap.read()
cap.release()
if not ok:
    raise SystemExit(f"could not read from {a.cam}")
live = cv2.resize(live, (train.shape[1], train.shape[0]))

blend = cv2.addWeighted(train, 0.5, live, 0.5, 0)
for im, t in ((train, f"training ep {a.episode}"), (live, "live now"), (blend, "blend")):
    cv2.putText(im, t, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imwrite("view_compare.png", np.hstack([train, live, blend]))
print("brightness  training: %.0f   live: %.0f" % (train.mean(), live.mean()))
print("saved view_compare.png")