#!/usr/bin/env python3
"""
grid_overlay.py - project the cube-placement grid onto the external camera feed,
so nothing has to be taped on the board.

Board frame (cm): origin = shoulder-pan axis on the table, x = forward, y = LEFT (+).

1) Make a randomized collection schedule (also used as demos.csv for episode_qc.py):
   python grid_overlay.py schedule --rounds 3 --seed 0 --out schedule.csv

2) Calibrate once (and again if the camera is ever moved):
   Lay a Letter sheet (27.94 x 21.59 cm) flat at a measured position, then
   python grid_overlay.py calibrate --cam 2 --ref "10,10;10,-11.59;37.94,-11.59;37.94,10"
   Click the 4 corners in the SAME order as --ref (board coords x,y in cm). Remove the sheet.

3) Live placement:
   python grid_overlay.py live --cam 2 --schedule schedule.csv
   Keys: n / p = next / previous episode target, h = toggle all labels, q = quit.
   --index N starts at schedule row N (record_loop.sh passes the next episode index).

Calibration reference points stay drawn in yellow: if they no longer sit on the
real marks, the camera has moved (re-calibrate AND treat it as a dataset problem).
"""
import argparse, csv, json, random, sys
import numpy as np

# ----- layout (cm, board frame) - edit here if the plan changes ------------------
ROWS_X = [12, 16, 20, 24]                 # forward distance of rows 1..4
COLS_Y = [4, 8, 12, 16]                   # left distance of cols A..D
CUBE = 2.5                                # cube edge
TRAY_CENTER, TRAY_SIZE = (16.0, -15.0), 10.0
# ---------------------------------------------------------------------------------


def nodes():
    """Training nodes A1..D4."""
    return {f"{'ABCD'[j]}{i + 1}": (x, y) for i, x in enumerate(ROWS_X) for j, y in enumerate(COLS_Y)}


def cell_centers():
    """Unseen eval positions, named by the cell's lower-left node, e.g. 'c-A1'."""
    out = {}
    for i in range(len(ROWS_X) - 1):
        for j in range(len(COLS_Y) - 1):
            out[f"c-{'ABCD'[j]}{i + 1}"] = ((ROWS_X[i] + ROWS_X[i + 1]) / 2, (COLS_Y[j] + COLS_Y[j + 1]) / 2)
    return out


def all_positions():
    return {**nodes(), **cell_centers()}


def cube_footprint(x, y, size=CUBE):
    """Square footprint with one face pointing at the pan axis (radial yaw)."""
    yaw = np.arctan2(y, x)
    c, s = np.cos(yaw), np.sin(yaw)
    h = size / 2
    local = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
    return local @ np.array([[c, s], [-s, c]]) + [x, y]


def project(H, pts):
    pts = np.asarray(pts, float).reshape(-1, 1, 2)
    import cv2
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def make_schedule(rounds, seed, out):
    rng = random.Random(seed)
    ids = list(nodes())
    rows, ep = [], 0
    for r in range(rounds):
        order = ids[:]
        rng.shuffle(order)
        if rows and order[0] == rows[-1]["grid_id"]:   # no back-to-back repeat across rounds
            order[0], order[1] = order[1], order[0]
        for gid in order:
            rows.append(dict(episode_index=ep, grid_id=gid, round=r + 1))
            ep += 1
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode_index", "grid_id", "round"])
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} episodes to {out}")


def open_cam(cam, w, h):
    """Open with the V4L2 backend explicitly (a path string can otherwise pick a backend
    that ignores the resolution request), force MJPG + w x h, and verify the frame size:
    the calibration is only valid at the resolution it was made at."""
    import cv2
    src = int(cam) if str(cam).isdigit() else cam
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {cam} (is lerobot-record holding it?)")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    for _ in range(10):
        ok, frame = cap.read()
        if ok:
            break
    if not ok:
        sys.exit(f"Camera {cam} opened but returned no frames")
    fh, fw = frame.shape[:2]
    if (fw, fh) != (w, h):
        sys.exit(f"Camera {cam} delivers {fw}x{fh}, but the calibration is for {w}x{h}. "
                 f"Overlay would be misplaced; check the device / supported modes.")
    return cap


def calibrate(args):
    import cv2
    ref = [tuple(map(float, p.split(","))) for p in args.ref.split(";")]
    if len(ref) < 4:
        sys.exit("Need at least 4 reference points")
    cap = open_cam(args.cam, args.width, args.height)
    clicks = []
    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", lambda e, x, y, *_: clicks.append((x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        for k, (u, v) in enumerate(clicks):
            cv2.circle(frame, (u, v), 4, (0, 255, 255), -1)
            cv2.putText(frame, str(k + 1), (u + 6, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        msg = f"click point {len(clicks) + 1}/{len(ref)} = (fwd {ref[len(clicks)][0]}, left {ref[len(clicks)][1]}) cm" if len(clicks) < len(ref) else "done - press s to save, r to redo"
        cv2.putText(frame, msg, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.imshow("calibrate", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("r"):
            clicks.clear()
        if key == ord("q"):
            return
        if key == ord("s") and len(clicks) >= len(ref):
            H, _ = cv2.findHomography(np.array(ref), np.array(clicks[:len(ref)], float))
            err = np.linalg.norm(project(H, ref) - np.array(clicks[:len(ref)]), axis=1).max()
            json.dump(dict(H=H.tolist(), ref_board=ref, ref_img=clicks[:len(ref)],
                           width=args.width, height=args.height), open(args.calib, "w"), indent=2)
            print(f"Saved {args.calib} (max reprojection error {err:.1f} px)")
            return


def draw_overlay(frame, H, target=None, labels=True, ref=None):
    import cv2
    def poly(pts, color, thick=1):
        cv2.polylines(frame, [project(H, pts).astype(np.int32)], True, color, thick, cv2.LINE_AA)
    tx, ty = TRAY_CENTER; h = TRAY_SIZE / 2
    poly([[tx - h, ty - h], [tx + h, ty - h], [tx + h, ty + h], [tx - h, ty + h]], (80, 120, 255), 1)
    for gid, (x, y) in all_positions().items():
        is_node = not gid.startswith("c-")
        if gid == target:
            poly(cube_footprint(x, y), (0, 255, 0), 2)
            u, v = project(H, [[x, y]])[0]
            cv2.putText(frame, gid, (int(u) + 8, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            u, v = project(H, [[x, y]])[0]
            cv2.circle(frame, (int(u), int(v)), 3 if is_node else 2, (200, 120, 200) if is_node else (150, 200, 150), -1)
            if labels:
                cv2.putText(frame, gid, (int(u) + 4, int(v) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)
    if ref is not None:
        for (u, v) in project(H, ref):
            cv2.drawMarker(frame, (int(u), int(v)), (0, 255, 255), cv2.MARKER_CROSS, 10, 1)
    return frame


def live(args):
    import cv2
    cal = json.load(open(args.calib))
    H = np.array(cal["H"])
    sched = list(csv.DictReader(open(args.schedule))) if args.schedule else []
    i, labels = (min(args.index, len(sched) - 1) if sched else 0), True
    cap = open_cam(args.cam, cal["width"], cal["height"])
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        target = sched[i]["grid_id"] if sched else None
        draw_overlay(frame, H, target, labels, cal["ref_board"])
        if sched:
            cv2.putText(frame, f"episode {sched[i]['episode_index']}  round {sched[i].get('round', '-')}  place at {target}",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("placement", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("n") and sched:
            i = min(i + 1, len(sched) - 1)
        if key == ord("p") and sched:
            i = max(i - 1, 0)
        if key == ord("h"):
            labels = not labels


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("schedule"); s.add_argument("--rounds", type=int, default=3)
    s.add_argument("--seed", type=int, default=0); s.add_argument("--out", default="schedule.csv")
    c = sub.add_parser("calibrate"); c.add_argument("--cam", required=True); c.add_argument("--ref", required=True, help="'fwd,left;fwd,left;...' in cm from the pan axis; LEFT of robot is positive")
    c.add_argument("--calib", default="overlay_calib.json")
    c.add_argument("--width", type=int, default=640); c.add_argument("--height", type=int, default=480)
    l = sub.add_parser("live"); l.add_argument("--cam", required=True)
    l.add_argument("--calib", default="overlay_calib.json"); l.add_argument("--schedule", default=None)
    l.add_argument("--index", type=int, default=0, help="start at this schedule row (= episode index)")
    a = ap.parse_args()
    {"schedule": lambda: make_schedule(a.rounds, a.seed, a.out), "calibrate": lambda: calibrate(a), "live": lambda: live(a)}[a.cmd]()


if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # #!/usr/bin/env python3
# """
# grid_overlay.py - project the cube-placement grid onto the external camera feed,
# so nothing has to be taped on the board.

# Board frame (cm): origin = shoulder-pan axis on the table, x = forward, y = LEFT (+).

# 1) Make a randomized collection schedule (also used as demos.csv for episode_qc.py):
#    python grid_overlay.py schedule --rounds 3 --seed 0 --out schedule.csv

# 2) Calibrate once (and again if the camera is ever moved):
#    Lay a Letter sheet (27.94 x 21.59 cm) flat at a measured position, then
#    python grid_overlay.py calibrate --cam 2 --ref "10,10;10,-11.59;37.94,-11.59;37.94,10"
#    Click the 4 corners in the SAME order as --ref (board coords x,y in cm). Remove the sheet.

# 3) Live placement:
#    python grid_overlay.py live --cam 2 --schedule schedule.csv
#    Keys: n / p = next / previous episode target, h = toggle all labels, q = quit.
#    --index N starts at schedule row N (record_loop.sh passes the next episode index).

# Calibration reference points stay drawn in yellow: if they no longer sit on the
# real marks, the camera has moved (re-calibrate AND treat it as a dataset problem).
# """
# import argparse, csv, json, random, sys
# import numpy as np

# # ----- layout (cm, board frame) - edit here if the plan changes ------------------
# ROWS_X = [12, 16, 20, 24]                 # forward distance of rows 1..4
# COLS_Y = [4, 8, 12, 16]                   # left distance of cols A..D
# CUBE = 2.5                                # cube edge
# TRAY_CENTER, TRAY_SIZE = (16.0, -15.0), 10.0
# # ---------------------------------------------------------------------------------


# def nodes():
#     """Training nodes A1..D4."""
#     return {f"{'ABCD'[j]}{i + 1}": (x, y) for i, x in enumerate(ROWS_X) for j, y in enumerate(COLS_Y)}


# def cell_centers():
#     """Unseen eval positions, named by the cell's lower-left node, e.g. 'c-A1'."""
#     out = {}
#     for i in range(len(ROWS_X) - 1):
#         for j in range(len(COLS_Y) - 1):
#             out[f"c-{'ABCD'[j]}{i + 1}"] = ((ROWS_X[i] + ROWS_X[i + 1]) / 2, (COLS_Y[j] + COLS_Y[j + 1]) / 2)
#     return out


# def all_positions():
#     return {**nodes(), **cell_centers()}


# def cube_footprint(x, y, size=CUBE):
#     """Square footprint with one face pointing at the pan axis (radial yaw)."""
#     yaw = np.arctan2(y, x)
#     c, s = np.cos(yaw), np.sin(yaw)
#     h = size / 2
#     local = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
#     return local @ np.array([[c, s], [-s, c]]) + [x, y]


# def project(H, pts):
#     pts = np.asarray(pts, float).reshape(-1, 1, 2)
#     import cv2
#     return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


# def make_schedule(rounds, seed, out):
#     rng = random.Random(seed)
#     ids = list(nodes())
#     rows, ep = [], 0
#     for r in range(rounds):
#         order = ids[:]
#         rng.shuffle(order)
#         if rows and order[0] == rows[-1]["grid_id"]:   # no back-to-back repeat across rounds
#             order[0], order[1] = order[1], order[0]
#         for gid in order:
#             rows.append(dict(episode_index=ep, grid_id=gid, round=r + 1))
#             ep += 1
#     with open(out, "w", newline="") as f:
#         w = csv.DictWriter(f, fieldnames=["episode_index", "grid_id", "round"])
#         w.writeheader(); w.writerows(rows)
#     print(f"Wrote {len(rows)} episodes to {out}")


# def open_cam(cam, w, h):
#     """Open with the V4L2 backend explicitly (a path string can otherwise pick a backend
#     that ignores the resolution request), force MJPG + w x h, and verify the frame size:
#     the calibration is only valid at the resolution it was made at."""
#     import cv2
#     src = int(cam) if str(cam).isdigit() else cam
#     cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
#     if not cap.isOpened():
#         sys.exit(f"Could not open camera {cam} (is lerobot-record holding it?)")
#     cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
#     for _ in range(10):
#         ok, frame = cap.read()
#         if ok:
#             break
#     if not ok:
#         sys.exit(f"Camera {cam} opened but returned no frames")
#     fh, fw = frame.shape[:2]
#     if (fw, fh) != (w, h):
#         sys.exit(f"Camera {cam} delivers {fw}x{fh}, but the calibration is for {w}x{h}. "
#                  f"Overlay would be misplaced; check the device / supported modes.")
#     return cap


# def calibrate(args):
#     import cv2
#     ref = [tuple(map(float, p.split(","))) for p in args.ref.split(";")]
#     if len(ref) < 4:
#         sys.exit("Need at least 4 reference points")
#     cap = open_cam(args.cam, args.width, args.height)
#     clicks = []
#     cv2.namedWindow("calibrate")
#     cv2.setMouseCallback("calibrate", lambda e, x, y, *_: clicks.append((x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)
#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             continue
#         for k, (u, v) in enumerate(clicks):
#             cv2.circle(frame, (u, v), 4, (0, 255, 255), -1)
#             cv2.putText(frame, str(k + 1), (u + 6, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
#         msg = f"click point {len(clicks) + 1}/{len(ref)} = (fwd {ref[len(clicks)][0]}, left {ref[len(clicks)][1]}) cm" if len(clicks) < len(ref) else "done - press s to save, r to redo"
#         cv2.putText(frame, msg, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
#         cv2.imshow("calibrate", frame)
#         key = cv2.waitKey(30) & 0xFF
#         if key == ord("r"):
#             clicks.clear()
#         if key == ord("q"):
#             return
#         if key == ord("s") and len(clicks) >= len(ref):
#             H, _ = cv2.findHomography(np.array(ref), np.array(clicks[:len(ref)], float))
#             err = np.linalg.norm(project(H, ref) - np.array(clicks[:len(ref)]), axis=1).max()
#             json.dump(dict(H=H.tolist(), ref_board=ref, ref_img=clicks[:len(ref)],
#                            width=args.width, height=args.height), open(args.calib, "w"), indent=2)
#             print(f"Saved {args.calib} (max reprojection error {err:.1f} px)")
#             return


# def draw_overlay(frame, H, target=None, labels=True, ref=None):
#     import cv2
#     def poly(pts, color, thick=1):
#         cv2.polylines(frame, [project(H, pts).astype(np.int32)], True, color, thick, cv2.LINE_AA)
#     tx, ty = TRAY_CENTER; h = TRAY_SIZE / 2
#     poly([[tx - h, ty - h], [tx + h, ty - h], [tx + h, ty + h], [tx - h, ty + h]], (80, 120, 255), 1)
#     for gid, (x, y) in all_positions().items():
#         is_node = not gid.startswith("c-")
#         if gid == target:
#             poly(cube_footprint(x, y), (0, 255, 0), 2)
#             u, v = project(H, [[x, y]])[0]
#             cv2.putText(frame, gid, (int(u) + 8, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#         else:
#             u, v = project(H, [[x, y]])[0]
#             cv2.circle(frame, (int(u), int(v)), 3 if is_node else 2, (200, 120, 200) if is_node else (150, 200, 150), -1)
#             if labels:
#                 cv2.putText(frame, gid, (int(u) + 4, int(v) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)
#     if ref is not None:
#         for (u, v) in project(H, ref):
#             cv2.drawMarker(frame, (int(u), int(v)), (0, 255, 255), cv2.MARKER_CROSS, 10, 1)
#     return frame


# def live(args):
#     import cv2
#     cal = json.load(open(args.calib))
#     H = np.array(cal["H"])
#     sched = list(csv.DictReader(open(args.schedule))) if args.schedule else []
#     i, labels = (min(args.index, len(sched) - 1) if sched else 0), True
#     cap = open_cam(args.cam, cal["width"], cal["height"])
#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             continue
#         target = sched[i]["grid_id"] if sched else None
#         draw_overlay(frame, H, target, labels, cal["ref_board"])
#         if sched:
#             cv2.putText(frame, f"episode {sched[i]['episode_index']}  round {sched[i].get('round', '-')}  place at {target}",
#                         (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#         cv2.imshow("placement", frame)
#         key = cv2.waitKey(30) & 0xFF
#         if key == ord("q"):
#             break
#         if key == ord("n") and sched:
#             i = min(i + 1, len(sched) - 1)
#         if key == ord("p") and sched:
#             i = max(i - 1, 0)
#         if key == ord("h"):
#             labels = not labels


# def main():
#     ap = argparse.ArgumentParser()
#     sub = ap.add_subparsers(dest="cmd", required=True)
#     s = sub.add_parser("schedule"); s.add_argument("--rounds", type=int, default=3)
#     s.add_argument("--seed", type=int, default=0); s.add_argument("--out", default="schedule.csv")
#     c = sub.add_parser("calibrate"); c.add_argument("--cam", required=True); c.add_argument("--ref", required=True, help="'fwd,left;fwd,left;...' in cm from the pan axis; LEFT of robot is positive")
#     c.add_argument("--calib", default="overlay_calib.json")
#     c.add_argument("--width", type=int, default=640); c.add_argument("--height", type=int, default=480)
#     l = sub.add_parser("live"); l.add_argument("--cam", required=True)
#     l.add_argument("--calib", default="overlay_calib.json"); l.add_argument("--schedule", default=None)
#     l.add_argument("--index", type=int, default=0, help="start at this schedule row (= episode index)")
#     a = ap.parse_args()
#     {"schedule": lambda: make_schedule(a.rounds, a.seed, a.out), "calibrate": lambda: calibrate(a), "live": lambda: live(a)}[a.cmd]()


# if __name__ == "__main__":
#     main()