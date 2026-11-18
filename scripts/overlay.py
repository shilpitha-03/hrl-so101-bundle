import glob, os, av, cv2

CAM = "/dev/video6"   # front (720P) camera
root = os.path.expanduser("~/.cache/huggingface/lerobot/shilpitha-03/so101_cube_in_box")
vid = sorted(glob.glob(f"{root}/videos/observation.images.front/*/*.mp4"))[0]
ref = next(av.open(vid).decode(video=0)).to_ndarray(format="bgr24")   # first training frame

cap = cv2.VideoCapture(CAM, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
for _ in range(10): ok, live = cap.read()      # let exposure settle
cap.release()
live = cv2.rotate(live, cv2.ROTATE_90_CLOCKWISE)
live = cv2.resize(live, (ref.shape[1], ref.shape[0]))
cv2.imwrite("overlay.png", cv2.addWeighted(live, 0.5, ref, 0.5, 0))
print("Saved overlay.png")