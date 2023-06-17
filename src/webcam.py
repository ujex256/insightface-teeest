import cv2
import insightface

from time import time


camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
app = insightface.app.FaceAnalysis(
    name="buffalo_s",
    # CUDA最強
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    allowed_modules=["detection", "genderage"]
)
app.prepare(ctx_id=0)

cv2.namedWindow("out", cv2.WINDOW_NORMAL)
while True:
    ret, frame = camera.read()
    s = time()
    faces = app.get(frame)
    ps = time() - s
    cv2.putText(frame, text=f"Process time: {ps:.5f}", org=(10, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                color=(0, 255, 0), thickness=2)

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(dtype="int")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.rectangle(frame, (x1, y1 - 30), (min(x1 + 200, x2), y1), (255, 0, 0), -1)
        cv2.putText(frame, text=f"age: {face.age}", org=(x1, y1-10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                    color=(0, 255, 0), thickness=2)
        cv2.putText(frame, text=f"gender: {face.sex}", org=(x1+85, y1-10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                    color=(0, 255, 0), thickness=2)

    cv2.imshow("out", frame)
    if cv2.waitKey(1) == 27 or cv2.getWindowProperty("out", cv2.WND_PROP_AUTOSIZE) == -1:
        break
camera.release()
cv2.destroyAllWindows()