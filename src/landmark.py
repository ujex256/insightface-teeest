from time import time
from uuid import uuid4

import cv2
import numpy as np
import insightface


SAVE = True

print("Starting GPU...")
app = insightface.app.FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    allowed_modules=["detection", "landmark_2d_106", "genderage"],
)
app.prepare(ctx_id=0)


s = time()
app.get(np.zeros((1, 1, 3), np.uint8))
print("\nTest image detect time:", time() - s)
del s


print("Loading image...")
img = cv2.imread("data/aaa.jpg")
if img.shape[:2][1] < 550:
    img = cv2.resize(img, None,fx=1.5, fy=1.5)

print("Detection started.")
start = time()
faces = app.get(img)
process_time = time() - start

flag = False
for face in faces:
    x1, y1, x2, y2 = face.bbox.astype(dtype="int")
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)


    for index, i in enumerate(face.landmark_2d_106.astype(dtype="int")):
        cv2.circle(img, (i[0], i[1]), 1, (0, 255, 0), 2)
        if not flag:
            print(index)
            cv2.imshow("Output", img)
            if cv2.waitKey(0) == 27:
                flag = True


if SAVE:
    cv2.imwrite(rf"data/output/{uuid4()}.jpg", img)

cv2.imshow("Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
