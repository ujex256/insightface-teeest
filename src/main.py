import glob
import os
import shutil
from collections import deque
from time import time
from uuid import uuid4

import cv2
import numpy as np
import insightface


SAVE = False

if glob.glob("data/output/*.jpg") and SAVE:
    print("Clearing cache...")
    shutil.rmtree("data/output")
    os.mkdir("data/output")


print("Starting GPU...")
app = insightface.app.FaceAnalysis(
    name="buffalo_s",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    allowed_modules=["detection", "landmark_2d_106", "genderage"]
)
app.prepare(ctx_id=0)

s=time()
app.get(np.zeros((1, 1, 3), np.uint8))
print("Test image detect time:", time()-s)
del s


print("Loading images...")
images = deque()
for path in glob.glob("data/*.jpg"):
    images.append(cv2.imread(path))


times = deque()
output = deque()
print("Detection started.")
for img in list(images):
    start = time()
    faces = app.get(img)
    process_time = time() - start
    times.append(process_time)
    print(process_time)

    cp = img.copy()
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(dtype="int")
        cv2.rectangle(cp, (x1, y1), (x2, y2), (255, 0, 0), 3)
    output.append(cp)


print(f"平均: {sum(times) / len(times)}")
if SAVE:
    with open("time.txt", "w", encoding="utf8") as f:
        f.write("\n".join(map(str, times)))

    for i in output:
        cv2.imwrite(rf"data/output/{uuid4()}.jpg", i)

for i in list(output):
    cv2.imshow("Output", i)
    if cv2.waitKey(0) == 27:
        break
cv2.destroyAllWindows()
