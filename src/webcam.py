import cv2, insightface


camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
app = insightface.app.FaceAnalysis(
    name="buffalo_s",
    # CUDA最強
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    allowed_modules=["detection", "landmark_2d_106", "genderage"]
)
app.prepare(ctx_id=0)

cv2.namedWindow("out", cv2.WINDOW_NORMAL)
while True:
    _, frame = camera.read()
    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(dtype="int")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, text=f"age: {face.age}", org=(10, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9,
                color=(0, 255, 0), thickness=2)
        cv2.putText(frame, text=f"gender: {face.sex}", org=(10, 100),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9,
                color=(0, 255, 0), thickness=2)

    cv2.imshow("out", frame)
    if cv2.waitKey(1) == 27:
        break
camera.release()
cv2.destroyAllWindows()