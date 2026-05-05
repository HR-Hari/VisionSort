import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


def detect_faces_and_eyes(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5
    )

    face_data = []

    for (x, y, w, h) in faces:

        face_roi = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(face_roi)

        # If at least 2 eyes → assume open
        eyes_open = 1 if len(eyes) >= 2 else 0

        face_data.append({
            "bbox": (x, y, w, h),
            "eyes_open": eyes_open,
            "area": w * h
        })

    return face_data