import cv2
import numpy as np
import dlib
from imutils import face_utils
import pygame

# Initialize Pygame for audio
pygame.mixer.init()

# Load audio files for drowsy and sleeping states
drowsy_audio = "drowsy_alarm.wav"
sleeping_audio = "sleeping_alarm.wav"

# Initializing the camera and taking the instance
cap = cv2.VideoCapture(0)

# Set a lower resolution for frame capture
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# define the necessary variables
active = 0
counter = 0
status = ""
color = (0, 0, 0)
threshold = 0.2

# compute and return the Euclidean distance between two points
def euclidean_dist(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

# calculate eye_aspect_ratio (EAR)
def EAR(a, b, c, d, e, f):
    up = euclidean_dist(b, d) + euclidean_dist(c, e)
    down = euclidean_dist(a, f)
    ratio = up / (2.0 * down)
    return ratio

while True:
    ret, frame = cap.read()

    face_frame = frame.copy()
    result = frame.copy()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use the standard face detector
    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

        left_blink = EAR(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = EAR(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        if 0 <= left_blink < threshold or 0 <= right_blink < threshold:
            counter += 1
            active = 0
            if counter > 30:
                status = "SLEEPING"
                color = (255, 0, 0)
                pygame.mixer.music.stop()
                pygame.mixer.music.load(sleeping_audio)
                pygame.mixer.music.play()
            elif counter > 10:
                status = "DROWSY"
                color = (0, 0, 255)
                pygame.mixer.music.stop()
                pygame.mixer.music.load(drowsy_audio)
                pygame.mixer.music.play()
        else:
            active += 1
            if active > 6:
                counter = 0
                status = "Active"
                color = (0, 255, 0)
                pygame.mixer.music.stop()

        cv2.putText(result, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Detected face and landmarks", face_frame)
    cv2.imshow("Result of detector", result)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()




