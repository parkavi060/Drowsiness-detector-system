import cv2
import dlib
import time
from scipy.spatial import distance
from imutils import face_utils
from playsound import playsound
import threading

def sound_alarm():
    playsound("alarm.wav")  # Replace with your alarm sound file

def eye_aspect_ratio(eye):
    # Compute the distances between the vertical eye landmarks
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    # Compute the distance between the horizontal eye landmarks
    C = distance.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Constants for eye aspect ratio to indicate blink
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20

# Initialize the frame counter
COUNTER = 0
ALARM_ON = False

# Load face detector and landmark predictor
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this model

# Get indexes for left and right eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Start video capture
print("[INFO] Starting video stream...")
vs = cv2.VideoCapture(0)
time.sleep(1.0)

while True:
    ret, frame = vs.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculate EAR
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Draw eye contours
        leftHull = cv2.convexHull(leftEye)
        rightHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightHull], -1, (0, 255, 0), 1)

        # Check if EAR is below threshold
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    t = threading.Thread(target=sound_alarm)
                    t.deamon = True
                    t.start()

                cv2.putText(frame, "DROWSINESS DETECTED!...Take rest or Drink Coffee", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            ALARM_ON = False

        cv2.putText(frame, f"EAR: {ear:.2f}", (500, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break

vs.release()
cv2.destroyAllWindows()
