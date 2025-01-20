
import dlib
import cv2
from imutils import face_utils
from scipy.spatial import distance
from pygame import mixer
import imutils

# Initialize pygame mixer for sound
mixer.init()
mixer.music.load('music.wav')  # Ensure 'music.wav' is in the same directory as this script

# Define the function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Threshold and frame check values
thresh = 0.25
frame_check = 20

# Load the face detector and shape predictor from the current directory
try:
    predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Path to the shape predictor file
    detect = dlib.get_frontal_face_detector()
except Exception as e:
    print(f"Error loading dlib's shape predictor: {e}")
    exit(1)

# Define indices for the left and right eye landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Start video capture
cap = cv2.VideoCapture(0)
flag = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    subjects = detect(gray, 0)

    for subject in subjects:
        # Get facial landmarks
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculate Eye Aspect Ratio (EAR) for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Average EAR between both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # Draw contours around the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check if EAR is below the threshold, indicating drowsiness
        if ear < thresh:
            flag += 1
            if flag >= frame_check:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()
        else:
            flag = 0

    # Display the frame with the eye contours and EAR status
    cv2.imshow("Driver Drowsiness Detection", frame)

    # Exit on pressing 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
