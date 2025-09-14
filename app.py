import cv2
import datetime
import streamlit as st
import tempfile
import os

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

st.title("AI Exam Malpractice Detector")

# Button to start webcam
run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])
mal_count = 0

if run:
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Webcam not accessible")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Date & Time
        dt = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        cv2.putText(frame, dt, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Conditions
        if len(faces) == 0:
            cv2.putText(frame, "No Student Detected!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        elif len(faces) == 1:
            cv2.putText(frame, "Safe", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "Malpractice Detected!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Save screenshot
            filename = f"malpractice_{dt.replace(':', '-')}.jpg"
            filepath = os.path.join(tempfile.gettempdir(), filename)
            cv2.imwrite(filepath, frame)

            mal_count += 1
            cv2.putText(frame, f"Malpractice Count: {mal_count}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Draw rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Show in Streamlit (BGR → RGB)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()