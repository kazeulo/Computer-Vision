import cv2
import numpy as np
import random

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Game variables
score = 0
target_x, target_y = random.randint(100, 500), random.randint(100, 400)

def draw_target(frame, x, y):
    cv2.circle(frame, (x, y), 30, (0, 0, 255), -1)
    cv2.putText(frame, 'Hit me!', (x - 20, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame to create a mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the target
    draw_target(frame, target_x, target_y)

    # Process each detected face
    for (x, y, w, h) in faces:
        face_center = (x + w//2, y + h//2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Check if the face is near the target
        if abs(face_center[0] - target_x) < 30 and abs(face_center[1] - target_y) < 30:
            score += 1
            target_x, target_y = random.randint(100, 500), random.randint(100, 400)

    # Display score
    cv2.putText(frame, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the output frame
    cv2.imshow('Face Detection Game', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()