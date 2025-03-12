import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# Load the pre-trained face aging model (dummy placeholder here, replace with actual model)
class FaceAgingModel:
    def __init__(self, model_path):
        self.model = torch.load(model_path)  # Load a pre-trained model
        self.model.eval()

    def age_face(self, face_image):
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        face_tensor = transform(face_image).unsqueeze(0)  # Convert to tensor

        with torch.no_grad():
            aged_face = self.model(face_tensor)  # Apply aging model

        return aged_face.squeeze(0)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the face aging model
face_aging_model = FaceAgingModel('path_to_aging_model.pth')

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Crop the face from the frame
        face_image = frame[y:y + h, x:x + w]
        
        # Convert the face image to PIL for model processing
        pil_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        
        # Apply face aging
        aged_face = face_aging_model.age_face(pil_image)

        # Convert the aged face back to OpenCV format
        aged_face = transforms.ToPILImage()(aged_face).convert("RGB")
        aged_face = cv2.cvtColor(np.array(aged_face), cv2.COLOR_RGB2BGR)

        # Replace the original face with the aged face
        frame[y:y + h, x:x + w] = aged_face

    # Display the frame with aging applied
    cv2.imshow('Real-Time Face Aging', frame)

    # Break the loop on pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the display window
cap.release()
cv2.destroyAllWindows()