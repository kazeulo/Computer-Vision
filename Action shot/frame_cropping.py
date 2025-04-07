import cv2
import numpy as np
import matplotlib.pyplot as plt

def custom_crop(frame, start_x, end_x, crop_height):
    """
    Crops the frame from a specific start_x to end_x while keeping the full height.
    """
    frame_height, frame_width, _ = frame.shape
    start_x = max(0, start_x)
    end_x = min(frame_width, end_x)
    cropped_frame = frame[0:crop_height, start_x:end_x]
    return cropped_frame

# Load video and capture frames
cap = cv2.VideoCapture('video/kick4.mp4')
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_numbers = [1, 35, 48, 58, 70, 95]
frames = []

for frame_number in frame_numbers:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        frames.append(frame)
    else:
        print(f"Error: Could not read frame {frame_number}")
cap.release()

if not frames:
    print("Error: No frames were extracted.")
    exit()

plt.figure(figsize=(10, 10))
crop_height = frames[0].shape[0]
crop_positions = [
    (1, 600), 
    (410, 640), 
    (620, 920), 
    (875, 1150), 
    (1120, 1380), 
    (1280, 2000)
]

# Plot cropped frames without feathering
for i, frame in enumerate(frames):
    start_x, end_x = crop_positions[i]
    cropped_frame = custom_crop(frame, start_x, end_x, crop_height)
    plt.subplot(1, len(frames), i + 1)
    plt.imshow(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
    plt.title(f"Frame {frame_numbers[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()