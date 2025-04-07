import cv2
import numpy as np
import random

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Game variables
score = 0
player_x, player_y = 250, 320  # Starting position of the player (face-controlled character), a bit higher up
player_width, player_height = 50, 50
player_velocity = 0
gravity = 1
jump_strength = -15
bounce_strength = -12  # Bounce strength when hitting the ground
on_ground = False  # Initially the player is not on the ground
jump_range = 50  # Max height the player can jump

# Ground setup (solid ground at the bottom of the screen)
ground_y = 480  # Y position of the solid ground (bottom of the screen)
ground_height = 20  # Height of the ground

# Platform setup (initialize first platform)
platform_width, platform_height = 100, 20
platforms = [(0, ground_y)]  # Starting platform (solid ground at the bottom of the screen)

# Function to generate a new platform within the player's jump range
def generate_platforms(last_platform_y):
    platform_y = random.randint(last_platform_y - jump_range, last_platform_y - 50)  # Ensure the platform is within jump range
    platform_x = random.randint(100, 500)  # Random horizontal position
    return (platform_x, platform_y)

# Generate initial platforms
for _ in range(5):
    platforms.append(generate_platforms(platforms[-1][1]))

# Game loop
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

    # Draw platforms (including the ground)
    for platform in platforms:
        cv2.rectangle(frame, (platform[0], platform[1]), (platform[0] + platform_width, platform[1] + platform_height), (0, 255, 0), -1)
    
    # Draw the solid ground
    cv2.rectangle(frame, (0, ground_y), (frame.shape[1], ground_y + ground_height), (0, 0, 255), -1)

    # Draw player (face-controlled character)
    cv2.rectangle(frame, (player_x, player_y), (player_x + player_width, player_y + player_height), (255, 0, 0), -1)

    # Process each detected face
    for (x, y, w, h) in faces:
        face_center = (x + w // 2, y + h // 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Update player position based on face location
        player_x = face_center[0] - player_width // 2

    # Apply gravity and check if player is on ground or on a platform
    if not on_ground:
        player_velocity += gravity  # Gravity pulls the player down
    player_y += player_velocity

    # Check for collisions with the solid ground
    if player_y + player_height >= ground_y:
        player_y = ground_y - player_height  # Place player on top of the ground
        player_velocity = bounce_strength  # Apply bounce effect when landing on the ground
        on_ground = True

    # Check for collisions with other platforms (simple platform collision detection)
    for platform in platforms:
        if player_x + player_width > platform[0] and player_x < platform[0] + platform_width:
            if player_y + player_height <= platform[1] and player_y + player_height + player_velocity >= platform[1]:
                player_velocity = jump_strength  # Jump if colliding with platform
                player_y = platform[1] - player_height  # Place player on top of platform
                on_ground = True
                # After landing on a platform, generate the next platform within jumping range
                platforms.append(generate_platforms(platform[1]))
                break
    else:
        # If not on any platform, we are still falling
        on_ground = False

    # Score based on height (player's Y position)
    score = max(0, 100 - (frame.shape[0] - player_y))  # Score decreases as player falls

    # Display score
    cv2.putText(frame, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the output frame
    cv2.imshow('Face Detection Game', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
