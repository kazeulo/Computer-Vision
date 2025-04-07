import os
import cv2
import numpy as np
import random

print("Current working directory:", os.getcwd())

# spaceship - student
# firing at obstacles can be triggered by a smile
# there should be obstacles (kwatro) around arranged randomly
# when you fire at the obstacles, the pieces will be (uno)
# collecting uno = 3 points
# singkos should be coming from everywhere but frequency of this should be low 
# hitting kwatro or singko means sudden death

# Load images and check if they are loaded successfully
spaceship_img = cv2.imread('imgs/character.jpg')
if spaceship_img is None:
    print("Error: character.jpg not found!")
    exit()

spaceship_img = cv2.resize(spaceship_img, (50, 50))

enemy_img1 = cv2.imread('imgs/kwatro.jpg')
if enemy_img1 is None:
    print("Error: kwatro.jpg not found!")
    exit()

enemy_img1 = cv2.resize(enemy_img1, (50, 50))

enemy_img2 = cv2.imread('imgs/singko.jpg')
if enemy_img2 is None:
    print("Error: singko.jpg not found!")
    exit()

enemy_img2 = cv2.resize(enemy_img2, (50, 50))

booster_img = cv2.imread('imgs/uno.jpg')
if booster_img is None:
    print("Error: uno.jpg not found!")
    exit()

booster_img = cv2.resize(booster_img, (30, 30))  # Resize booster to 30x30

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Game variables
score = 0
spaceship_width, spaceship_height = 50, 50
spaceship_x, spaceship_y = 300, 500
enemy_radius = 30
enemy_speed = 3
game_over = False
speed_boost = False
boost_duration = 0
boosters = []
enemies = []

# Function to draw the spaceship using the image
def draw_spaceship(frame, x, y):
    # Ensure that the spaceship coordinates are within bounds of the frame
    if y + spaceship_height > frame.shape[0]:  # Check if y is out of the frame's vertical bounds
        y = frame.shape[0] - spaceship_height
    if x + spaceship_width > frame.shape[1]:  # Check if x is out of the frame's horizontal bounds
        x = frame.shape[1] - spaceship_width
    if x < 0:
        x = 0  # Prevent x from going out of the left bound
    if y < 0:
        y = 0  # Prevent y from going out of the top bound
    
    # Now draw the spaceship on the frame
    frame[y:y + spaceship_height, x:x + spaceship_width] = spaceship_img


# Function to draw enemies (Failing grades) using the image
def draw_enemies(frame, enemies):
    for enemy in enemies:
        # Ensure that the enemy coordinates are within bounds of the frame
        if enemy[1] + 50 > frame.shape[0]:  # Check if enemy y is out of the frame's vertical bounds
            enemy[1] = frame.shape[0] - 50  # Adjust enemy y to be within bounds
        if enemy[0] + 50 > frame.shape[1]:  
            enemy[0] = frame.shape[1] - 50 
        if enemy[0] < 0:
            enemy[0] = 0
        if enemy[1] < 0:
            enemy[1] = 0 
        
        # Randomly choose which enemy image to use
        enemy_img = enemy_img1 if random.random() > 0.5 else enemy_img2
        
        # Now draw the enemy on the frame
        frame[enemy[1]:enemy[1] + 50, enemy[0]:enemy[0] + 50] = enemy_img


def draw_boosters(frame, boosters):
    for booster in boosters:
        # Ensure that the booster coordinates are within bounds of the frame
        if booster[1] + 30 > frame.shape[0]:  # Check if booster y is out of the frame's vertical bounds
            booster[1] = frame.shape[0] - 30  # Adjust booster y to be within bounds
        if booster[0] + 30 > frame.shape[1]:  # Check if booster x is out of the frame's horizontal bounds
            booster[0] = frame.shape[1] - 30  # Adjust booster x to be within bounds
        if booster[0] < 0:
            booster[0] = 0  # Prevent x from going out of the left bound
        if booster[1] < 0:
            booster[1] = 0  # Prevent y from going out of the top bound
        
        # Now draw the booster on the frame
        frame[booster[1]:booster[1] + 30, booster[0]:booster[0] + 30] = booster_img

# Function to move enemies (Failing grades)
def move_enemies(enemies):
    new_enemies = []
    global game_over  # Track game over condition
    for enemy in enemies:
        enemy[1] += enemy_speed  # Move enemy down
        if enemy[1] > 600:  # If enemy reaches bottom, it goes off screen
            continue
        new_enemies.append(enemy)
        # Check collision with spaceship
        if spaceship_x < enemy[0] < spaceship_x + spaceship_width and spaceship_y < enemy[1] < spaceship_y + spaceship_height:
            game_over = True  # Game over if spaceship collides with enemy
            return []  # Return an empty list to stop the game
    return new_enemies  # Return the updated list of enemies

# Function to move boosters (Passing grades)
def move_boosters(boosters):
    new_boosters = []
    global score, speed_boost, boost_duration
    for booster in boosters:
        booster[1] += enemy_speed  # Move booster down
        if booster[1] > 600:  # If booster reaches bottom, it goes off screen
            continue
        new_boosters.append(booster)
        # Check collection with spaceship
        if spaceship_x < booster[0] < spaceship_x + spaceship_width and spaceship_y < booster[1] < spaceship_y + spaceship_height:
            score += 10  # Increase score when booster is collected
            speed_boost = True  # Activate speed boost
            boost_duration = 100  # Boost lasts for 100 frames
            new_boosters.remove(booster)
    return new_boosters

# Main game loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip frame for mirror effect
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the Haar Cascade Classifier
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Get the center of the face to move spaceship
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        # Move spaceship based on face position
        spaceship_x = face_center_x - spaceship_width // 2
        spaceship_y = face_center_y - spaceship_height // 2

    # If speed boost is active, move spaceship faster
    if speed_boost:
        enemy_speed = 5  # Increase speed during boost
        boost_duration -= 1
        if boost_duration <= 0:
            speed_boost = False
            enemy_speed = 3  # Reset speed back to normal

    # Randomly add new enemies (Failing grades)
    if random.random() < 0.03:  # 3% chance of new failing grade
        enemies.append([random.randint(50, 550), -enemy_radius])

    # Randomly add new boosters (Passing grades)
    if random.random() < 0.05:  # 5% chance of new passing grade
        boosters.append([random.randint(50, 550), -15])

    # Move enemies and boosters
    enemies = move_enemies(enemies)
    boosters = move_boosters(boosters)

    # Draw enemies (Failing grades)
    draw_enemies(frame, enemies)

    # Draw boosters (Passing grades)
    draw_boosters(frame, boosters)

    # Draw spaceship
    draw_spaceship(frame, spaceship_x, spaceship_y)

    # Display score
    cv2.putText(frame, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Check for game over (collision with failing grades)
    if game_over:
        cv2.putText(frame, "GAME OVER", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.putText(frame, f"Final Score: {score}", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Grade Escape Game', frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):  # Wait for 'q' to quit
            break

    # Display the frame
    cv2.imshow('Grade Escape Game', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()