import cv2
import numpy as np
import matplotlib.pyplot as plt

FEATURE_DETECTOR = "SIFT"  

def extract_frames(video_path, frame_numbers):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []
    frames = []     
    for frame_number in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            print(f"Error: Could not read frame {frame_number}")
    cap.release()
    return frames

def crop_image(frame, start_x, end_x, crop_height):
    frame_height, frame_width, _ = frame.shape
    start_x = max(0, start_x)
    end_x = min(frame_width, end_x)
    cropped_frame = frame[0:crop_height, start_x:end_x]
    return cropped_frame

def detect_and_match_features(img1, img2):
    detector = cv2.SIFT_create(nfeatures=15000)

    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = detector.detectAndCompute(gray_img1, None)
    kp2, des2 = detector.detectAndCompute(gray_img2, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    flann_matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in flann_matches if m.distance < 0.85 * n.distance]

    print(f"Found {len(good_matches)} good matches.")

    return kp1, kp2, good_matches

def find_homography(kp1, kp2, matches):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    return H

def feather_blend(img1, img2):

    # Create binary masks
    mask1 = (np.sum(img1, axis=2) > 0).astype(np.uint8) * 255
    mask2 = (np.sum(img2, axis=2) > 0).astype(np.uint8) * 255

    # Perform distance perform
    dist_transform1 = cv2.distanceTransform(255 - mask1, cv2.DIST_L2, 5)
    dist_transform2 = cv2.distanceTransform(255 - mask2, cv2.DIST_L2, 5)
    
    # Apply Gaussian blur to smooth the alpha blending
    dist_transform1 = cv2.GaussianBlur(dist_transform1, (5, 5), 5)
    dist_transform2 = cv2.GaussianBlur(dist_transform2, (5, 5), 5)
    
    # Calculate the blending mask
    total_dist = dist_transform1 + dist_transform2 + 1e-6
    alpha = dist_transform2 / total_dist
    alpha = np.expand_dims(alpha, axis=2)

    # Blend the images using the alpha blending
    blended = (alpha * img1 + (1 - alpha) * img2).astype(np.uint8)
    return blended

def warp_and_blend(base_img, new_img, H):
    h1, w1 = base_img.shape[:2]

    corners1 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32)
    corners2 = cv2.perspectiveTransform(np.array([corners1]), H)[0]

    x_min = int(min(0, corners2[:, 0].min()))
    y_min = int(min(0, corners2[:, 1].min()))
    x_max = int(max(w1, corners2[:, 0].max()))
    y_max = int(max(h1, corners2[:, 1].max()))

    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)

    base_warped = cv2.warpPerspective(base_img, translation @ H, (x_max - x_min, y_max - y_min))
    new_warped = cv2.warpPerspective(new_img, translation, (x_max - x_min, y_max - y_min))

    return feather_blend(base_warped, new_warped)

def crop_black_background(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours of the image
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get coordinates of the bounding box 
    x, y, w, h = cv2.boundingRect(contours[0])

    cropped_image = image[y:y + h, x:x + w]
    cropped_image = cropped_image[20:-20, 15:-15]
    
    return cropped_image

def stitch_images(images):
    primary_img = images[0]
    for i in range(1, len(images)):
        print(f"Stitching image {i + 1}...")
        kp1, kp2, matches = detect_and_match_features(primary_img, images[i])
        H = find_homography(kp1, kp2, matches)
        primary_img = warp_and_blend(primary_img, images[i], H)
    
    # Fill canvas and crop the image
    final_image = crop_black_background(primary_img)
    return final_image

def process_video(video_path, frame_numbers, crop_positions):
    frames = extract_frames(video_path, frame_numbers)
    
    crop_height = frames[0].shape[0]
    
    cropped_frames = []
    for i, frame in enumerate(frames):
        start_x, end_x = crop_positions[i]
        cropped_frame = crop_image(frame, start_x, end_x, crop_height)
        cropped_frames.append(cropped_frame)
    
    stitched_image = stitch_images(cropped_frames)

    return stitched_image

video_path = "video/kick.mp4"
frame_numbers = [1, 36, 48, 58, 70, 95]
crop_positions = [
    (1, 2200), 
    (410, 640), 
    (615, 920), 
    (870, 1150), 
    (1120, 1350), 
    (1285, 2200)
]

stitched_image = process_video(video_path, frame_numbers, crop_positions)

cv2.imwrite('Action_shot.jpg', stitched_image)

plt.figure(figsize=(12, 12))
plt.imshow(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))
plt.title('Stitched Video Frames')
plt.axis('off')
plt.show()