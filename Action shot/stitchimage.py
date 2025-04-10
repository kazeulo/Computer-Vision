import cv2
import numpy as np
import os

FEATURE_DETECTOR = "SIFT"  
RESIZE_WIDTH = 800

def resize_image(image, width):
    scale_factor = width / image.shape[1]
    return cv2.resize(image, (width, int(image.shape[0] * scale_factor)))

def detect_and_match_features(img1, img2, method="SIFT"):
    detector = cv2.SIFT_create(nfeatures=15000) if method == "SIFT" else cv2.ORB_create(nfeatures=15000)

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

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)
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

    # Blend the images using alpha blending
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

def final_crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    non_zero_coords = cv2.findNonZero(gray)
    
    # Get the bounding box of the non-zero area
    x, y, w, h = cv2.boundingRect(non_zero_coords)
        
    # Remove black spaces and crop
    cropped_image = image[y:y+h, x:x+w]
    cropped_image = cropped_image[20:-20, 15:-15]

    return cropped_image

def stitch_images(images):
    primary_img = images[0]

    for i in range(1, len(images)):
        print(f"Stitching image {i + 1}...")

        kp1, kp2, matches = detect_and_match_features(primary_img, images[i], FEATURE_DETECTOR)
        H = find_homography(kp1, kp2, matches)

        primary_img = warp_and_blend(primary_img, images[i], H)

        final_image = final_crop(primary_img)
    return final_image

    # return primary_img

# Load and resize images
image_folder = "DATA"
image_files = sorted([os.path.join(image_folder, img) for img in os.listdir(image_folder)])
images = [resize_image(cv2.imread(img), RESIZE_WIDTH) for img in image_files if cv2.imread(img) is not None]

if len(images) < 2:
    print("Not enough images to stitch!")
else:
    stitched_image = stitch_images(images)
    cv2.imwrite("stitched_output.jpg", stitched_image)
    print("Stitching completed. Saved as 'stitched_output.jpg'")
    display_image = resize_image(stitched_image, 550) if stitched_image.shape[1] > 550 else stitched_image
    cv2.imshow("Final Panorama", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()