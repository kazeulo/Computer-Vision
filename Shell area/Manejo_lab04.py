import cv2 
import numpy as np
import matplotlib.pyplot as plt

#  Function for resizing the image
def resize_img(img, target_height):

    height, width = img.shape[:2]  
    scale_factor = target_height / height  
    new_width = int(width * scale_factor)  

    resized_img = cv2.resize(img, (new_width, target_height))
    return resized_img

# Conversion function from mm² to inch²
def mm_to_inches(area_mm2):
    return area_mm2 / (25.4 ** 2)

# Function to check if the contour is circular
def is_circular(contour):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)

    return (4 * np.pi * area) / (perimeter ** 2) > 0.85

# Function for showing the detected edges in the image
def show_images(images, titles):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(15, 5))
    
    for i in range(n):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(titles[i])
        axes[i].axis('off') 
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def compute_area(images, coin_diameter):
    processed_images = []
    titles = []

    # Iterate through the shells
    for idx, shell_img in enumerate(images):
        
        # Convert to grayscale
        gray_image = cv2.cvtColor(shell_img, cv2.COLOR_BGR2GRAY)

        # Apply equalization on the histogram of the image
        gray_image = cv2.equalizeHist(gray_image)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
        gray_image = clahe.apply(gray_image)

        # Apply Gaussian blur
        blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 7)

        # Apply binary thresholding
        _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Perform edge detection
        edged = cv2.Canny(binary_image, 50, 200)
    
        # Refinement using close morphology
        edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, (3,3))

        # Get contours
        contours_list, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_list = [contour for contour in contours_list if cv2.contourArea(contour) > 100]

        # Sort contours by area, largest first
        contours_list = sorted(contours_list, key=cv2.contourArea, reverse=True)

        print(f"\nShell {idx + 1}")
        print("Number of Contours found = " + str(len(contours_list)))

        if not contours_list:
            print("No valid contours found.")
            continue  

        # Keep a copy of the edged image for visualization
        detected_edges = gray_image.copy()
        cv2.drawContours(detected_edges, contours_list, -1, (0, 255, 0), 3) 

        # find the coin
        coin_contour = None
        for contour in contours_list:
            if is_circular(contour):
                coin_contour = contour
                break 

        if coin_contour is not None:
            # The largest contour is the shell
            max_contour = contours_list[0]  
            coin_area = cv2.contourArea(coin_contour)
            shell_area = cv2.contourArea(max_contour)

            print("Coin area in pixels:", coin_area)
            print("Shell area in pixels:", shell_area)

            # Find the diameter of the coin in pixels
            (x, y), radius = cv2.minEnclosingCircle(coin_contour)
            diameter_pixels = 2 * radius

            # Compute the area ratio based on the coin diameter
            area_ratio = (coin_diameter / diameter_pixels) ** 2

            # Calculate the real area of the shell in mm²
            real_area_mm2 = shell_area * area_ratio
            print("Shell area in mm²:", real_area_mm2)

            # Convert real area from mm² to inches²
            real_area_in2 = mm_to_inches(real_area_mm2)
            print("Shell area in inches²:", real_area_in2)

        else:
            print("No circular contour found for the coin/no coin detected.")

        # Add the image and its title to the lists
        processed_images.append(detected_edges)
        titles.append(f"Shell {idx + 1}")

    show_images(processed_images, titles)

shell1 = cv2.imread('images/Shell001.png')
shell2 = cv2.imread('images/Shell002.png')
shell3 = cv2.imread('images/Shell003.png')

# Use the height of shell image 1 to resize the other images
target_height = shell1.shape[1]

# Resize image 2 and 3 for consistency
shell2 = resize_img(cv2.imread('images/Shell002.png'), target_height)
shell3 = resize_img(cv2.imread('images/Shell003.png'), target_height)

# reference coin
coin_diameter = 26.76 

# List of shell images
shell_images = [shell1, shell2, shell3]

compute_area(shell_images, coin_diameter)

cv2.destroyAllWindows()
