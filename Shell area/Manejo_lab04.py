import cv2 
import numpy as np
import matplotlib.pyplot as plt

# Preprocess the image
def process_image(img):

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.equalizeHist(gray_image)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
    gray_image = clahe.apply(gray_image)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (13, 13), 13)

    # Apply binary thresholding
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  

    edged_image = cv2.Canny(binary_image, 50, 200)
    edged_image = cv2.morphologyEx(edged_image, cv2.MORPH_CLOSE, (3,3))

    return edged_image

# Conversion function from mm² to inch²
def mm_to_inches(area_mm2):
    return area_mm2 / (25.4 ** 2)

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
    
    # for visualization purposes
    processed_images = []
    titles = []

    # Iterate through the shells
    for idx, shell_img in enumerate(images):

        # Keep a copy of the original image for visualization
        original_image = shell_img.copy()
    
        processed_image = process_image(shell_img)

        # Get contours
        contours_list, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area, largest first
        contours_list = sorted(contours_list, key=cv2.contourArea, reverse=True)

        # draw contours
        cv2.drawContours(original_image, contours_list, 0, (255, 0, 255), 3) 
        cv2.drawContours(original_image, contours_list, 1, (0, 255, 0), 3)

        # find contour and area
        coin_contour = contours_list[1]
        shell_contour = contours_list[0]  

        # Find the diameter of the coin in pixels
        _, radius = cv2.minEnclosingCircle(coin_contour)
        coin_diameter_pixels = 2 * radius

        # Compute the area ratio based on the coin diameter
        # The scale factor
        area_ratio = (coin_diameter / coin_diameter_pixels) ** 2
        
        # Get area of shell in pixels
        shell_area = cv2.contourArea(shell_contour)

        # PRINTING OUPUT
        print(f"\nShell {idx + 1}")
        print("Number of Contours found = " + str(len(contours_list)))
        
        # Calculate the real area of the shell in mm²
        real_area_mm2 = shell_area * area_ratio
        real_area_mm2 = round(real_area_mm2, 2) 
        print(f"Shell area in mm²: {real_area_mm2} mm²")

        # Convert real area from mm² to inches²
        real_area_in2 = mm_to_inches(real_area_mm2)
        real_area_in2 = round(real_area_in2, 2)
        print(f"Shell area in inches²: {real_area_in2} in²")

        processed_images.append(original_image)
        titles.append(f"Shell {idx + 1}")

    show_images(processed_images, titles)

shell1 = cv2.imread('images/Shell001.png')
shell2 = cv2.imread('images/Shell002.png')
shell3 = cv2.imread('images/Shell003.png')

# reference coin
coin_diameter = 26.76 

# List of shell images
shell_images = [shell1, shell2, shell3]

compute_area(shell_images, coin_diameter)

cv2.destroyAllWindows()