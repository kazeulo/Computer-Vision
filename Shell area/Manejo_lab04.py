import cv2
import numpy as np

# Function for converting image to grayscale
def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Function for applying gaussian blur
def gaussian_blur(img, kernel, sigma):
    return cv2.GaussianBlur(img, kernel, sigma)

# Function for edge detection 
def edge_detection(img, t_lower, t_upper):
    return cv2.Canny(img, t_lower, t_upper)

# Function for binary thresholding 
def binary_threshold(img):
    return cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

# Function to get contours
def find_contours(img):
    return cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Function to apply dilation
def dilation(img):
    return cv2.dilate(img, None, iterations=3)

# Function to apply erosion
def erosion(img):
    return cv2.erode(img, None, iterations=2)

# Conversion function from mm² to inch²
def mm_to_inches(area_mm2):
    return area_mm2 / (25.4 ** 2)

# Function to check if the contour is circular
def is_circular(contour):
    # Get the contour's perimeter and area
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    area = cv2.contourArea(contour)
    
    # Circularity = (4 * pi * area) / (perimeter^2)
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    
    # If the circularity is close to 1, it is roughly a circle
    return circularity > 0.7  # Adjust threshold as necessary

def compute_area(images, coin_diameter):
    for idx, shell_image in enumerate(images):
        # Convert to grayscale
        gray_image = to_grayscale(shell_image)

        # Apply Gaussian blur
        blurred_image = gaussian_blur(gray_image, (11, 11), 8)

        # Apply binary thresholding
        _, binary_image = binary_threshold(blurred_image)

        # Perform edge detection
        edged = edge_detection(binary_image, 60, 150)

        # Apply dilation and erosion
        edged = dilation(edged)
        edged = erosion(edged)

        # Get contours
        contours_list, _ = find_contours(edged)

        # Filter out small or irrelevant contours
        contours_list = [contour for contour in contours_list if cv2.contourArea(contour) > 100]

        # Sort contours by area, largest first
        contours_list = sorted(contours_list, key=cv2.contourArea, reverse=True)

        # Print number of contours found
        print(f"\nShell {idx + 1}")
        print("Number of Contours found = " + str(len(contours_list)))

        # Draw contours on the edged image
        cv2.drawContours(edged, contours_list, -1, (0, 255, 0), 3)

        # Try to find the coin contour
        coin_contour = None
        for contour in contours_list:
            if is_circular(contour):  # Find the most circular contour
                coin_contour = contour
                break

        if coin_contour is not None:
            # Coin contour found, now get the largest one as the shell
            max_contour = contours_list[0]  # Largest contour is the shell
            coin_area = cv2.contourArea(coin_contour)
            shell_area = cv2.contourArea(max_contour)

            print("Coin area in pixels: ", coin_area)
            print("Shell area in pixels: ", shell_area)

            # Find the diameter of the shell in pixels
            (x, y), radius = cv2.minEnclosingCircle(coin_contour)
            diameter_pixels = 2 * radius

            # Compute the area ratio based on the coin diameter
            area_ratio = (coin_diameter / diameter_pixels) ** 2

            # Calculate the real area of the shell in mm²
            real_area_mm2 = shell_area * area_ratio
            print("Real area of shell in mm²: ", real_area_mm2)

            # Convert real area from mm² to inches²
            real_area_in2 = mm_to_inches(real_area_mm2)
            print("Real area of shell in inches²: ", real_area_in2)
        else:
            print("No circular contour found for the coin")

# Example usage with shell images
shell1 = cv2.imread('images/Shell001.png')
shell2 = cv2.imread('images/Shell002.png')
shell3 = cv2.imread('images/Shell003.png')

coin_diameter = 26.76  # Coin diameter in mm

# List of shell images
shell_images = [shell1, shell2, shell3]

# Compute the area for each shell image
compute_area(shell_images, coin_diameter)

cv2.waitKey(0) 
cv2.destroyAllWindows()