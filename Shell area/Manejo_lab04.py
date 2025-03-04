import cv2
import numpy as np

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
    return circularity > 0.7 

def compute_area(images, coin_diameter):

    # iterate through the shells
    for idx, shell_img in enumerate(images):
        
        # Convert to grayscale
        gray_image = cv2.cvtColor(shell_img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred_image = cv2.GaussianBlur(gray_image, (5,5), 5)

        # Apply binary thresholding
        _, binary_image = cv2.threshold(blurred_image, 100, 255, cv2.THRESH_BINARY)

        # Perform edge detection
        edged = cv2.Canny(binary_image, 50, 150)

        # Apply dilation and erosion
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)

        # Get contours
        contours_list, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
            # Find the most circular contour
            if is_circular(contour):  
                coin_contour = contour
                break

        if coin_contour is not None:
            # now get the largest one as the shell
            # Largest contour is the shell
            max_contour = contours_list[0]  
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
            print("Shell area in mm²: ", real_area_mm2)

            # Convert real area from mm² to inches²
            real_area_in2 = mm_to_inches(real_area_mm2)

            print("Shell area in inches²: ", real_area_in2)

        else:
            print("No circular contour found for the coin/no coin detected.")

# read shell images
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