import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to apply Gaussian blur
def apply_gaussian(image, levels, sigma):

    gaussian = []
    for i in range(levels):
        kernel_size = ((3*sigma)//2)*2+1

        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        gaussian.append(blurred_image)

        # Increase sigma for the next level
        sigma *= 2

    return gaussian

# Function to create Laplacian stack
def laplacian_effect(img, levels, sigma):
    laplacian = []

    # Apply Gaussian blur and create the Gaussian pyramid
    gaussian = apply_gaussian(img, levels + 1, sigma)

    for i in range(levels):
        laplacian_image = cv2.subtract(gaussian[i], gaussian[i + 1])
        laplacian.append(laplacian_image)

    return laplacian

# Function to blend two images (color version)
def blend_images_color(img1, img2, mask, levels, sigma):
    # Split the color images into channels (BGR in OpenCV)
    img1_b, img1_g, img1_r = cv2.split(img1)
    img2_b, img2_g, img2_r = cv2.split(img2)

    # Apply Gaussian and Laplacian for each channel
    gaussian1_b = apply_gaussian(img1_b, levels, sigma)
    gaussian2_b = apply_gaussian(img2_b, levels, sigma)
    gaussianMask_b = apply_gaussian(mask, levels, sigma)

    gaussian1_g = apply_gaussian(img1_g, levels, sigma)
    gaussian2_g = apply_gaussian(img2_g, levels, sigma)
    gaussianMask_g = apply_gaussian(mask, levels, sigma)

    gaussian1_r = apply_gaussian(img1_r, levels, sigma)
    gaussian2_r = apply_gaussian(img2_r, levels, sigma)
    gaussianMask_r = apply_gaussian(mask, levels, sigma)

    laplacian1_b = laplacian_effect(img1_b, levels, sigma)
    laplacian2_b = laplacian_effect(img2_b, levels, sigma)
    
    laplacian1_g = laplacian_effect(img1_g, levels, sigma)
    laplacian2_g = laplacian_effect(img2_g, levels, sigma)

    laplacian1_r = laplacian_effect(img1_r, levels, sigma)
    laplacian2_r = laplacian_effect(img2_r, levels, sigma)

    # Blend the channels
    laplacian_stack_b = []
    for i in range(levels):
        blended_laplacian = gaussianMask_b[i] * laplacian1_b[i] + (1 - gaussianMask_b[i]) * laplacian2_b[i]
        laplacian_stack_b.append(blended_laplacian)

    laplacian_stack_g = []
    for i in range(levels):
        blended_laplacian = gaussianMask_g[i] * laplacian1_g[i] + (1 - gaussianMask_g[i]) * laplacian2_g[i]
        laplacian_stack_g.append(blended_laplacian)

    laplacian_stack_r = []
    for i in range(levels):
        blended_laplacian = gaussianMask_r[i] * laplacian1_r[i] + (1 - gaussianMask_r[i]) * laplacian2_r[i]
        laplacian_stack_r.append(blended_laplacian)

    # Final Gaussian
    final_gaussian_b = gaussianMask_b[-1] * gaussian1_b[-1] + (1 - gaussianMask_b[-1]) * gaussian2_b[-1]
    final_gaussian_g = gaussianMask_g[-1] * gaussian1_g[-1] + (1 - gaussianMask_g[-1]) * gaussian2_g[-1]
    final_gaussian_r = gaussianMask_r[-1] * gaussian1_r[-1] + (1 - gaussianMask_r[-1]) * gaussian2_r[-1]

    output_b = final_gaussian_b
    output_g = final_gaussian_g
    output_r = final_gaussian_r

    for i in range(levels):
        output_b = cv2.add(laplacian_stack_b[i], output_b)
        output_g = cv2.add(laplacian_stack_g[i], output_g)
        output_r = cv2.add(laplacian_stack_r[i], output_r)

    # Merge the channels back together
    blended_image = cv2.merge([output_b, output_g, output_r])

    return blended_image

# Function to display the Gaussian stack
def display_gaussian_stack(img, levels, sigma):
    gaussian_stack = apply_gaussian(img, levels, sigma)

    # Display the Gaussian stack
    plt.figure(figsize=(12, 6))
    plt.title("Gaussian Stack")
    for i in range(levels):
        plt.subplot(1, levels, i + 1)
        plt.imshow(gaussian_stack[i], cmap='gray')
        plt.title(f'Gaussian Level {i+1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Function to display the Laplacian stack
def display_laplacian_stack(img, levels, sigma):
    laplacian_stack = laplacian_effect(img, levels, sigma)

    # Display the Laplacian stack
    plt.figure(figsize=(12, 6))
    plt.title("Laplacian Stack")
    for i in range(levels):
        plt.subplot(1, levels, i + 1)
        plt.imshow(laplacian_stack[i], cmap='gray', clim=(0, 0.1))
        plt.title(f'Laplacian Level {i+1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Load the images (as color images)
imgA = cv2.imread('images/CirunayManejo_lab03_a.png', cv2.IMREAD_COLOR)
imgB = cv2.imread('images/CirunayManejo_lab03_b.png', cv2.IMREAD_COLOR)
mask = cv2.imread('images/CirunayManejo_lab03_mask.png', cv2.IMREAD_GRAYSCALE)

# Convert imgA and imgB to uint8 before using cvtColor
imgA_uint8 = (imgA * 255).astype(np.uint8)
imgB_uint8 = (imgB * 255).astype(np.uint8)

# Convert the images from BGR to RGB for proper display with matplotlib
imgA_rgb = cv2.cvtColor(imgA_uint8, cv2.COLOR_BGR2RGB)
imgB_rgb = cv2.cvtColor(imgB_uint8, cv2.COLOR_BGR2RGB)

# Initialize parameters
levels = 6
sigma = 1

# Normalize images
imgA_rgb = imgA / 255.0
imgB_rgb = imgB / 255.0
mask = mask / 255.0

# Blend the color images
blended_image = blend_images_color(imgA_rgb, imgB_rgb, mask, levels, sigma)
blended_image = np.clip(blended_image, 0, 1) 
blended_image = (blended_image * 255).astype(np.uint8)

blended_rgb = cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB)

# Display the blended image
plt.imshow(blended_rgb)
plt.title("Blended Image")
plt.axis("off")
plt.show()

# Save the output as 
cv2.imwrite('CirunayManejo_Lab03_colored.png', blended_image)

cv2.waitKey(0)
cv2.destroyAllWindows()