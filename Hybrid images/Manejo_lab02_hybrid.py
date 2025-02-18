import sys
import cv2
import numpy as np

def cross_correlation(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    # TODO-BLOCK-BEGIN

    kernel_height, kernel_width = kernel.shape
    image_height, image_width = img.shape

    # Padding the image with zeros
    image_padded = np.zeros((image_height + kernel_height - 1, image_width + kernel_width - 1))
    image_padded[kernel_height//2:image_height + kernel_height//2, kernel_width//2:image_width + kernel_width//2] = img

    # Prepare the output image
    output = np.zeros_like(img)

    # Perform the cross-correlation
    for i in range(image_height):
        for j in range(image_width):
            window = image_padded[i:i+kernel_height, j:j+kernel_width]
            output[i, j] = np.sum(window * kernel)
    
    return output

    # TODO-BLOCK-END

def convolution(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    # TODO-BLOCK-BEGIN

    flipped_kernel = np.flip(kernel)
    output = cross_correlation(img, flipped_kernel)

    return output

    # TODO-BLOCK-END

def gaussian_kernel(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. 

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''

    # TODO-BLOCK-BEGIN

    center = int(width / 2)
    kernel = np.zeros((height, width))
    
    for i in range(height):
        for j in range(width):
            kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2))

    output = kernel / np.sum(kernel)
    
    return output

    # TODO-BLOCK-END

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    # TODO-BLOCK-BEGIN

    kernel = gaussian_kernel(sigma, size, size)
    output = cross_correlation(img, kernel)

    cv2.imshow('Low pass Image', output)

    return output

    # TODO-BLOCK-END

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)

    '''
    # TODO-BLOCK-BEGIN

    low_pass_img = low_pass(img, sigma, size)
    output = img - low_pass_img

    cv2.imshow('High pass Image', output)

    return output

    # TODO-BLOCK-END

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2, high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''

    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 = np.copy(img1)
    img2 = np.copy(img2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    output = (hybrid_img * 255).clip(0, 255).astype(np.uint8)
    
    return output

# IMPLEMENTATION

# import image
image1 = cv2.imread('Manejo_lab02_left.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('Manejo_lab02_right.png', cv2.IMREAD_GRAYSCALE)

cv2.imshow('original Image', image2)

# parameters for creating hybrid image
sigma1 = 10
size1 = 7
high_low1 = 'high'      # Apply high-pass filter to image1/left

sigma2 = 30
size2 = 7           
high_low2 = 'low'       # Apply low-pass filter to image2/right

mixin_ratio = 0.25      # Blending ratio between the two images
scale_factor = 2.5

# Create the hybrid image
hybrid_image = create_hybrid_image(
    img1=image1, 
    img2=image2, 
    sigma1=sigma1, size1=size1, high_low1=high_low1,  
    sigma2=sigma2, size2=size2, high_low2=high_low2, 
    mixin_ratio=mixin_ratio, scale_factor=scale_factor
)

# Display result
cv2.imshow('Hybrid Image', hybrid_image)
cv2.imwrite('Manejo_lab02_hybrid2.jpg', hybrid_image)  
cv2.waitKey(0)
cv2.destroyAllWindows()