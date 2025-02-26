import numpy as np
import cv2
from tkinter import filedialog

def gaussian_blur(img, kernel_size, sigma):
    result = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    return result

def generate_gaussian_stack(img, kernel_size, sigma, stack_size):
    # Create an empty list to store Gaussian filtered images
    gaussian_stack = []
    gaussian_stack.append(img)
    temp_img = img.copy()
    
    #temp_img: The current image, which starts as a copy of the original image but becomes progressively more blurred in each iteration.
    #kernel_size: The size of the Gaussian kernel used for blurring.
    #sigma: The standard deviation of the Gaussian kernel, determining the amount of blurring.

    #Result: gina overwrite ang temp by a more blurred version of the original image .
    for i in range(1, stack_size):
        temp_img1 = gaussian_blur(temp_img, kernel_size, sigma)
        gaussian_stack.append(temp_img1)
    return gaussian_stack


# Iteratively convolving the Gaussian filter onto the previous image
def generate_laplacian_stack(img, kernel_size, sigma, stack_size):
    gaussian_pyramid = generate_gaussian_stack(img, kernel_size, sigma, stack_size) # create a Gaussian stack of the input image
    laplacian_result = []

    # get difference between two consecutive images in the Gaussian stack (gaussian_pyramid[i] and gaussian_pyramid[i + 1]) 
    # copy() -- create copies of the images to ensure na unchanged yung Gaussian stack 
 
    for i in range(0, stack_size - 1):
        temp = gaussian_pyramid[i].copy() - gaussian_pyramid[i + 1].copy() 
        laplacian_result.append(temp)
    laplacian_result.append(gaussian_pyramid[-1].copy()) #last element in the Laplacian stack is the last image of the Gaussian stack.
    return laplacian_result

#generate a series of blurred masks
def generate_masks(mask, kernel_size, sigma, size):
    blurred_mask_result = []
    blurred_mask = gaussian_blur(mask, kernel_size, sigma)
    blurred_mask_result.append(blurred_mask)

    #generates additional blurred masks. 
    for i in range(size):
        blurred_mask = gaussian_blur(blurred_mask, kernel_size, sigma) #apply gaussian blur to the previously generated mask (blurred_mask)
        blurred_mask_result.append(blurred_mask) #append the resulting blurred mask to results
    return blurred_mask_result

#computes the final blended image using the images from the blended stack by summing up all levels of the Laplacian stack.
def blend_images(img1, img2, masks, kernel_size, sigma, stack_size):
    result = np.zeros_like(img1, dtype=np.float32)
    
    for i in range(stack_size):
        # Generate Laplacian pyramid for both images
        laplacian_stack1 = generate_laplacian_stack(img1, kernel_size, sigma, stack_size)
        laplacian_stack2 = generate_laplacian_stack(img2, kernel_size, sigma, stack_size)
        
        # Blend images at each level of the Laplacian stck 
        # formula that i followed: final_img = laplacian stack imgA * Gaussian mask + (1-gaussian mask)* laplacian imgB (CS Berkley)
        blended_img = laplacian_stack1[i] * masks[i] + laplacian_stack2[i] * (1 - masks[i])
        
        # Add the blended image to the result
        result += blended_img
    
    return result


#linear transformation on the input image to scale and shift its values to the uint8 range [0, 255], suitable for image display and processing.
def convert_from_float32_to_uint8(img):
    min_val = np.min(img)
    scaled_img = (img - min_val) * (255 / (np.max(img) - min_val))
    return scaled_img.astype(np.uint8)

#allow user to pick images
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        return img
    return None

#display images 
def show_images(images, titles):
    for img, title in zip(images, titles):
        cv2.imshow(title, img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def main ():
    img1 = open_image()
    img2 = open_image()

    # image pre-processing: resizing the second image to match the dimensions of the first image 
                            # and converting both images to floating-point arrays 
    
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    img_float1 = (img1 / 255).astype('float32')
    img_float2 = (img2 / 255).astype('float32')

    kernel_size, sigma, n = 151, 20, 5
    img1_gaussian_stack = generate_gaussian_stack(img1, kernel_size, sigma, n)
    img2_gaussian_stack = generate_gaussian_stack(img2, kernel_size, sigma, n)

    show_images(img1_gaussian_stack, [f'Gaussian Level {i+1} - Image 1' for i in range(len(img1_gaussian_stack))])
    show_images(img2_gaussian_stack, [f'Gaussian Level {i+1} - Image 2' for i in range(len(img2_gaussian_stack))])

    img1_laplacian_stack = generate_laplacian_stack(img1, kernel_size, sigma, n)
    img2_laplacian_stack = generate_laplacian_stack(img2, kernel_size, sigma, n)

    show_images(img1_laplacian_stack, [f'Laplacian Level {i+1} - Image 1' for i in range(len(img1_laplacian_stack))])
    show_images(img2_laplacian_stack, [f'Laplacian Level {i+1} - Image 2' for i in range(len(img2_laplacian_stack))])

    
    # Blending images
    h, w = img1.shape[0], img1.shape[1]
    mask = np.zeros(img1.shape, dtype='float32') #initialized mask with zeroes, --> no blending initially.
    mask[:, 0:w//2] = 1 #sets the left half of the mask to 1, indicating full blending for that region
    #mask[0:w//4, :] = 1
    masks = generate_masks(mask, kernel_size, sigma, n) #generate series of masks for blending 
    blended_image = blend_images(img_float1, img_float2, masks, kernel_size, sigma, n)
    blended_image = convert_from_float32_to_uint8(blended_image) #ensure that the pixel values are in the correct range (0 to 255) 

    # Display the final blended image
    cv2.imshow("Blended Image", blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()