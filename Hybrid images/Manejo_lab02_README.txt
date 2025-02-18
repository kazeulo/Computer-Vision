Manejo, Kzlyr Shaira
CMSC 174 Lab02 - Hybrid Image


High-pass filter parameters (for image1/left):
- Kernel size: 7x7
- Sigma: 10

Low-pass filter parameters (for image2/right):
- Kernel size: 7x7
- Sigma: 30

Mix-in Ratio: 0.25
- This parameter controls the blending of the two images after applying the high and low pass filters. The mix-in ratio of 0.25 means that 25% of the final hybrid image is formed by the high-pass filtered image1 (left), and the remaining 75% comes from the low-pass filtered image2(right)

Other comments:
- I experimented with different kernel sizes and sigma values for both filters. A larger sigma for the low-pass filter reduces the fine details of the image, making it smoother. While having a lower sigma value for high-pass filter suppresses the low-frequency details of the image, highlighting the fine details of the image.