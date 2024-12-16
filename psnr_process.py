import cv2
import numpy as np

def calculate_psnr(img1_path, img2_path):
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None:
        raise ValueError(f"Input image not found or invalid: {img1_path}")
    if img2 is None:
        raise ValueError(f"Output image not found or invalid: {img2_path}")

    # Resize output image to match input image dimensions
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Convert to grayscale if necessary
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2_resized.shape) == 3:
        img2_resized = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)

    # Calculate MSE
    mse = np.mean((img1 - img2_resized) ** 2)

    # Avoid division by zero
    if mse == 0:
        return float('inf')

    # Calculate PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Example usage
input_image_path = 'inputs/IMG_0079.png'
output_image_path = 'results/IMG_0079_out.png'

try:
    psnr_value = calculate_psnr(input_image_path, output_image_path)
    print(f"PSNR: {psnr_value} dB")
except ValueError as e:
    print(e)

