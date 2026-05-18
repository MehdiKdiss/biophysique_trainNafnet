import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def apply_classic_filter(image_path, output_path="classic_filter_comparison.png"):
    # 1. Load the image
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Read in BGR format (OpenCV default)
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return

    # Convert to RGB for Matplotlib display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Eliminate Noise (Denoising)
    # Bilateral Filter is great for removing noise while preserving edges
    # Arguments: src, d (diameter of pixel neighborhood), sigmaColor, sigmaSpace
    denoised_img = cv2.bilateralFilter(img_rgb, d=9, sigmaColor=75, sigmaSpace=75)
    
    # 3. Eliminate Blur (Sharpening) using Unsharp Masking
    # Gaussian Blur the denoised image
    gaussian_blur = cv2.GaussianBlur(denoised_img, (0, 0), 3.0)
    # Calculate the weighted sum of source and blur
    # Formula: constants * (original - blur) + original
    # This enhances the edges
    filtered_img = cv2.addWeighted(denoised_img, 1.5, gaussian_blur, -0.5, 0)
    
    # Alternatively, you can use a kernel for sharpening directly if the above is not strong enough
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # filtered_img = cv2.filter2D(denoised_img, -1, kernel)

    # 4. Display Results Side-by-Side
    plt.figure(figsize=(12, 6))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_rgb)
    plt.axis("off")

    # Filtered Image
    plt.subplot(1, 2, 2)
    plt.title("Filtered (Bilateral Filter + Unsharp Masking)")
    plt.imshow(filtered_img)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Comparison saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply a classical denoising and sharpening baseline to a blood-smear image."
    )
    parser.add_argument("image_path", help="Path to the input image.")
    parser.add_argument(
        "--output",
        default="classic_filter_comparison.png",
        help="Path where the comparison figure will be saved.",
    )
    args = parser.parse_args()
    apply_classic_filter(args.image_path, args.output)
