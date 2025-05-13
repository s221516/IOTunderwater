from PIL import Image
import os

# Load the image
image_path = "dtu_logo.png"  # Use your image path
img = Image.open(image_path).convert("L")  # Convert to grayscale

# Apply a threshold to binarize
threshold = 128
bw_img = img.point(lambda x: 255 if x > threshold else 0, mode='1')  # Pure black & white (1-bit)

# Save as 1-bit PNG
bw_path = "dtu_binary_bw.png"
bw_img.save(bw_path, format="PNG", optimize=True)

# File size
compressed_size = os.path.getsize(bw_path)
print(f"1-bit Binary PNG size: {compressed_size} bytes")
import pywt
import matplotlib.pyplot as plt
import numpy as np

bw_binary = Image.open(bw_path).convert("L")  # Load the binary image

# Convert to float image for wavelet
gray_float = bw_binary.convert("F")

# Apply 2D Discrete Wavelet Transform
coeffs = pywt.dwt2(gray_float, 'haar')
cA, (cH, cV, cD) = coeffs

# Show approximation coefficients (lower resolution image)
plt.imshow(cA, cmap='gray')
plt.title('Wavelet Approximation')
plt.show()
