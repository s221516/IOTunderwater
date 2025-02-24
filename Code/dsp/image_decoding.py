# import required libraries
import cv2
import numpy as np

# load the input image
img = cv2.imread("Code\dsp\data\Original_Doge_meme.jpg")

# convert the input image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply thresholding to convert grayscale to binary image
ret, thresh = cv2.threshold(gray,175,255,0)

def convert_binary_image_to_bits(binary_image):
    bits = []
    for row in binary_image:
        for pixel in row:
            if pixel == 255:
                bits.append(1)
            else:
                bits.append(0)
    return bits

bits = convert_binary_image_to_bits(thresh)

width = len(thresh[0])
height = len(thresh)
print(width, height)

picture_in_binary = (''.join(str(x) for x in bits))
# Open a new file in write mode
with open("Code\dsp\picture_in_binary.py", "w") as file:
    # Write the variable name and the bit string to the file
    file.write(f'picture_in_binary = "{picture_in_binary}"')
    file.close()


def convert_bits_to_image(bit_string):
    # Initialize a NumPy array with zeros
    pixels = np.zeros((height, width), dtype=np.uint8)

    # Fill the array with 255 for '1' bits
    for i in range(height):
        for j in range(width):
            bit = bit_string[i * width + j]
            if bit == '1':
                pixels[i, j] = 255
     
    return pixels

pixels = convert_bits_to_image(picture_in_binary)

cv2.imwrite('Code\dsp\image_decoded.jpg', pixels)   
cv2.imshow("image", pixels)
cv2.waitKey()