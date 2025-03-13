# import required libraries
import cv2
import numpy as np

from Code.dsp.config import PATH_TO_PICTURE, THRESHOLD_BINARY_VAL

# load the input image
img = cv2.imread(PATH_TO_PICTURE)

# convert the input image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply thresholding to convert grayscale to binary image
_, dimensions = cv2.threshold(gray, THRESHOLD_BINARY_VAL, 255, 0)

def convert_binary_image_to_bits(binary_image):
    bits = []
    for row in binary_image:
        for pixel in row:
            if pixel == 255:
                bits.append(1)
            else:
                bits.append(0)
    
    return bits

bits = convert_binary_image_to_bits(dimensions)
width = len(dimensions[0])
height = len(dimensions)

picture_in_binary = (''.join(str(x) for x in bits))

# Open a new file in write mode
with open("Code\dsp\picture_in_binary.txt", "w") as file:
    file.write(picture_in_binary)
    print("Saved picture as binary file...")
    file.close()


def convert_bits_to_image(bit_string):
    # Initialize a NumPy array with zeros
    # 0 = black
    pixels = np.zeros((height, width), dtype=np.uint8)
    
    # print(len(bit_string))
    # print(height * width)

    if (len(bit_string) < height * width):
        diff = height * width - len(bit_string)
        bit_string += '0' * diff
    
    # Fill the array with 255 for '1' bits
    for i in range(height):
        for j in range(width):
            bit = bit_string[i * width + j]
            # 1 = white
            if bit == '1' or bit == "L" or bit == "@":
                pixels[i, j] = 255
            
                
    return pixels

     
def show_picture(pixels):
    cv2.imwrite('Code\dsp\data\image_decoded.jpg', pixels)   
    cv2.imshow("image", pixels)
    cv2.waitKey()