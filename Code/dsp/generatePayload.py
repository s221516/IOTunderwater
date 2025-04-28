import random
import scipy.signal as signal
import numpy as np
import config

def string_to_bits(s):
    # Convert a string to a list of bits (0 or 1)
    result = []
    for char in s:
        bits = bin(ord(char))[2:].zfill(8)  # Get 8-bit binary representation
        result.extend([int(b) for b in bits])
    return result

def bits_to_string(bits):
    # Convert a list of bits back to a string
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        byte_str = ''.join(str(b) for b in byte)
        chars.append(chr(int(byte_str, 2)))
    return ''.join(chars)

def generate_payload(size):
    # Ensure size is a multiple of 8
    if size % 8 != 0:
        size += 8 - (size % 8)  # Round up to next multiple of 8
        print(f"Size rounded up to {size} to match byte boundaries.")

    while True:
        # Generate a random string of printable characters (excluding space)
        characters = [chr(i) for i in range(33, 127)]  # Printable ASCII excluding space
        random_string = ''.join(random.choices(characters, k=(size // 8 + 5)))  # a bit longer

        # Convert string to bits
        bit_list = string_to_bits(random_string)

        # Take the first 'size' bits
        payload = bit_list[:size]

        # Check correlation with Barker 13
        correlation = signal.correlate(config.BINARY_BARKER, payload, mode='same')
        is_not_similar_to_BARKER_13 = np.all(correlation != 9)
        # check if correlation contains a 9
        if is_not_similar_to_BARKER_13:
            return bits_to_string(payload)  # Good payload

if __name__ == "__main__":
    size = 100  # Example size
    payload_bits = generate_payload(size)
    payload_string = bits_to_string(payload_bits)
    
    print("Generated Payload (bits):", payload_bits)
    print("Generated Payload (string):", payload_string)
    print("Payload Length (bits):", len(payload_bits))
    print("Payload Length (chars):", len(payload_string))
