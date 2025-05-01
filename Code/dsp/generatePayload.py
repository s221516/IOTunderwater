import random
import scipy.signal as signal
import numpy as np
import config
import csv
import os

BARKER_13 = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]

def save_payload_combination_to_the_banned_combinations(string, bits):
    """Save string and its bit representation to CSV when correlation is 9"""
    filename = "Code/dsp/data/correlation_matches.csv"
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['String', 'Bits'])  # Write header if new file
        writer.writerow([string, bits])

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
        # print(f"Byte: {byte_str} -> Char: {chars[-1]}")  # Debugging line
    return ''.join(chars)

def generate_payload(size, max_correlation=6):
    # Ensure size is a multiple of 8
    if size % 8 != 0:
        size += 8 - (size % 8)  # Round up to next multiple of 8
        # print(f"Size rounded up to {size} to match byte boundaries.")

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
        is_not_similar_to_BARKER_13 = max(correlation) == max_correlation
        # check if correlation contains a 9
        if is_not_similar_to_BARKER_13:
            return bits_to_string(payload)  # Good payload

if __name__ == "__main__":
    size = 100  # Example size
    payload_bits = generate_payload(size, 3)
    # payload_string = bits_to_string(payload_bits)
    # example of payload with barker corr < 3 : `u%.1@@C829fS
    print("Generated Payload (bits):", payload_bits)
    # print("Generated Payload (string):", payload_string)
    print("Payload Length (bits):", len(payload_bits))
    # print("Payload Length (chars):", len(payload_string))
    a = max(signal.correlate(config.BINARY_BARKER, string_to_bits(payload_bits), mode='same'))
    print(f"Max correlation with Barker 13: {a}")
    unique_payload_correlations_size_100 = []
    nums = np.arange(2, 10, 1)
    for i in range(len(nums)):
        payload = generate_payload(size, nums[i])
        correlation = signal.correlate(config.BINARY_BARKER, string_to_bits(payload), mode='same')
        max_correlation = max(correlation)
        print(f"Max correlation with Barker 13: {max_correlation}")
        unique_payload_correlations_size_100.append(payload)
        
    print(unique_payload_correlations_size_100)
    ["]3MH'@@H9&e6W",
     '}VvF*E@9>-go*',
     '+,M4J1ABraRJ&',
     'i3aw,*X@j&y;y',
     '~7,w]@s,V+{2Y',
     ']_TzaWWF+Exg;',
     'Oi67/(~V8]w,x',
     'N(#-c~nC(^v>A',
    ]
    unique_payloads_dict = {
        "2" : "]3MH'@@H9&e6W", 
        "3" : '}VvF*E@9>-go*', 
        "4" : '+,M4J1ABraRJ&',
        "5" : 'i3aw,*X@j&y;y', 
        "6" : '~7,w]@s,V+{2Y', 
        "7" : ']_TzaWWF+Exg;', 
        "8" : 'Oi67/(~V8]w,x', 
        "9" : 'N(#-c~nC(^v>A'
    }