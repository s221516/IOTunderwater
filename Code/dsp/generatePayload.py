import random
import scipy.signal as signal
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
    return ''.join(chars)

def correlate(bits, pattern):
    correlation = signal.correlate(bits, pattern, mode="valid")
    # print("Correlate: ", correlation)
    # find max value of the correlation, which if 9 should be added to the banned combinations
    max_correlation = max(correlation)
    # print("Max corr: ", max_correlation)
    return max_correlation

def generate_payload(size, custom_chars = None):
    # Ensure size is a multiple of 8
    if size % 8 != 0:
        size += 8 - (size % 8)  # Round up to next multiple of 8
        # print(f"Size rounded up to {size} to match byte boundaries.")

    while True:
        # Use custom characters if provided, otherwise use printable ASCII
        if custom_chars:
            characters = custom_chars
        else:
            characters = [chr(i) for i in range(33, 127)]  # Printable ASCII excluding space

        random_string = ''.join(random.choices(characters, k=(size // 8 + 5)))  # a bit longer

        # Convert string to bits
        bit_list = string_to_bits(random_string)

        # Take the first 'size' bits
        payload = bit_list[:size]

        # Check correlation with Barker 13
        if correlate(payload, BARKER_13) != 9:
            # now just returns a string representation to be used in main when generating the messages for testing
            payload_as_string = bits_to_string(payload)
            # print(payload_as_string)
            return payload_as_string
        else: 
            matching_string = random_string[:(size // 8)]  # Get just the used portion
            save_payload_combination_to_the_banned_combinations(matching_string, payload)
            print(f"Payload with correlation {correlate(payload, BARKER_13)} found. Saved to CSV, retrying...")

if __name__ == "__main__":
    size_in_bits = 11  # Example size
    payload_bits = generate_payload(size_in_bits)
    payload_string = bits_to_string(payload_bits)
    
    print("Generated Payload (bits):", payload_bits)
    print("Generated Payload (string):", payload_string)
    print("Payload Length (bits):", len(payload_bits))
    print("Payload Length (chars):", len(payload_string))

    # # Test with custom characters
    # custom_chars = ["Hello_world"]
    # payload_bits = generate_payload(size, custom_chars)
    # payload_string = bits_to_string(payload_bits)
    # print("\nCustom character set:", custom_chars)
    # print("Generated Payload (bits):", payload_bits)
    # print("Generated Payload (string):", payload_string)
    # print("Payload Length (bits):", len(payload_bits))
    # print("Payload Length (chars):", len(payload_string))
