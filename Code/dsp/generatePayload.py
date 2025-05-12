import random
import networkx as nx
import commpy.channelcoding.convcode as cc
import scipy.signal as signal
import numpy as np
import config
import csv
import os

def get_trellis(constraint_length=3, generators=None, rate=2):
    """Get trellis for different generator polynomials and rates"""
    if generators is None:
        if rate == 2:
            generators = [[5, 7]]      # Rate 1/2
        elif rate == 3:
            generators = [[7, 3, 5]]   # Rate 1/3
        elif rate == 4:
            generators = [[7, 7, 5, 5]]  # Rate 1/4
        elif rate == "2/3":
            # Rate 2/3: 2 inputs, 3 outputs
            # These are example polynomials for rate 2/3
            generators = [[7, 7, 5],    # For first input bit
                        [7, 5, 7]]     # For second input bit
        else:
            raise ValueError(f"Unsupported default rate: {rate}")
    
    generator_matrix = np.array(generators)
    memory = np.array([constraint_length - 1] * len(generators))  # Memory for each input
    return cc.Trellis(memory, generator_matrix)

def conv_encode(message_bits, trellis=None):
    if trellis is None:
        trellis = get_trellis()
    
    encoded_bits = cc.conv_encode(message_bits, trellis)
    return encoded_bits
def conv_decode(encoded_bits, message_length=None, trellis=None):
    if trellis is None:
        trellis = get_trellis()
    
    # Convert to numpy array if not already
    encoded_bits = np.array(encoded_bits)
    
    decoded_bits = cc.viterbi_decode(encoded_bits, trellis)
    
    if message_length is not None:
        decoded_bits = decoded_bits[:message_length]
    
    return list(decoded_bits)


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

# def generate_payload(size, max_correlation=6):
#     # Ensure size is a multiple of 8
#     if size % 8 != 0:
#         size += 8 - (size % 8)  # Round up to next multiple of 8
#         # print(f"Size rounded up to {size} to match byte boundaries.")

#     while True:
#         # Generate a random string of printable characters (excluding space)
#         characters = [chr(i) for i in range(33, 127)]  # Printable ASCII excluding space
#         random_string = ''.join(random.choices(characters, k=(size // 8 + 5)))  # a bit longer

#         # Convert string to bits
#         bit_list = string_to_bits(random_string)

#         # Take the first 'size' bits
#         payload = bit_list[:size]

#         # Check correlation with Barker 13
#         correlation = signal.correlate(config.BINARY_BARKER, payload, mode='same')
#         is_not_similar_to_BARKER_13 = max(correlation) == max_correlation
#         # check if correlation contains a 9
#         if is_not_similar_to_BARKER_13:
#             return bits_to_string(payload)  # Good payload

def generate_payload(size, target_correlation=6, convolutional_encoding=config.CONVOLUTIONAL_CODING):
    """
    Generates a random payload string.
    Checks correlation against Barker-13.
    If encoding=True, finds bits where original_corr == target_corr AND encoded_corr == target_corr.
    If encoding=False, finds bits where original_corr == target_corr.

    Args:
        size (int): Desired length of the payload in bits (before encoding).
        target_correlation (int): Exact max correlation value required.
        convolutional_encoding (bool): Apply convolutional encoding and check.

    Returns:
        str: Generated payload string (original or encoded/padded). None on failure.
    """
    # Round up generation size for initial string creation if needed
    generation_size = size + (8 - size % 8) % 8

    trellis = None
    if convolutional_encoding:
        try:
            # Using default trellis (Rate 1/2, K=3, G=[5,7])
            trellis = get_trellis()
        except Exception as e:
             print(f"Error creating trellis: {e}")
             return None

    attempts = 0
    max_attempts = 100000 # Safety break

    while attempts < max_attempts:
        attempts += 1
        # Generate random string -> bits
        num_chars = generation_size // 8 + 5 # Generate extra chars
        characters = [chr(i) for i in range(33, 127)] # Printable ASCII
        random_string = ''.join(random.choices(characters, k=num_chars))
        bit_list = string_to_bits(random_string)

        if len(bit_list) < size: continue # Need enough bits

        payload_bits = bit_list[:size] # Use exact size for checks

        # --- Check Correlation of Original Bits ---
        try:
            correlation_orig = signal.correlate(payload_bits, config.BINARY_BARKER, mode='valid')
            max_corr_orig = np.max(correlation_orig) if correlation_orig.size else -np.inf
        except ValueError as e:
            # print(f"Correlation error (orig): {e}") # e.g., if payload_bits is empty
            max_corr_orig = -np.inf
            continue # Try next random string

        # --- Condition 1: Original bits correlation must match target ---
        if max_corr_orig <= target_correlation:
            if not convolutional_encoding:
                # Success without encoding
                return bits_to_string(payload_bits)
            else:
                # --- Apply Encoding and Check Correlation of Encoded Bits ---
                try:
                    encoded_bits = conv_encode(payload_bits, trellis) # Get numpy array

                    correlation_enc = signal.correlate(encoded_bits, config.BINARY_BARKER, mode='valid')
                    max_corr_enc = np.max(correlation_enc) if correlation_enc.size else -np.inf

                    # --- Condition 2: Encoded bits correlation must ALSO match target ---
                    if max_corr_enc <= 8:
                        # Success with encoding! Return encoded bits as string.
                        return bits_to_string(encoded_bits)
                    # else: Encoded correlation didn't match, loop continues

                except Exception as e:
                    print(f"Error during encoding or encoded correlation: {e}")
                    continue # Try next random string

        # If original correlation didn't match, or encoded didn't match, loop continues

    print(f"Failed to generate payload after {max_attempts} attempts for target={target_correlation}, encoding={convolutional_encoding}.")
    return None

if __name__ == "__main__":
    size = 100  # Example size
    payload_bits = generate_payload(size, 5)
    # payload_string = bits_to_string(payload_bits)
    # example of payload with barker corr < 3 : `u%.1@@C829fS
    print("Generated Payload (bits):", payload_bits)
    # print("Generated Payload (string):", payload_string)
    print("Payload Length (bits):", len(payload_bits))
    # print("Payload Length (chars):", len(payload_string))
    a = max(signal.correlate(string_to_bits(payload_bits), config.BINARY_BARKER, mode='valid'))
    print(f"Max correlation with Barker 13: {a}")
    unique_payload_correlations_size_100 = []
    nums = np.arange(5, 10, 1)
    for i in range(len(nums)):
        payload = generate_payload(size, nums[i])
        correlation = signal.correlate(string_to_bits(payload_bits), config.BINARY_BARKER, mode='valid')
        max_correlation = max(correlation)
        print(f"Max correlation with Barker 13: {max_correlation}")
        unique_payload_correlations_size_100.append(payload)
        
    # print(unique_payload_correlations_size_100)
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

    trellis = get_trellis()

    for payload in unique_payload_correlations_size_100:
        print(f"Payload: {payload}, Decoded: {bits_to_string(conv_decode(string_to_bits(payload), size, trellis))}")
        