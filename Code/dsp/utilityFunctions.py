
import random
import scipy.signal as signal
import numpy as np
import config
def bytes_to_bin_array(byte_array):
    """Converts a byte array into a binary array."""
    bin_array = []
    for byte in byte_array:
        bin_array.extend([int(bit) for bit in format(byte, "08b")])
    return bin_array

def string_to_bin_array(string):
    """Converts a string into a binary array."""
    byte_array = string.encode("utf-8")
    return bytes_to_bin_array(byte_array)

def message_toBitArray(self, message: str):  # ON OFF KEYING
    message_binary = "".join(format(ord(i), "08b") for i in message)
    # print(f"Message in binary: {message_binary}")
    # TODO: determine the exact number of samples per bit that makes sense in relation to our sample rate
    # and bits per second and also how this is done in the signal generator

    square_wave = []
    for bit in message_binary:
        if bit == "0":
            square_wave += [0]
        elif bit == "1":
            square_wave += [1]

    return square_wave


def bits_to_string(bits):
    # Convert a list of bits back to a string
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        byte_str = ''.join(str(b) for b in byte)
        chars.append(chr(int(byte_str, 2)))
        # print(f"Byte: {byte_str} -> Char: {chars[-1]}")  # Debugging line
    return ''.join(chars)

def generate_payload(size, target_correlation=6):
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

    attempts = 0
    max_attempts = 100000 # Safety break

    while attempts < max_attempts:
        attempts += 1
        # Generate random string -> bits
        num_chars = generation_size // 8 + 5 # Generate extra chars
        characters = [chr(i) for i in range(33, 127)] # Printable ASCII
        random_string = ''.join(random.choices(characters, k=num_chars))
        bit_list = string_to_bin_array(random_string)


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
            return bits_to_string(payload_bits)
        
    
    return None