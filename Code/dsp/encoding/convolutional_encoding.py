import numpy as np


def hamming_distance(received, expected):
    """Computes Hamming distance between received bits and expected bits."""
    return sum(r != e for r, e in zip(received, expected))

def encode_bit(bit, state):
    """Encodes a single bit using convolutional encoding."""
    state = [bit] + state[:-1]  # Shift register update

    # Compute output bits based on the connections
    c1 = state[0] ^ state[1] ^ state[2]  # [1,2,3]
    c2 = state[1] ^ state[2]             # [2,3]
    c3 = state[0] ^ state[2]             # [1,3]

    return [c1, c2, c3], state

def conv_encode(bits):
    """Encodes a sequence of bits."""
    state = [0, 0, 0]  # Shift register initialized to 0
    encoded_bits = []
    for bit in bits:
        encoded, state = encode_bit(bit, state)
        encoded_bits.extend(encoded)

    return encoded_bits


def conv_decode(received_bits):
    """Decodes a sequence of received bits using Viterbi algorithm."""
    num_states = 4  # 2^(constraint length - 1) = 2^(3-1) = 4 states
    state_history = {s: [] for s in range(num_states)}
    path_metrics = {s: float('inf') for s in range(num_states)}
    path_metrics[0] = 0  # Start from all-zero state

    expected_outputs = {
        0: ([0, 0, 0], [1, 0, 1]),  # State 00 → next states 00, 10
        1: ([1, 1, 1], [0, 1, 0]),  # State 01 → next states 00, 10
        2: ([1, 1, 0], [0, 1, 1]),  # State 10 → next states 01, 11
        3: ([0, 0, 1], [1, 0, 0]),  # State 11 → next states 01, 11
    }

    num_steps = len(received_bits) // 3

    for step in range(num_steps):

        new_path_metrics = {s: float('inf') for s in range(num_states)}
        new_state_history = {s: [] for s in range(num_states)}

        received = received_bits[step * 3: step * 3 + 3]

        for prev_state in range(num_states):
            for bit in [0, 1]:  # Two possible transitions per state
                next_state = ((prev_state << 1) | bit) & 0b11
                expected = expected_outputs[prev_state][bit]
                branch_metric = path_metrics[prev_state] + hamming_distance(received, expected)

                if branch_metric < new_path_metrics[next_state]:
                    new_path_metrics[next_state] = branch_metric
                    new_state_history[next_state] = state_history[prev_state] + [bit]

        path_metrics = new_path_metrics
        state_history = new_state_history

    # Find the best path (smallest metric)
    best_state = min(path_metrics, key=path_metrics.get)
    return state_history[best_state]

def bit_string_to_list(bit_string):
    """Converts a bit string to a list of integers."""
    return [int(bit) for bit in bit_string]

def list_to_bit_string(bit_list):
    """Converts a list of integers to a bit string."""
    return ''.join(str(bit) for bit in bit_list)

import random

def inject_errors(encoded_bits, num_errors):
    """
    Injects a specified number of random bit errors into the encoded data.
    
    Args:
        encoded_bits (list): The encoded bits to inject errors into
        num_errors (int): Number of errors to inject
    
    Returns:
        list: Encoded bits with errors injected
    """
    # Convert to list if input is a string
    if isinstance(encoded_bits, str):
        bits = bit_string_to_list(encoded_bits)
    else:
        bits = encoded_bits.copy()
    
    # Get random positions to inject errors
    positions = random.sample(range(len(bits)), num_errors)
    
    # Flip bits at the selected positions
    for pos in positions:
        bits[pos] = 1 - bits[pos]  # Flip the bit (0->1 or 1->0)
    
    return bits

def string_to_bits(message):
    """Convert a string to a list of bits."""
    # Convert string to bytes, then to binary representation
    bits = []
    for char in message:
        # Get ASCII value and convert to 8-bit binary
        ascii_val = ord(char)
        for i in range(7, -1, -1):  # MSB first
            bits.append((ascii_val >> i) & 1)
    return bits

def bits_to_string(bits):
    """Convert a list of bits back to a string."""
    # Ensure the number of bits is divisible by 8
    if len(bits) % 8 != 0:
        padding = 8 - (len(bits) % 8)
        bits = bits + [0] * padding
    
    # Convert bits back to characters
    message = ""
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        char_code = sum(bit << (7-j) for j, bit in enumerate(byte))
        message += chr(char_code)
    return message

# Update the main example usage:
if __name__ == "__main__":
    # Test with a string message
    message = "AA"
    print(f"Original message: {message}")
    
    # Convert string to bits and encode
    message_bits = string_to_bits(message)
    encoded_message = conv_encode(message_bits)
    print(f"Encoded message: {encoded_message}")
    
    # Inject some errors
    num_errors = 4
    corrupted_message = inject_errors(encoded_message, num_errors)
    print(f"Corrupted message (with {num_errors} errors): {corrupted_message}")
    
    # Decode the corrupted message
    decoded_bits = conv_decode(corrupted_message)
    decoded_message = bits_to_string(decoded_bits)
    print(f"Decoded message: {decoded_message}")
