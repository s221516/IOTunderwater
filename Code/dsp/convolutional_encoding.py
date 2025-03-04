import numpy as np
from reedsolo import RSCodec, ReedSolomonError

class ConvolutionalEncoder:
    def __init__(self):
        self.state = [0, 0, 0]  # Shift register initialized to 0

    def encode_bit(self, bit):
        """Encodes a single bit using convolutional encoding."""
        self.state = [bit] + self.state[:-1]  # Shift register update

        # Compute output bits based on the connections
        c1 = self.state[0] ^ self.state[1] ^ self.state[2]  # [1,2,3]
        c2 = self.state[1] ^ self.state[2]                  # [2,3]
        c3 = self.state[0] ^ self.state[2]                  # [1,3]

        return [c1, c2, c3]

    def encode(self, bits):
        """Encodes a sequence of bits."""
        encoded_bits = []
        for bit in bits:
            encoded_bits.extend(self.encode_bit(bit))
        return encoded_bits

    def pack_bits(self, bits):
        """Packs bits into bytes, six bits per byte."""
        packed_bytes = bytearray()
        for i in range(0, len(bits), 6):
            byte = (bits[i] << 5) | (bits[i + 1] << 4) | (bits[i + 2] << 3) | (bits[i + 3] << 2) | (bits[i + 4] << 1) | (bits[i + 5] if i + 5 < len(bits) else 0)
            packed_bytes.append(byte)
        return packed_bytes

    def unpack_bits(self, packed_bytes):
        """Unpacks bytes into bits, six bits per byte."""
        bits = []
        for byte in packed_bytes:
            bits.append((byte >> 5) & 0x1)
            bits.append((byte >> 4) & 0x1)
            bits.append((byte >> 3) & 0x1)
            bits.append((byte >> 2) & 0x1)
            bits.append((byte >> 1) & 0x1)
            bits.append(byte & 0x1)
        return bits


class ViterbiDecoder:
    def __init__(self):
        self.trellis = {}  # Stores path metrics and history

    def decode(self, received_bits):
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
                    metric = path_metrics[prev_state] + hamming_distance(received, expected)

                    if metric < new_path_metrics[next_state]:
                        new_path_metrics[next_state] = metric
                        new_state_history[next_state] = state_history[prev_state] + [bit]

            path_metrics = new_path_metrics
            state_history = new_state_history

        # Find the best path (smallest metric)
        best_state = min(path_metrics, key=path_metrics.get)
        return state_history[best_state]

def hamming_distance(received, expected):
    """Computes Hamming distance between received bits and expected bits."""
    return sum(r != e for r, e in zip(received, expected))

# Function to inject random errors into the encoded sequence
def inject_errors(encoded_bits, num_errors=2, burst=False, error_pos=None, custom_errors=None):
    """
    Introduces errors into the encoded bits.
    
    Parameters:
    - num_errors: Number of errors to introduce.
    - burst: If True, introduces a burst error; otherwise, introduces random errors.
    - error_pos: If burst is True, specifies the starting index of the burst error.
    - custom_errors: A list of indices where errors should be introduced manually.
    
    Returns:
    - The received bit sequence with errors.
    """
    received_bits = encoded_bits.copy()
    
    if custom_errors:
        # Manually specified error positions
        for idx in custom_errors:
            if 0 <= idx < len(received_bits):
                received_bits[idx] ^= 1  # Flip the bit
    
    elif burst and error_pos is not None:
        # Introduce a burst error at the given position
        for i in range(error_pos, min(error_pos + num_errors, len(received_bits))):
            received_bits[i] ^= 1  # Flip the bit
    
    else:
        # Introduce random errors
        error_indices = np.random.choice(len(encoded_bits), num_errors, replace=False)
        for idx in error_indices:
            received_bits[idx] ^= 1  # Flip the bit
    
    return received_bits


def text_to_bits(text):
    """Converts a text string into a bit sequence."""
    bits = []
    for char in text:
        bits.extend([int(bit) for bit in format(ord(char), '08b')])
    return bits

def bits_to_text(bits):
    """Converts a bit sequence into a text string."""
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        char = chr(int(''.join(map(str, byte)), 2))
        chars.append(char)
    return ''.join(chars)


def bytes_to_bits(byte_array):
    """Converts a byte array into a bit sequence, removing leading zeros."""
    bits = []
    for byte in byte_array:
        # Convert byte to binary string and strip leading zeros
        bit_str = format(byte, '08b').lstrip('0')
        # If the result is an empty string, it means the byte was 0
        if bit_str == '':
            bit_str = '0'
        bits.extend([int(bit) for bit in bit_str])
    return bits

# Example Usage
encoder = ConvolutionalEncoder()
decoder = ViterbiDecoder()
rsc = RSCodec(2)  # Reed-Solomon codec with 2 error correction symbols

# Example input bit sequence
input_bits = text_to_bits("Test")

# Encode the input bits using Reed-Solomon encoding
message = bytes(input_bits)
encoded_message = rsc.encode(message)
print("Reed-Solomon Encoded Message: ", bytes_to_bits(encoded_message))

# Convert the encoded message to a bit sequence
encoded_bits = []
for byte in encoded_message:
    encoded_bits.extend([int(bit) for bit in format(byte, '08b')])

# Encode the bit sequence using convolutional encoding
conv_encoded_bits = encoder.encode(encoded_bits)
# print("Convolutional Encoded Bits:  ", conv_encoded_bits)

# Pack the convolutionally encoded bits into bytes
packed_bits = encoder.pack_bits(conv_encoded_bits)
print("Packed Bits: ", bytes_to_bits(packed_bits))

# Modify these parameters to tweak error injection
num_errors = 32
burst_error = False
error_position = 0
custom_error_positions = None  # Example: [2, 5, 10] to manually select positions

# Inject errors into the packed bits
received_bits = inject_errors(packed_bits, num_errors=num_errors, burst=burst_error, 
                              error_pos=error_position, custom_errors=custom_error_positions)

print("Received Bits: ", bytes_to_bits(received_bits))


# Unpack the received bits
unpacked_bits = encoder.unpack_bits(received_bits)
# print("Unpacked Bits: ", unpacked_bits)

# Decode the unpacked bits using Viterbi algorithm
viterbi_decoded_bits = decoder.decode(unpacked_bits)
# print("Viterbi Decoded Bits: ", viterbi_decoded_bits)

# Convert the decoded bits back to bytes
decoded_bytes = bytearray()
for i in range(0, len(viterbi_decoded_bits), 8):
    byte = int(''.join(map(str, viterbi_decoded_bits[i:i+8])), 2)
    decoded_bytes.append(byte)

# Decode the bytes using Reed-Solomon decoding
try:
    rs_decoded_message = rsc.decode(decoded_bytes)[0]
    print("Reed-Solomon Decoded Message: ", bytes_to_bits(rs_decoded_message))
    print("Decoded message: ", bits_to_text(bytes_to_bits(rs_decoded_message)))
except ReedSolomonError as e:
    print("Reed-Solomon Decoding failed: ", e)

print(hamming_distance(bytes_to_bits(packed_bits), bytes_to_bits(received_bits)))

# Check if the original message matches the decoded message
if rs_decoded_message == message:
    print("YAYAYAYAYAYAYAYA")
else:
    print("FAILED")