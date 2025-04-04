import commpy.channelcoding.convcode as cc
import numpy as np


def BER_calc(a, b):
    num_ber = np.sum(np.abs(a - b))
    ber = np.mean(np.abs(a - b))
    return int(num_ber), ber


N = 10  # Original message length

# Generate message bits
message_bits = np.random.randint(0, 2, N)

# Define a rate 1/2 convolutional code with correct parameters
generator_matrix = np.array([[5, 7]])  # Rate 1/2 generators (octal)
memory = np.array([2])  # Memory matches generator constraint length-1 (3-1=2)
trellis = cc.Trellis(memory, generator_matrix)
print(trellis.total_memory)
# Encode the message (appends 2 tail bits, encoded length = (10+2)*2=24)
encoded_bits = cc.conv_encode(message_bits, trellis)

# NOTE: FLIPPING A BIT
encoded_bits[1] = (
    encoded_bits[1] + 1
) % 2  # Flip the second bit of the encoded message
encoded_bits[2] = (
    encoded_bits[2] + 1
) % 2  # Flip the second bit of the encoded message

# Decode and truncate to original length
decoded_bits = cc.viterbi_decode(encoded_bits, trellis)
decoded_bits = decoded_bits[:N]  # Remove the 2 tail bits

print("Original length:", len(message_bits))
print("Decoded length:", len(decoded_bits))
print("Original message:", message_bits)
print("Encoded message:", encoded_bits)
print("Decoded message:", decoded_bits)
num_ber, ber = BER_calc(message_bits, decoded_bits)
