from typing import List
from math import log2, ceil
from random import randrange

def hamming_distance(received, expected):
    """Computes Hamming distance between received bits and expected bits."""
    if (len(received) != len(expected)):
        return 
    else:
        return sum(r != e for r, e in zip(received, expected))

def __hamming_common(src: List[List[int]], s_num: int, encode=True) -> None:
    """
    Performs Hamming encoding/decoding operations
    """
    for block in src:
        sindrome = 0
        for s in range(s_num):
            sind = 0
            for p in range(2 ** s, len(block) + 1, 2 ** (s + 1)):
                for j in range(2 ** s):
                    if (p + j) > len(block):
                        break
                    sind ^= block[p + j - 1]
            
            if encode:
                block[2 ** s - 1] = sind
            else:
                sindrome += (2 ** s * sind)
        
        if (not encode) and sindrome:
            if 0 < sindrome <= len(block):
                block[sindrome - 1] ^= 1  # Correct the error by flipping the bit
                print(f"Corrected error at position {sindrome}")

def hamming_encode(msg: str, mode: int=8) -> str:
    """
    Encoding the message with Hamming code.
    :param msg: Message string to encode
    :param mode: number of significant bits
    :return: 
    """

    result = ""

    msg_b = msg.encode("utf-8")
    s_num = ceil(log2(log2(mode + 1) + mode + 1))   # number of control bits
    bit_seq = []
    for byte in msg_b:  # get bytes to binary values; every bits store to sublist
        bit_seq += list(map(int, f"{byte:08b}"))

    res_len = ceil((len(msg_b) * 8) / mode)     # length of result (bytes)
    bit_seq += [0] * (res_len * mode - len(bit_seq))    # filling zeros

    to_hamming = []

    for i in range(res_len):    # insert control bits into specified positions
        code = bit_seq[i * mode:i * mode + mode]
        for j in range(s_num):
            code.insert(2 ** j - 1, 0)
        to_hamming.append(code)

    __hamming_common(to_hamming, s_num, True)   # process

    for i in to_hamming:
        result += "".join(map(str, i))

    return result

def hamming_decode(msg: str, mode: int=8) -> str:
    """
    Decoding the message with Hamming code
    """
    s_num = ceil(log2(log2(mode + 1) + mode + 1))
    code_len = mode + s_num
    res_len = len(msg) // code_len
    
    # Convert message to blocks of integers
    to_hamming = []
    for i in range(res_len):
        block = list(map(int, msg[i * code_len:(i + 1) * code_len]))
        to_hamming.append(block)
    
    # Process each block
    __hamming_common(to_hamming, s_num, False)
    
    # Remove parity bits and construct result
    result = ""
    for block in to_hamming:
        data_bits = []
        # Skip parity bit positions (powers of 2)
        for i in range(1, len(block) + 1):
            if not (i & (i - 1) == 0):  # Check if i is not a power of 2
                data_bits.append(str(block[i - 1]))
        result += ''.join(data_bits)
    
    return result

def noizer_single_error_per_block(msg: str, mode: int) -> str:
    """
    Generates exactly one error per block using random positions
    """
    seq = list(map(int, msg))
    s_num = ceil(log2(log2(mode + 1) + mode + 1))
    code_len = mode + s_num
    cnt = len(msg) // code_len
    result = []

    for i in range(cnt):
        block = seq[i * code_len:(i + 1) * code_len]
        # Introduce one random error in each block
        error_pos = randrange(code_len)
        block[error_pos] = 1 - block[error_pos]
        result.extend(block)
        print(f"Block {i}: Introduced error at position {error_pos}")

    return ''.join(map(str, result))

def noizer_multiple_errors_per_block(msg: str, mode: int, errors_per_block: int) -> str:
    """
    Generates multiple errors per block using random positions
    
    Args:
        msg: The encoded message
        mode: Number of significant bits
        errors_per_block: Number of errors to introduce in each block
    """
    seq = list(map(int, msg))
    s_num = ceil(log2(log2(mode + 1) + mode + 1))
    code_len = mode + s_num
    cnt = len(msg) // code_len
    result = []

    for i in range(cnt):
        block = seq[i * code_len:(i + 1) * code_len]
        # Get random positions for errors, ensuring no duplicates
        error_positions = set()
        while len(error_positions) < errors_per_block:
            error_positions.add(randrange(code_len))
        
        # Introduce errors at selected positions
        for pos in error_positions:
            block[pos] = 1 - block[pos]  # Flip the bit
            print(f"Block {i}: Introduced error at position {pos}")
        
        result.extend(block)

    return ''.join(map(str, result))

def decode_bytes_to_bits(bits: list) -> str:
	if len(bits) % 8 != 0:
		remainder = len(bits) % 8
		bits += ([0] * (8 - remainder))
	message = ""
	for i in range(0, len(bits), 8):
		byte = bits[i : i + 8]
		char = chr(int("".join(map(str, byte)), 2))
		if 32 <= ord(char) <= 126:
			message += char
	return message


# Updated test function
def test_multiple_error_correction():
    """Test Hamming code's ability to handle multiple errors per block"""
    
    test_string = "A"
    print("\nTest Case: Multiple errors per block")
    print(f"Original message: {test_string}")
    print(f"Original binary: {''.join(format(ord(c), '08b') for c in test_string)}")
    
    # Encode
    encoded = hamming_encode(test_string, 8)
    print(f"Encoded message: {encoded}")
    print(f"Number of 12-bit blocks: {len(encoded) // 12}")
    
    # Try different numbers of errors per block
    for errors in range(1, 3):  # Test with 1, 2, and 3 errors per block
        print(f"\nTesting with {errors} errors per block:")
        
        # Introduce errors
        corrupted = noizer_multiple_errors_per_block(encoded, 8, errors)
        print(f"Corrupted message: {corrupted}")
        print("Hamming dist: ", hamming_distance(corrupted, encoded))
        
        # Count differences
        differences = sum(1 for a, b in zip(encoded, corrupted) if a != b)
        print(f"Total number of introduced errors: {differences}")
        
        # Try to decode and correct
        decoded = hamming_decode(corrupted, 8)
        decoded_string = decode_bytes_to_bits([int(bit) for bit in decoded])
        print(f"Decoded message: {decoded_string}")
        print(f"Success: {'Yes' if decoded_string == test_string else 'No'}")

if __name__ == "__main__":
    test_multiple_error_correction()