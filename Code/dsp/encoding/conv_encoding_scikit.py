import commpy.channelcoding.convcode as cc
import numpy as np

def string_to_bits(string_input):
    """Convert a string to a numpy array of bits"""
    # Convert each character to 8-bit binary
    bit_strings = [format(ord(char), '08b') for char in string_input]
    # Join all bits and convert to numpy array of ints
    bits = np.array([int(bit) for byte in bit_strings for bit in byte])
    return bits

def bits_to_string(bits):
    """Convert a numpy array of bits back to a string"""
    # Ensure the number of bits is divisible by 8
    if len(bits) % 8 != 0:
        raise ValueError("Number of bits must be divisible by 8")
    
    # Convert bits to characters
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        char = chr(int(''.join(map(str, byte)), 2))
        chars.append(char)
    
    return ''.join(chars)

def BER_calc(a, b):
    """Calculate bit error rate"""
    num_ber = np.sum(np.abs(a - b))
    ber = np.mean(np.abs(a - b))
    return int(num_ber), ber

def inject_bit_errors(bits, num_errors):
    """Inject random bit errors"""
    corrupted_bits = bits.copy()
    error_positions = np.random.choice(
        len(bits), 
        size=num_errors, 
        replace=False
    )
    for pos in error_positions:
        corrupted_bits[pos] = (corrupted_bits[pos] + 1) % 2
    return corrupted_bits

def test_error_correction(message="Hello_there", max_errors=10, num_trials=100):
    """Test error correction capabilities for different numbers of bit errors"""
    global contraint_length
    contraint_length = 7 # size of shift register
    # Setup convolutional code parameters
    generator_matrix = np.array([[171, 133]])  # Rate 1/2 generators
    memory = np.array([contraint_length - 1])  # Memory length
    trellis = cc.Trellis(memory, generator_matrix)

    # Convert message to bits
    message_bits = string_to_bits(message)
    results = {
        'num_errors': [],
        'success_rate': [],
        'avg_ber': []
    }
    
    # Test different numbers of errors
    for num_errors in range(1, max_errors + 1):
        successful_decodes = 0
        total_ber = 0
        
        for _ in range(num_trials):
            # Encode
            encoded_bits = cc.conv_encode(message_bits, trellis)
            
            # Inject errors
            corrupted_bits = inject_bit_errors(encoded_bits, num_errors)
            
            # Decode
            decoded_bits = cc.viterbi_decode(corrupted_bits, trellis)
            decoded_bits = decoded_bits[:len(message_bits)]  # Remove tail bits
            
            # Calculate BER
            _, ber = BER_calc(message_bits, decoded_bits)
            total_ber += ber
            
            # Check if decoded correctly
            if np.array_equal(decoded_bits, message_bits):
                successful_decodes += 1
        
        # Store results
        results['num_errors'].append(num_errors)
        results['success_rate'].append(successful_decodes / num_trials * 100)
        results['avg_ber'].append(total_ber / num_trials * 100)
        
        print(f"Number of errors: {num_errors}")
        print(f"Success rate: {results['success_rate'][-1]:.1f}%")
        print(f"Average BER: {results['avg_ber'][-1]:.2f}%")
        print("-" * 40)
    
    return results

def get_trellis():
    constraint_length = 7  # size of shift register
    generator_matrix = np.array([[171, 133]])  # Rate 1/2 generators (octal)
    memory = np.array([constraint_length - 1])  # Memory length
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


# # Example usage in your main block:
# if __name__ == "__main__":
#     # Example usage in another script
#     message = "Hello_there"
#     message_bits = string_to_bits(message)

#     # Encode
#     encoded_bits = conv_encode(message_bits)
#     print(len(encoded_bits))

#     # Decode
#     decoded_bits = conv_decode(encoded_bits, len(message_bits))
#     print(len(message_bits))

#     # Convert back to string
#     decoded_message = bits_to_string(decoded_bits)
#     print(decoded_message)  # Should print "Hello World"

if __name__ == "__main__":
    message = "Hello_there"
    message_bits = string_to_bits(message)
    results = test_error_correction(message=message, max_errors=20)
    
    # Print summary
    print("\nError Correction Analysis:")
    print("------------------------")
    print(f"Message length: {len(message_bits)} bits")
    print(f"Encoded length: {len(message_bits) * 2 + 12} bits")
    
    # Find maximum reliable correction
    reliable_threshold = 95
    max_reliable_errors = 0
    for i, rate in enumerate(results['success_rate']):
        if rate >= reliable_threshold:
            max_reliable_errors = results['num_errors'][i]
    
    print(f"Practical reliable correction: {max_reliable_errors} bits total")


# # Main execution
# if __name__ == "__main__":
#     # Simulation parameters
#     num_simulations = 100
#     input_string = "Test_test_test Morten disse nødder"
#     num_errors = 4
#     total_bers = []
#     successful_decodes = 0
    
#     # Define convolutional code parameters
#     generator_matrix = np.array([[5, 7]])  # Rate 1/2 generators (octal)
#     memory = np.array([2])  # Memory matches generator constraint length-1
#     trellis = cc.Trellis(memory, generator_matrix)
    
#     print(f"\nRunning {num_simulations} simulations with {num_errors} bit errors each...")
#     print(f"Original string: {input_string}")
    
#     # Convert string to bits (do this once)
#     message_bits = string_to_bits(input_string)
    
#     # Run simulations
#     for i in range(num_simulations):
#         # Encode the message
#         encoded_bits = cc.conv_encode(message_bits, trellis)
        
#         # Inject errors
#         corrupted_bits = inject_bit_errors(encoded_bits, num_errors)
        
#         # Decode the corrupted message
#         decoded_bits = cc.viterbi_decode(corrupted_bits, trellis)
#         decoded_bits = decoded_bits[:len(message_bits)]
        
#         # Calculate BER
#         num_ber, ber = BER_calc(message_bits, decoded_bits)
#         total_bers.append(ber)
        
#         # Try to decode to string
#         try:
#             decoded_string = bits_to_string(decoded_bits)
#             if decoded_string == input_string:
#                 successful_decodes += 1
#         except ValueError:
#             continue
    
#     # Calculate statistics
#     average_ber = np.mean(total_bers) * 100  # Convert to percentage
#     min_ber = np.min(total_bers) * 100
#     max_ber = np.max(total_bers) * 100
#     success_rate = (successful_decodes / num_simulations) * 100
    
#     # Print results
#     print("\nSimulation Results:")
#     print("-" * 50)
#     print(f"Success Rate: {success_rate:.1f}% ({successful_decodes}/{num_simulations} successful decodes)")
#     print(f"Average BER: {average_ber:.2f}%")
#     print(f"Min BER: {min_ber:.2f}%")
#     print(f"Max BER: {max_ber:.2f}%")
#     print(f"Message length: {len(message_bits)} bits")
#     print(f"Encoded length: {len(encoded_bits)} bits")