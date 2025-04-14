import commpy.channelcoding.convcode as cc
import numpy as np
import matplotlib.pyplot as plt

def string_to_bits(string_input):
    """Convert a string to a numpy array of bits"""
    bit_strings = [format(ord(char), '08b') for char in string_input]
    bits = np.array([int(bit) for byte in bit_strings for bit in byte])
    return bits

def bits_to_string(bits):
    """Convert a numpy array of bits back to a string"""
    if len(bits) % 8 != 0:
        raise ValueError("Number of bits must be divisible by 8")
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

def add_awgn(signal, eb_no_db, rate=1/2):
    """Add Additive White Gaussian Noise to achieve desired Eb/N0"""
    eb = np.sum(np.abs(signal) ** 2) / len(signal)
    eb_no = 10 ** (eb_no_db / 10)
    n0 = eb / eb_no / rate
    noise = np.random.normal(0, np.sqrt(n0/2), len(signal))
    return signal + noise

def test_conv_code_awgn(message="Hello_there", eb_no_range=None, num_trials=100, trellis=None, rate=1/2):
    """Test convolutional code performance over AWGN channel"""
    if eb_no_range is None:
        eb_no_range = np.arange(0, 11, 1)
    
    if trellis is None:
        trellis = get_trellis()
        
    message_bits = string_to_bits(message)
    results = {
        'eb_no_db': eb_no_range,
        'ber': [],
        'success_rate': []
    }
    
    for eb_no in eb_no_range:
        successful_decodes = 0
        total_ber = 0
        
        for _ in range(num_trials):
            # Encode and modulate
            encoded_bits = cc.conv_encode(message_bits, trellis)
            am_signal = encoded_bits  # Use amplitude modulation
            
            # Add noise
            noisy_signal = add_awgn(am_signal, eb_no, rate)
            received_bits = (noisy_signal > 0.5).astype(int)
            
            # Decode
            decoded_bits = cc.viterbi_decode(received_bits, trellis)
            decoded_bits = decoded_bits[:len(message_bits)]
            
            # Calculate performance
            num_errors, ber = BER_calc(message_bits, decoded_bits)
            total_ber += ber
            if np.array_equal(decoded_bits, message_bits):
                successful_decodes += 1
        
        # Store results
        results['ber'].append(total_ber / num_trials)
        results['success_rate'].append(successful_decodes / num_trials * 100)
        
        print(f"Eb/N0: {eb_no:.1f} dB")
        print(f"BER: {results['ber'][-1]:.2%}")
        print(f"Success Rate: {results['success_rate'][-1]:.1f}%")
        print("-" * 40)
    
    return results

def test_multiple_generators(message="Hello_there", eb_no_range=None, num_trials=100):
    """Test different generator polynomials and rates"""
    test_configs = [
        (3, [[5, 7]], 2),                # Rate 1/2 basic
        # (7, [[171, 133]], 2),            # Rate 1/2 NASA
        # (3, [[7, 3, 5]], 3),             # Rate 1/3 basic
        # (7, [[171, 165, 133]], 3),       # Rate 1/3 stronger
        (3, [[7, 7, 5], [7, 5, 7]], "2/3")  # Rate 2/3
    ]
    
    all_results = {}
    for constraint_length, generators, rate in test_configs:
        print(f"\nTesting rate {rate if isinstance(rate, str) else f'1/{rate}'} code with generators {generators}")
        trellis = get_trellis(constraint_length, generators, rate)
        actual_rate = 2/3 if rate == "2/3" else 1/rate
        results = test_conv_code_awgn(
            message=message,
            eb_no_range=eb_no_range,
            num_trials=num_trials,
            trellis=trellis,
            rate=actual_rate
        )
        config_name = f"K={constraint_length}, R={rate if isinstance(rate, str) else f'1/{rate}'}, G={generators}"
        all_results[config_name] = results
    
    return all_results

def plot_generator_comparison(all_results):
    """Plot BER comparison for different generator polynomials"""
    plt.figure(figsize=(12, 6))
    
    for config_name, results in all_results.items():
        plt.semilogy(
            results['eb_no_db'], 
            results['ber'], 
            '.-', 
            label=config_name
        )
    
    plt.grid(True)
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('Bit Error Rate')
    plt.title('BER vs Eb/N0 for Different Generator Polynomials')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

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


# this is just to simulate some BER, maybe this is not even neccessary as we cant really use conv encoding :)
if __name__ == "__main__":
    message = "Hello_there"
    eb_no_range = np.arange(0, 11, 1)
    
    print("\nTesting different generator polynomials...")
    all_results = test_multiple_generators(
        message=message,
        eb_no_range=eb_no_range,
        num_trials=25
    )
    
    plot_generator_comparison(all_results)