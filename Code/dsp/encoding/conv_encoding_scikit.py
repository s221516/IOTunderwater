import commpy.channelcoding.convcode as cc
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from .. import config

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

def test_error_correction(message, num_errors, trellis=None):
    """
    Test error correction capability by adding specific number of errors
    and attempting to decode the message
    
    Args:
        message (str): Input message to encode and test
        num_errors (int): Number of bit errors to introduce
        trellis: Optional trellis object for encoding/decoding
    
    Returns:
        dict: Results containing error positions, if decoding was successful, etc.
    """
    # Convert message to bits and encode
    message_bits = string_to_bits(message)
    encoded_bits = conv_encode(message_bits, trellis)
    
    # Try different error patterns
    num_trials = 1000  # Number of different error patterns to try
    successful_decodes = 0
    failed_decodes = 0
    
    for trial in range(num_trials):
        # Create copy of encoded bits
        corrupted_bits = encoded_bits.copy()
        
        # Randomly select positions for errors
        error_positions = np.random.choice(
            len(encoded_bits), 
            size=num_errors, 
            replace=False
        )
        
        # Flip bits at error positions
        for pos in error_positions:
            corrupted_bits[pos] = 1 - corrupted_bits[pos]
        
        # Try to decode
        decoded_bits = conv_decode(corrupted_bits, len(message_bits), trellis)
        
        # Check if decoded correctly
        if np.array_equal(decoded_bits, message_bits):
            successful_decodes += 1
        else:
            failed_decodes += 1
            
    success_rate = (successful_decodes / num_trials) * 100
    
    print(f"\nResults for {num_errors} errors:")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Successful decodes: {successful_decodes}/{num_trials}")
    print(f"Failed decodes: {failed_decodes}/{num_trials}")
    
    return {
        'num_errors': num_errors,
        'success_rate': success_rate,
        'successful_decodes': successful_decodes,
        'failed_decodes': failed_decodes
    }

def find_dfree(message="test", max_errors=10, trellis=None):
    """
    Find the free distance by testing increasing numbers of errors
    until decoding consistently fails
    
    Args:
        message (str): Test message
        max_errors (int): Maximum number of errors to test
        trellis: Optional trellis object for encoding/decoding
    """
    print(f"Testing error correction capability with message: {message}")
    
    results = []
    for num_errors in range(1, max_errors + 1):
        result = test_error_correction(message, num_errors, trellis)
        results.append(result)
        
        # If success rate drops below 50%, we've likely exceeded d_free/2
        if result['success_rate'] < 50:
            estimated_dfree = (num_errors - 1) * 2
            print(f"\nEstimated d_free ≈ {estimated_dfree}")
            print(f"Can reliably correct up to {estimated_dfree//2 - 1} errors")
            break
    
    return results
def plot_trellis_diagram(num_stages=5):
    """
    Plot trellis diagram for rate 1/2 convolutional code with generators [5,7]
    
    Args:
        num_stages (int): Number of stages to show in trellis diagram
    """
    # Create directed graph
    G = nx.DiGraph()
    
    # For [5,7] with constraint length 3, we have 4 states (00,01,10,11)
    states = ['00', '01', '10', '11']
    
    # Create state to position mapping (reverse order to put 00 at top)
    state_positions = {state: 3 - int(state, 2) for state in states}
    
    # Add nodes for each stage
    for stage in range(num_stages):
        for state in states:
            # Add node with position - using state_positions for y-coordinate
            G.add_node(f's{stage}_{state}', 
                      pos=(stage, state_positions[state]))
    
    # Add edges between states
    for stage in range(num_stages-1):
        for current_state in states:
            # Convert state to bits
            reg = [int(b) for b in current_state]
            
            # For input bit 0
            input_bit = 0
            new_reg = [input_bit] + reg[:-1]
            next_state_0 = ''.join(map(str, new_reg))
            output_0 = compute_output(input_bit, reg)
            
            # For input bit 1
            input_bit = 1
            new_reg = [input_bit] + reg[:-1]
            next_state_1 = ''.join(map(str, new_reg))
            output_1 = compute_output(input_bit, reg)
            
            # Add edges with output labels
            G.add_edge(f's{stage}_{current_state}', 
                      f's{stage+1}_{next_state_0}',
                      label=f'{output_0}')
            G.add_edge(f's{stage}_{current_state}', 
                      f's{stage+1}_{next_state_1}',
                      label=f'{output_1}')
    
    # Draw the trellis diagram
    plt.figure(figsize=(12, 6))
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, labels={node: node.split('_')[1] for node in G.nodes()})
    
    # Draw edges with different colors for 0 and 1 input
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos)
    
    # Add time labels below each column
    plt.text(0.0, -0.3, 't', ha='center')
    for i in range(1, num_stages):
        plt.text(i, -0.3, f't+{i}', ha='center')
    
    plt.title('Trellis Diagram for Rate 1/2 Code [5,7]')
    plt.axis('off')  # Remove all borders and axes
    
    plt.tight_layout()
    plt.show()

def compute_output(input_bit, reg):
    """Compute output bits for given input and register state"""
    # Generators [5,7] in binary: [101, 111]
    g1 = [1, 0, 1]  # 5 in binary
    g2 = [1, 1, 1]  # 7 in binary
    
    state = [input_bit] + reg
    
    # Compute outputs
    out1 = sum(a*b for a,b in zip(state, g1)) % 2
    out2 = sum(a*b for a,b in zip(state, g2)) % 2
    
    return f'{out1}{out2}'

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
    ## this simulates some additive white noise to the channel, to see how well it handles that
    # message = "Hello_there"
    # eb_no_range = np.arange(0, 20, 1)
    
    # print("\nTesting different generator polynomials...")
    # all_results = test_multiple_generators(
    #     message=message,
    #     eb_no_range=eb_no_range,
    #     num_trials=50
    # )
    
    # plot_generator_comparison(all_results)

    # # this just injects errors straight up
    # # Test with default rate 1/2 code
    # MESSAGE = "Hello_there"
    # print("\nTesting Rate 1/2 Basic Code (generators=[5,7])")
    # trellis = get_trellis(constraint_length=3, generators=[[5, 7]], rate=2)
    # results = find_dfree(message=MESSAGE, max_errors=10, trellis=trellis)

    plot_trellis_diagram(5)  # Show 5 stages
    # msg = "}VvF*E@9>-go*" # 3
    # msg = "+,M4J1ABraRJ&" # 4
    # msg = "i3aw,*X@j&y;y" # 5
    msg = "~7,w]@s,V+{2Y" # 6
    msg_corr = signal.correlate(config.BINARY_BARKER, string_to_bits(msg), mode='same')
    

    max_corr_msg = np.max(msg_corr)
    print(f"Max correlation with original message: {max_corr_msg}")
    
    msg_conv = conv_encode(string_to_bits(msg))
    corr_conv_encoded = signal.correlate(config.BINARY_BARKER, msg_conv, mode='same')
    max_corr = np.max(corr_conv_encoded)
    print(f"Max correlation: {max_corr}")