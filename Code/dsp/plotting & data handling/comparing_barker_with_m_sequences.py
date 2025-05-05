import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate

def generate_barker13():
    """Generate Barker-13 sequence"""
    barker = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
    return (barker + 1) // 2  # Convert from {-1,1} to {0,1}

def generate_m_sequence(degree=4):
    """Generate M-sequence of given degree"""
    length = 2**degree - 1
    sequence = np.zeros(length)
    register = np.ones(degree)  # Initial state
    
    # Feedback taps for degree 4 (x^4 + x^3 + 1)
    for i in range(length):
        sequence[i] = register[-1]
        feedback = (register[3] + register[2]) % 2
        register[1:] = register[:-1]
        register[0] = feedback
    
    return sequence  # Already in {0,1} format

def compute_autocorrelation(sequence):
    """Compute autocorrelation of sequence"""
    # Convert to {-1,1} for correlation, then back to {0,1}
    return correlate(sequence, sequence, mode='full')

def analyze_sequence_properties(sequence_name, sequence):
    """Analyze properties of given sequence"""
    autocorr = compute_autocorrelation(sequence)
    peak = np.max(np.abs(autocorr))
    sidelobes = np.delete(autocorr, len(sequence)-1)  # Remove main peak
    max_sidelobe = np.max(np.abs(sidelobes))
    
    return {
        'name': sequence_name,
        'length': len(sequence),
        'peak': peak,
        'max_sidelobe': max_sidelobe,
        'peak_to_sidelobe': peak/max_sidelobe if max_sidelobe != 0 else float('inf')
    }


def plot_sequence_comparison(sequences):
    """Plot comparison of sequences"""
    fig, axes = plt.subplots(2, len(sequences), figsize=(15, 8))
    
    for i, (name, seq) in enumerate(sequences.items()):
        # Plot sequence as binary points
        axes[0, i].scatter(range(len(seq)), seq, color='blue', s=100)
        axes[0, i].set_title(f'{name} Sequence')
        axes[0, i].set_xlabel('Index')
        axes[0, i].set_ylabel('Bit Value')
        axes[0, i].set_yticks([0, 1])
        axes[0, i].set_xticks(np.arange(0, len(seq), 1))
        axes[0, i].set_ylim(-0.1, 1.1)  # Add small padding
        axes[0, i].grid(True, linestyle='--', alpha=0.7)
        
        # Plot autocorrelation
        autocorr = compute_autocorrelation(seq)
        axes[1, i].plot(autocorr, color='blue')
        axes[1, i].set_title(f'{name} Autocorrelation')
        axes[1, i].set_xlabel('Lag')
        axes[1, i].set_ylabel('Correlation')
        axes[1, i].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig
def main():
    # Generate sequences
    barker13 = generate_barker13()
    m_seq = generate_m_sequence(4)  # 4th degree M-sequence (length 15)
    
    sequences = {
        'Barker-13': barker13,
        'M-sequence': m_seq
    }
    
    # Analyze properties
    results = []
    for name, seq in sequences.items():
        props = analyze_sequence_properties(name, seq)
        results.append(props)
    
    # Display results
    df = pd.DataFrame(results)
    print("\nSequence Properties:")
    print(df)
    
    # Plot comparison
    fig = plot_sequence_comparison(sequences)
    plt.show()
    
    return df, fig

if __name__ == "__main__":
    main()