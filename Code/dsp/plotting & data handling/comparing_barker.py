import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate

def generate_barker13_bipolar():
    """Generate Barker-13 sequence in {-1,1} format"""
    return np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])

def generate_barker13_binary():
    """Generate Barker-13 sequence in {0,1} format"""
    barker = generate_barker13_bipolar()
    return (barker + 1) // 2

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
    """Plot comparison of sequence autocorrelations"""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    colors = {
        'Barker-13 {-1,1}': 'red',
        'Barker-13 {0,1}': 'blue'
    }
    
    # Plot autocorrelations
    for name, seq in sequences.items():
        autocorr = compute_autocorrelation(seq)
        ax.plot(autocorr, color=colors[name], label=name)
    
    ax.set_title('Autocorrelation Comparison', fontsize=16)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Correlation')
    
    # Set integer y-ticks for autocorrelation
    y_max = np.ceil(np.max([compute_autocorrelation(seq) for seq in sequences.values()]))
    y_min = np.floor(np.min([compute_autocorrelation(seq) for seq in sequences.values()]))
    ax.set_yticks(np.arange(y_min, y_max + 1, 1))
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    return fig

def evaluate_preamble_detection(sequences, snr_range=np.arange(13, -15, -1)):
    """
    Evaluate preamble detection performance in noisy environments using {-1,1} sequences
    
    Args:
        sequences (dict): Dictionary of {-1,1} sequences to evaluate
        snr_range (array): Range of SNR values in dB to test
    
    Returns:
        DataFrame with detection statistics for each sequence at different SNR levels
    """
    results = []
    num_trials = 5000  # Number of Monte Carlo trials per SNR level
    
    for name, seq in sequences.items():
        seq_len = len(seq)
        # Use 70% of maximum correlation as threshold (max correlation = length of sequence)
        threshold = 0.7 * len(seq)
        
        for snr in snr_range:
            detections = 0
            
            for _ in range(num_trials):
                # Create noisy signal with preamble at random position
                noise_len = 5 * seq_len
                signal = np.zeros(noise_len)
                insert_pos = np.random.randint(0, noise_len - seq_len)
                signal[insert_pos:insert_pos + seq_len] = seq
                
                # Add noise based on SNR
                signal_power = np.mean(seq**2)  # Should be 1 for {-1,1} sequences
                noise_power = signal_power / (10**(snr/10))
                noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
                noisy_signal = signal + noise
                
                # Perform correlation detection
                correlation = correlate(noisy_signal, seq, mode='valid')
                peaks = np.where(correlation > threshold)[0]
                
                # Check if detection is correct
                if any(abs(p - insert_pos) <= 1 for p in peaks):
                    detections += 1
            
            detection_rate = detections / num_trials
            
            results.append({
                'Sequence': name,
                'SNR': snr,
                'Detection Rate': detection_rate
            })
    
    return pd.DataFrame(results)

def plot_detection_performance(results):
    """Plot detection performance results"""
    plt.rcParams.update({'font.size': 12})  # Increase base font size
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for name in results['Sequence'].unique():
        sequence_results = results[results['Sequence'] == name]
        ax.plot(sequence_results['SNR'], sequence_results['Detection Rate'], 
                marker='o', label=name)
    
    ax.set_xlabel('Signal-to-Noise Ratio (dB)', fontsize=12)
    ax.set_ylabel('Detection Rate (probability)', fontsize=12)
    ax.set_title('Detection Rate vs SNR', fontsize=14)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(True)
    ax.legend(fontsize=11)
    ax.set_ylim(0.7, 1.05)
    
    plt.tight_layout()
    return fig

def main():
    # Generate sequences
    barker_bipolar = generate_barker13_bipolar()
    barker_binary = generate_barker13_binary()

    sequences = {
        'Barker-13 {-1,1}': barker_bipolar,
        'Barker-13 {0,1}': barker_binary
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
    
    plt.show()
    
    return df, fig

if __name__ == "__main__":
    main()