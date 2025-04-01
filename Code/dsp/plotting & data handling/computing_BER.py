import pandas as pd
import numpy as np

def compute_BER(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Get total number of transmissions
    total_transmissions = len(df)
    
    # Get Hamming distances (total bit errors)
    ham_dist_no_bp = df["Hamming Dist without bnadpass"].sum()
    ham_dist_with_bp = df["Hamming Dist with bandpass"].sum()
    
    # Calculate total bits transmitted
    # Each transmission is "Hello_there" (10 characters * 8 bits per char * number of transmissions)
    bits_per_message = 101  # 10 chars * 8 bits
    total_bits = total_transmissions * bits_per_message
    
    # Calculate BER
    ber_no_bp = ham_dist_no_bp / total_bits
    ber_with_bp = ham_dist_with_bp / total_bits
    
    print(f"Number of transmissions: {total_transmissions}")
    print(f"Total bits transmitted: {total_bits}")
    print(f"Total bit errors without bandpass: {ham_dist_no_bp}")
    print(f"Total bit errors with bandpass: {ham_dist_with_bp}")
    print(f"\nBit Error Rate without bandpass: {ber_no_bp:.6f}")
    print(f"Bit Error Rate with bandpass: {ber_with_bp:.6f}")
    
    return ber_no_bp, ber_with_bp

if __name__ == "__main__":
    # Test with both files
    file = "c:/Users/morte/OneDrive - Danmarks Tekniske Universitet/Bachelor/IOTunderwater/large_test_no_encoding, 100bps, 50 cm.csv"

    ber_no_bp, ber_with_bp = compute_BER(file)