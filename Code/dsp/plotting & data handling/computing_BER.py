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
    ber_no_bp = (ham_dist_no_bp / total_bits) * 100
    ber_with_bp = (ham_dist_with_bp / total_bits) * 100
    
    print(f"Number of transmissions: {total_transmissions}")
    print(f"Total bits transmitted: {total_bits}")
    print(f"\nTotal bit errors without bandpass: {ham_dist_no_bp}")
    print(f"Total bit errors with bandpass: {ham_dist_with_bp}")
    print(f"Bit Error Rate without bandpass: {ber_no_bp:.2f}%")
    print(f"Bit Error Rate with bandpass: {ber_with_bp:.2f}% \n")
    
    return ber_no_bp, ber_with_bp

if __name__ == "__main__":
    # Test with both files
    files = ["c:/Users/morte/OneDrive - Danmarks Tekniske Universitet/Bachelor/IOTunderwater/Code/dsp/data/large_test_no_encoding, 100bps, 50 cm.csv"
            ,"c:/Users/morte/OneDrive - Danmarks Tekniske Universitet/Bachelor/IOTunderwater/Code/dsp/data/large_test_no_encoding, 200bps, 50 cm.csv"
            ,"c:/Users/morte/OneDrive - Danmarks Tekniske Universitet/Bachelor/IOTunderwater/Code/dsp/data/large_test_no_encoding, 300bps, 50 cm.csv" 

            ,"c:/Users/morte/OneDrive - Danmarks Tekniske Universitet/Bachelor/IOTunderwater/Code/dsp/data/large_test_hamming_encoding, 100bps, 50 cm.csv" ]

    for file in files:
        ber_no_bp, ber_with_bp = compute_BER(file)