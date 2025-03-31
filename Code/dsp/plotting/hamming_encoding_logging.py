import pandas as pd

def compute_avg_hamming_distance(file_path):
    # Load the CSV file with automatic delimiter detection
    df = pd.read_csv(file_path, engine='python', sep=None)
    
    # NOTE: ASK THE OTHERS WHAT THE FILTER SHOULD BE HERE
    # Filter out rows where Hamming Dist is above 30
    # df = df[df['Hamming Dist'] <= 30]
    # Compute average Hamming distance based on Encoding value
    avg_hamming_true = df[df['Encoding'] == True]['Hamming Dist'].mean()
    avg_hamming_false = df[df['Encoding'] == False]['Hamming Dist'].mean()
    
    return avg_hamming_true, avg_hamming_false

def count_correct_decodings(file_path):
    # Load the CSV file with automatic delimiter detection
    df = pd.read_csv(file_path, engine='python', sep=None)
    
    # Count occurrences where "Decoded without bandpass" is equal to "Hello_there" based on Encoding value
    count_true = df[(df['Encoding'] == True) & (df['Decoded without bandpass'] == "Hello_there")].shape[0]
    count_false = df[(df['Encoding'] == False) & (df['Decoded without bandpass'] == "Hello_there")].shape[0]
    
    return count_true, count_false

if __name__ == "__main__":
    file_path = "hamming_encoding_test.csv"  # Update path if necessary
    avg_true, avg_false = compute_avg_hamming_distance(file_path)
    print(f"Average Hamming Distance (Encoding=False): {avg_false}")
    print(f"Average Hamming Distance (Encoding=True): {avg_true}")
    
    correct_true, correct_false = count_correct_decodings(file_path)
    print(f"Number of correctly decoded messages (Encoding=False): {correct_false}")
    print(f"Number of correctly decoded messages (Encoding=True): {correct_true}")
