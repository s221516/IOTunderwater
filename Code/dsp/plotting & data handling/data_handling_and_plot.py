import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import re


def combining_pool_data_sweep(carrier_freq_input): 
    # Constants
    MESSAGE_BIT_LENGTH = 88  # "Hello_there" in bits
    CARRIER_FREQ = carrier_freq_input  

    # Find all relevant files
    base_path = "Code/dsp/data/pool"
    file_pattern = f"{base_path}/pool_testing_cf{CARRIER_FREQ}*.csv"
    files = glob.glob(file_pattern)

    # Initialize list to store all results
    all_results = []

    def extract_distance(filename):
        # Extract distance from filename (e.g., "70ds" -> 70)
        match = re.search(r'(\d+)ds', filename)
        return int(match.group(1)) if match else None

    # Process each file
    for file_path in files:
        df = pd.read_csv(file_path)
        distance = extract_distance(os.path.basename(file_path))
        
        # Column names from CSV
        bitrate_col = "Bitrate"
        decoded_no_bp_col = "Decoded without bandpass"
        hamming_no_bp_col = "Hamming Dist without bandpass"
        decoded_bp_col = "Decoded with bandpass"
        hamming_bp_col = "Hamming Dist with bandpass"
        
        # Compute bit errors for both cases
        no_bp_errors = []
        bp_errors = []
        
        for _, row in df.iterrows():
            # Without bandpass
            if pd.isna(row[decoded_no_bp_col]) or pd.isna(row[hamming_no_bp_col]):
                no_bp_errors.append(MESSAGE_BIT_LENGTH)
            else:
                no_bp_errors.append(int(row[hamming_no_bp_col]))
            
            # With bandpass
            if pd.isna(row[decoded_bp_col]) or pd.isna(row[hamming_bp_col]):
                bp_errors.append(MESSAGE_BIT_LENGTH)
            else:
                bp_errors.append(int(row[hamming_bp_col]))
        
        # Add to dataframe
        df["no_bp_errors"] = no_bp_errors
        df["bp_errors"] = bp_errors
        df["total_bits"] = MESSAGE_BIT_LENGTH
        
        # Group by bitrate
        grouped = df.groupby(bitrate_col).agg({
            "no_bp_errors": "sum",
            "bp_errors": "sum",
            "total_bits": "sum"
        })
        
            # Calculate BER for both cases
        grouped["BER_no_bandpass"] = (grouped["no_bp_errors"] / grouped["total_bits"]) * 100
        grouped["BER_with_bandpass"] = (grouped["bp_errors"] / grouped["total_bits"]) * 100
        
        # Add distance information
        for bitrate, row in grouped.iterrows():
            result_dict = {
                'Distance_cm': distance,
                'Bitrate': bitrate,
                'Total_Bits': row['total_bits'],
                'No_BP_Errors': row['no_bp_errors'],
                'BP_Errors': row['bp_errors'],
                'BER_No_BP': round(row['BER_no_bandpass'], 2),
                'BER_With_BP': round(row['BER_with_bandpass'], 2)
            }
            all_results.append(result_dict)
            
            # Print results for verification
            print(f"\nDistance: {distance} cm, Bitrate {bitrate} bps:")
            print(f"Without bandpass: BER = {row['BER_no_bandpass']:.2f}% "
                f"({int(row['no_bp_errors'])}/{int(row['total_bits'])} bits)")
            print(f"With bandpass:    BER = {row['BER_with_bandpass']:.2f}% "
                f"({int(row['bp_errors'])}/{int(row['total_bits'])} bits)")

    # Create final DataFrame and sort by distance and bitrate
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(['Distance_cm', 'Bitrate'])

    # Save to CSV
    output_file = f"Code/dsp/data/pool/all_ber_cf{CARRIER_FREQ}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nAll results saved to: {output_file}")

def plotting_pool_data_from_csv(carrier_freq_input):
    # Load the data
    carrier_freq = carrier_freq_input
    df = pd.read_csv(f"Code/dsp/data/pool/all_ber_cf{carrier_freq}.csv")

    # Create a figure with subplots for each bitrate
    bitrates = df['Bitrate'].unique()
    num_bitrates = len(bitrates)

    # Handle single vs multiple bitrates differently
    if num_bitrates == 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        axes = [ax]  # Make it a list to keep the rest of the code consistent
    else:
        fig, axes = plt.subplots(num_bitrates, 1, figsize=(10, 8))
        axes = axes.flatten()  # Convert to 1D array for consistent indexing

    fig.suptitle(f'BER vs Distance with carrier frequency: {carrier_freq} Hz', fontsize=16)

    for idx, bitrate in enumerate(sorted(bitrates)):
        ax = axes[idx]
        
        # Filter data for current bitrate
        bitrate_data = df[df['Bitrate'] == bitrate]
        
        # Rest of the plotting code remains the same
        # ...existing code...
        # Plot scatter with lines
        ax.plot(bitrate_data['Distance_cm'], bitrate_data['BER_No_BP'], 
                'r.-', label='Without Bandpass', markersize=10)
        ax.plot(bitrate_data['Distance_cm'], bitrate_data['BER_With_BP'], 
                'b.-', label='With Bandpass', markersize=10)
        
        # Makes a threshold line, if we want it
        # ax.axhline(y=5, color='gray', linestyle=':', label='5% Threshold')
        
        for x, y in zip(bitrate_data['Distance_cm'], bitrate_data['BER_No_BP']):
            ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', color='red')
        for x, y in zip(bitrate_data['Distance_cm'], bitrate_data['BER_With_BP']):
            ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                    xytext=(0,-15), ha='center', color='blue')
        
        ax.set_title(f'Bitrate: {bitrate} bps')
        ax.set_xlabel('Distance (cm)')
        ax.set_ylabel('BER (%)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Get BER values at distance 600cm
        ber_at_600 = bitrate_data[bitrate_data['Distance_cm'] == 600]
        if not ber_at_600.empty:
            max_ber_at_600 = max(ber_at_600['BER_No_BP'].iloc[0], 
                                ber_at_600['BER_With_BP'].iloc[0])
            # Place legend based on BER at 600cm
            if max_ber_at_600 > 47:
                ax.legend(loc='lower right')
            else:
                ax.legend(loc='upper right')
        else:
            # Default to lower right if no data at 600cm
            ax.legend(loc='lower right')
        
        # ax.set_ylim(-1, max(max(bitrate_data['BER_No_BP']), 
        #                 max(bitrate_data['BER_With_BP'])) + 5)
        
        unique_distances = sorted(df['Distance_cm'].unique())
        ax.set_xticks(unique_distances)
        ax.set_xticklabels([str(int(d)) for d in unique_distances])

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

def compute_ber(file_path):
    """
    Compute BER for entries with 'No Encoding' and 'Hamming Encoding'
    Returns BER for both with and without bandpass filtering
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    results = {}

    message_lengths = {
        'No Encoding': 88,  # Original message length
        'Hamming Encoding': 132  # Message length with Hamming encoding
    }
    

    for encoding in ['No Encoding', 'Hamming Encoding']:
        # Filter for specific encoding entries
        encoding_df = df[df['Encoding'] == encoding]
        
        # Count total entries
        total_entries = len(encoding_df)
        
        if total_entries == 0:
            print(f"No entries found with '{encoding}'")
            continue
        
        # Get the appropriate message length for this encoding
        message_length = message_lengths[encoding]

        # Calculate total errors (excluding 'No preamble found' entries)
        valid_entries_no_bp = encoding_df[encoding_df['Decoded without bandpass'] != 'No preamble found']
        valid_entries_bp = encoding_df[encoding_df['Decoded with bandpass'] != 'No preamble found']
        
        # Sum up Hamming distances
        total_errors_no_bp = valid_entries_no_bp['Hamming Dist without bandpass'].sum()
        total_errors_bp = valid_entries_bp['Hamming Dist with bandpass'].sum()
        
        # Calculate total bits (assuming each valid transmission has same length)
        total_bits_no_bp = len(valid_entries_no_bp) * message_length
        total_bits_bp = len(valid_entries_bp) * message_length       
        # Calculate BER
        ber_no_bp = (total_errors_no_bp / total_bits_no_bp) * 100 if total_bits_no_bp > 0 else 100
        ber_bp = (total_errors_bp / total_bits_bp) * 100 if total_bits_bp > 0 else 100
        
        # Store results
        results[encoding] = {
            'ber_no_bp': ber_no_bp,
            'ber_bp': ber_bp,
            'total_transmissions': total_entries,
            'valid_transmissions_no_bp': len(valid_entries_no_bp),
            'valid_transmissions_bp': len(valid_entries_bp),
            'total_errors_no_bp': total_errors_no_bp,
            'total_errors_bp': total_errors_bp,
            'total_bits_no_bp': total_bits_no_bp,
            'total_bits_bp': total_bits_bp
        }
        
        # Print results for each encoding
        print(f"\nBER Analysis for {encoding}:")
        print(f"\nWithout Bandpass:")
        print(f"Total errors: {total_errors_no_bp}")
        print(f"Total bits: {total_bits_no_bp}")
        print(f"BER: {ber_no_bp:.2f}%")
        print(f"\nWith Bandpass:")
        print(f"Total errors: {total_errors_bp}")
        print(f"Total bits: {total_bits_bp}")
        print(f"BER: {ber_bp:.2f}%")
        print("-" * 50)
    
    return results

def compute_and_save_multiple_bers(file_paths, output_path):
    """
    Compute BER for multiple files and save results to CSV
    """
    all_results = []
    
    for file_path in file_paths:
        # Extract bitrate from filename using regex
        bitrate_match = re.search(r'(\d+)bps', file_path)
        bitrate = int(bitrate_match.group(1)) if bitrate_match else None
        
        results = compute_ber(file_path)
        
        # For each encoding type in results
        for encoding_type, data in results.items():
            result_row = {
                'Bitrate': bitrate,
                'Encoding': encoding_type,
                'BER_No_BP': data['ber_no_bp'],
                'BER_With_BP': data['ber_bp'],
                'Total_Transmissions': data['total_transmissions'],
                'Valid_Transmissions_No_BP': data['valid_transmissions_no_bp'],
                'Valid_Transmissions_BP': data['valid_transmissions_bp'],
                'Total_Errors_No_BP': data['total_errors_no_bp'],
                'Total_Errors_BP': data['total_errors_bp'],
                'Total_Bits_No_BP': data['total_bits_no_bp'],
                'Total_Bits_BP': data['total_bits_bp']
            }
            all_results.append(result_row)
    
    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    return results_df

def plot_ber_vs_bitrate(results_df, distance=40, carrier_freq=6000):
    """
    Create scatter plot of BER vs Bitrate for different encoding types with connecting lines and labels
    """
    plt.figure(figsize=(12, 7))  # Increased figure size for better label visibility
    
    # Plot No Encoding points and lines
    no_encoding = results_df[results_df['Encoding'] == 'No Encoding'].sort_values('Bitrate')
    plt.plot(no_encoding['Bitrate'], no_encoding['BER_No_BP'], 
            'r-', alpha=0.3)  # Line
    plt.plot(no_encoding['Bitrate'], no_encoding['BER_With_BP'], 
            'b-', alpha=0.3)  # Line
    
    # Scatter plots for No Encoding
    plt.scatter(no_encoding['Bitrate'], no_encoding['BER_No_BP'], 
               color='red', marker='o', s=100, label='No Encoding (No Bandpass)')
    plt.scatter(no_encoding['Bitrate'], no_encoding['BER_With_BP'], 
               color='blue', marker='o', s=100, label='No Encoding (Bandpass)')
    
    # Add labels for No Encoding
    for x, y in zip(no_encoding['Bitrate'], no_encoding['BER_No_BP']):
        plt.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', color='red')
    for x, y in zip(no_encoding['Bitrate'], no_encoding['BER_With_BP']):
        plt.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                    xytext=(0,-15), ha='center', color='blue')
    
    # Plot Hamming Encoding points and lines
    hamming = results_df[results_df['Encoding'] == 'Hamming Encoding'].sort_values('Bitrate')
    plt.plot(hamming['Bitrate'], hamming['BER_No_BP'], 
            'r--', alpha=0.3)  # Dashed line
    plt.plot(hamming['Bitrate'], hamming['BER_With_BP'], 
            'b--', alpha=0.3)  # Dashed line
    
    # Scatter plots for Hamming
    plt.scatter(hamming['Bitrate'], hamming['BER_No_BP'], 
               color='red', marker='^', s=100, label='Hamming (No Bandpass)')
    plt.scatter(hamming['Bitrate'], hamming['BER_With_BP'], 
               color='blue', marker='^', s=100, label='Hamming (Bandpass)')
    
    # Add labels for Hamming
    for x, y in zip(hamming['Bitrate'], hamming['BER_No_BP']):
        plt.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', color='red')
    for x, y in zip(hamming['Bitrate'], hamming['BER_With_BP']):
        plt.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                    xytext=(0,-15), ha='center', color='blue')
    
    # Add labels and title
    plt.xlabel('Bitrate (bps)')
    plt.ylabel('Bit Error Rate (%)')
    plt.title(f'BER vs Bitrate at {distance}cm, {carrier_freq}Hz Carrier')
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.gca().invert_yaxis()
    # Show plot
    plt.tight_layout()
    plt.show()

def plot_ber_comparison_across_files(file_paths):
    """
    Compute BER for multiple files and create a scatter plot comparing
    Hamming encoding and no encoding, with and without bandpass filtering.
    """
    all_results = []

    for file_path in file_paths:
        # Extract short file name for labeling
        file_label = "v1"  # default
        if "_v2_" in file_path:
            file_label = "v2"
        elif "_v3_" in file_path:
            file_label = "v3"
        elif "v4" in file_path:
            file_label = "v4"
        
        # Compute BER for the file
        results = compute_ber(file_path)
        
        # Add results to the list
        for encoding_type, data in results.items():
            # Keep as percentage
            ber_no_bp = data['ber_no_bp']
            ber_with_bp = data['ber_bp']
            
            all_results.append({
                'File': file_label,
                'Encoding': encoding_type,
                'BER_No_BP': ber_no_bp,
                'BER_With_BP': ber_with_bp
            })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(all_results)

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    
    # Plot No Encoding
    no_encoding = results_df[results_df['Encoding'] == 'No Encoding']
    plt.scatter(no_encoding['File'], no_encoding['BER_No_BP'], 
                color='red', marker='o', s=100, label='No Encoding (No BP)')
    plt.scatter(no_encoding['File'], no_encoding['BER_With_BP'], 
                color='blue', marker='o', s=100, label='No Encoding (BP)')
    
    # Plot Hamming Encoding
    hamming = results_df[results_df['Encoding'] == 'Hamming Encoding']
    plt.scatter(hamming['File'], hamming['BER_No_BP'], 
                color='red', marker='^', s=100, label='Hamming (No BP)')
    plt.scatter(hamming['File'], hamming['BER_With_BP'], 
                color='blue', marker='^', s=100, label='Hamming (BP)')
    
    # Add value labels
    for df, marker_offset in [(no_encoding, 10), (hamming, -15)]:
        for idx, row in df.iterrows():
            # Label for No BP
            plt.annotate(f'{row["BER_No_BP"]:.1f}%', 
                        (row['File'], row['BER_No_BP']),
                        xytext=(0, 10), textcoords="offset points",
                        ha='center', color='red')
            # Label for With BP
            plt.annotate(f'{row["BER_With_BP"]:.1f}%',
                        (row['File'], row['BER_With_BP']),
                        xytext=(0, -15), textcoords="offset points",
                        ha='center', color='blue')
    
    # Add labels and title
    plt.xlabel('Version')
    plt.ylabel('Bit Error Rate (%)')
    plt.title('BER Comparison Across Different Versions')
    
    # Set y-axis limits and grid
    plt.ylim(0, 15)  # 0% to 100%
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

def combine_csv_versions():
    """Combine v1 and v2 CSV files into a single file without version numbers"""
    base_path = "Code/dsp/data/plastic"
    
    # Read both CSV files
    df1 = pd.read_csv(f"{base_path}/SG_plastic_testing_cf6000_100-300bps, 5sd, 50ds.csv")
    df2 = pd.read_csv(f"{base_path}/SG_plastic_testing_cf6000_400bps, 5sd, 50ds.csv")
    
    # Combine the dataframes
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Create new filename without version number
    new_filename = f"{base_path}/ESP_plastic_testing_cf6000_100-400bps, 5sd, 50ds.csv"
    
    # Save combined data
    combined_df.to_csv(new_filename, index=False)
    print(f"Combined file saved as: {new_filename}")

def compare_esp_and_sg_ber():
    """Compare BER between ESP and Signal Generator implementations"""
    # Read both CSV files
    base_path = "Code/dsp/data/plastic"
    esp_df = pd.read_csv(f"{base_path}/ESP_plastic_testing_cf6000_100-300bps, 5sd, 50ds.csv")
    sg_df = pd.read_csv(f"{base_path}/SG_plastic_testing_cf6000_100-400bps, 5sd, 50ds.csv")

    # Filter out 400 bps from Signal Generator data
    sg_df = sg_df[sg_df['Bitrate'] != 400]

    # Process each dataframe
    def process_df(df, message_length=88):
        results = []
        for bitrate in sorted(df['Bitrate'].unique()):
            bitrate_data = df[df['Bitrate'] == bitrate]
            
            # Calculate BER for without bandpass
            valid_no_bp = bitrate_data[bitrate_data['Decoded without bandpass'] != 'No preamble found']
            total_bits_no_bp = len(valid_no_bp) * message_length
            total_errors_no_bp = valid_no_bp['Hamming Dist without bandpass'].sum()
            ber_no_bp = (total_errors_no_bp / total_bits_no_bp * 100) if total_bits_no_bp > 0 else 100
            
            # Calculate BER for with bandpass
            valid_bp = bitrate_data[bitrate_data['Decoded with bandpass'] != 'No preamble found']
            total_bits_bp = len(valid_bp) * message_length
            total_errors_bp = valid_bp['Hamming Dist with bandpass'].sum()
            ber_bp = (total_errors_bp / total_bits_bp * 100) if total_bits_bp > 0 else 100
            
            results.append({
                'Bitrate': bitrate,
                'BER_No_BP': ber_no_bp,
                'BER_With_BP': ber_bp
            })
        return pd.DataFrame(results)

    # Process both datasets
    esp_results = process_df(esp_df)
    sg_results = process_df(sg_df)

    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot SG data
    plt.plot(sg_results['Bitrate'], sg_results['BER_No_BP'], 'r.-', label='SG (No BP)', markersize=10)
    plt.plot(sg_results['Bitrate'], sg_results['BER_With_BP'], 'b.-', label='SG (With BP)', markersize=10)
    
    # Plot ESP data
    plt.plot(esp_results['Bitrate'], esp_results['BER_No_BP'], 'g.-', label='ESP (No BP)', markersize=10)
    plt.plot(esp_results['Bitrate'], esp_results['BER_With_BP'], 'k.-', label='ESP (With BP)', markersize=10)
    
    # Add labels and annotations
    for df, color1, color2 in [(sg_results, 'red', 'blue'), (esp_results, 'green', 'black')]:
        for x, y1, y2 in zip(df['Bitrate'], df['BER_No_BP'], df['BER_With_BP']):
            plt.annotate(f'{y1:.1f}%', (x, y1), textcoords="offset points", 
                        xytext=(0,10), ha='center', color=color1)
            plt.annotate(f'{y2:.1f}%', (x, y2), textcoords="offset points", 
                        xytext=(0,-15), ha='center', color=color2)

    plt.xlabel('Bitrate (bps)')
    plt.ylabel('Bit Error Rate (%)')
    plt.title('BER Comparison: ESP vs Signal Generator')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":  
    # combine_csv_versions()
    compare_esp_and_sg_ber()
    # file_paths = [
    #     "Code/dsp/data/plastic/plastic_testing_hamming_encoding_cf6000_100bps, 5sd, 40ds.csv",
    #     "Code/dsp/data/plastic/plastic_testing_hamming_encoding_cf6000_200bps, 5sd, 40ds.csv",
    #     "Code/dsp/data/plastic/plastic_testing_hamming_encoding_cf6000_300bps, 5sd, 40ds.csv",
    #     "Code/dsp/data/plastic/plastic_testing_hamming_encoding_cf6000_400bps, 5sd, 40ds.csv",
    #     "Code/dsp/data/plastic/plastic_testing_hamming_encoding_cf6000_500bps, 5sd, 40ds.csv",
    #     "Code/dsp/data/plastic/plastic_testing_hamming_encoding_cf6000_600bps, 5sd, 40ds.csv"
    # ]
    
    # output_path = "Code/dsp/data/plastic/combined_ber_results.csv"
    # results_df = compute_and_save_multiple_bers(file_paths, output_path)
    # plot_ber_vs_bitrate(results_df)

    # carrier_freq_input = 12000
    # combining_pool_data_sweep(carrier_freq_input)
    # plotting_pool_data_from_csv(carrier_freq_input)


    # file_paths = ["Code/dsp/data/plastic/plastic_testing_hamming_encoding_cf6000_200bps, 5sd, 40ds.csv", 
    #             "Code/dsp/data/plastic/plastic_testing_hamming_encoding_v2_cf6000_200bps, 5sd, 40ds.csv", 
    #             "Code/dsp/data/plastic/plastic_testing_hamming_encoding_v3_cf6000_200bps, 5sd, 40ds.csv",
    #             "Code/dsp/data/plastic/plastic_testing_hamming_encoding_v4_cf6000_200bps, 5sd, 40ds.csv"]
    
    # file_paths = ["Code/dsp/data/plastic/esp_plastic_testing_v1_cf6000_100-300bps, 5sd, 5ds.csv", 
    #             "Code/dsp/data/plastic/esp_plastic_testing_v2_cf6000_100-300bps, 5sd, 5ds.csv", 
    #             "Code/dsp/data/plastic/esp_plastic_testing_v3_cf6000_100-300bps, 5sd, 5ds.csv",
    #             "Code/dsp/data/plastic/esp_plastic_testing_v4_cf6000_100-300bps, 5sd, 5ds.csv"]

    # plot_ber_comparison_across_files(file_paths)