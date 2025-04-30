import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import re


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

        # Convert Hamming distances to integers and handle NaN values
        encoding_df['Hamming Dist without bandpass'] = pd.to_numeric(
            encoding_df['Hamming Dist without bandpass'], errors='coerce').fillna(-1).astype(int)
        encoding_df['Hamming Dist with bandpass'] = pd.to_numeric(
            encoding_df['Hamming Dist with bandpass'], errors='coerce').fillna(-1).astype(int)

        # Count incomplete transmissions (no decoded message and no Hamming distance)
        incomplete_no_bp = encoding_df[
            (encoding_df['Decoded without bandpass'].isna()) & 
            (encoding_df['Hamming Dist without bandpass'] == -1)].shape[0]
        
        incomplete_bp = encoding_df[
            (encoding_df['Decoded with bandpass'].isna()) & 
            (encoding_df['Hamming Dist with bandpass'] == -1)].shape[0]

        # Get the appropriate message length for this encoding
        message_length = message_lengths[encoding]

        # Filter out incomplete transmissions and 'No preamble found'
        valid_entries_no_bp = encoding_df[
            (encoding_df['Decoded without bandpass'] != 'No preamble found') & 
            (encoding_df['Hamming Dist without bandpass'] >= 0)]
        
        valid_entries_bp = encoding_df[
            (encoding_df['Decoded with bandpass'] != 'No preamble found') & 
            (encoding_df['Hamming Dist with bandpass'] >= 0)]
        
        # Sum up Hamming distances
        total_errors_no_bp = valid_entries_no_bp['Hamming Dist without bandpass'].sum()
        total_errors_bp = valid_entries_bp['Hamming Dist with bandpass'].sum()
        
        # Calculate total bits
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
            'incomplete_transmissions_no_bp': incomplete_no_bp,
            'incomplete_transmissions_bp': incomplete_bp,
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
        print(f"Total entries: {total_entries}")
        print(f"Incomplete transmissions: {incomplete_no_bp}")
        print(f"Valid transmissions: {len(valid_entries_no_bp)}")
        print(f"Total errors: {total_errors_no_bp}")
        print(f"Total bits: {total_bits_no_bp}")
        print(f"BER: {ber_no_bp:.2f}%")
        
        print(f"\nWith Bandpass:")
        print(f"Incomplete transmissions: {incomplete_bp}")
        print(f"Valid transmissions: {len(valid_entries_bp)}")
        print(f"Total errors: {total_errors_bp}")
        print(f"Total bits: {total_bits_bp}")
        print(f"BER: {ber_bp:.2f}%")
        print("-" * 50)
    
    return results

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
    base_path = "Code/dsp/data/pool/pool_testing_cf_2000"
    
    # Read both CSV files
    df1 = pd.read_csv(f"{base_path}_100-300bps, 5sd, 50ds.csv")
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
            total_entries = len(bitrate_data)
            
            # Calculate BER for without bandpass
            valid_no_bp = bitrate_data[bitrate_data['Decoded without bandpass'] != 'No preamble found']
            total_bits_no_bp = len(valid_no_bp) * message_length
            total_errors_no_bp = valid_no_bp['Hamming Dist without bandpass'].sum()
            ber_no_bp = (total_errors_no_bp / total_bits_no_bp * 100) if total_bits_no_bp > 0 else 100
            invalid_no_bp = total_entries - len(valid_no_bp)

            # Calculate BER for with bandpass
            valid_bp = bitrate_data[bitrate_data['Decoded with bandpass'] != 'No preamble found']
            total_bits_bp = len(valid_bp) * message_length
            total_errors_bp = valid_bp['Hamming Dist with bandpass'].sum()
            ber_bp = (total_errors_bp / total_bits_bp * 100) if total_bits_bp > 0 else 100
            invalid_bp = total_entries - len(valid_bp)


            print(f"\nBitrate {bitrate} bps:")
            print(f"Total entries: {total_entries}")
            print(f"Invalid entries (No BP): {invalid_no_bp}")
            print(f"Invalid entries (BP): {invalid_bp}")
        
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

def combine_csv_by_carrier_freq(carrier_freq):
    """
    Combine CSV files for a specific carrier frequency
    Args:
        carrier_freq (int): Carrier frequency to combine files for (e.g., 6000)
    """
    base_path = "Code/dsp/data/pool"
    # Get all CSV files matching the carrier frequency
    file_pattern = f"{base_path}/pool_testing_cf{carrier_freq}*.csv"
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"No files found with carrier frequency {carrier_freq}Hz")
        return
    
    # Read and combine all matching files
    dfs = []
    for file in files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"Reading {file}")
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if dfs:
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Create new filename
        new_filename = f"{base_path}/pool_testing_cf{carrier_freq}_all_distances.csv"
        
        # Save combined data
        combined_df.to_csv(new_filename, index=False)
        print(f"Combined {len(files)} files into: {new_filename}")

def compute_ber_by_distance_bitrate(file_path):
    """
    Compute BER for different distances and bitrates, excluding entries with missing Hamming distances
    
    Args:
        file_path (str): Path to CSV file containing transmission data
    
    Returns:
        pd.DataFrame: DataFrame with BER results grouped by distance and bitrate
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert Hamming distances to integers and handle NaN values
    df['Hamming Dist without bandpass'] = pd.to_numeric(df['Hamming Dist without bandpass'], errors='coerce')
    df['Hamming Dist with bandpass'] = pd.to_numeric(df['Hamming Dist with bandpass'], errors='coerce')
    
    # Group by distance and bitrate
    grouped = df.groupby(['Distance to speaker', 'Bitrate'])
    
    results = []
    message_length = 88  # Length of "Hello_there" message in bits
    
    for (distance, bitrate), group in grouped:
        # Count total transmissions
        total_entries = len(group)
        
        # Without bandpass
        valid_no_bp = group[group['Hamming Dist without bandpass'].notna()]
        total_bits_no_bp = len(valid_no_bp) * message_length
        total_errors_no_bp = valid_no_bp['Hamming Dist without bandpass'].sum()
        incomplete_no_bp = total_entries - len(valid_no_bp)
        
        # With bandpass
        valid_bp = group[group['Hamming Dist with bandpass'].notna()]
        total_bits_bp = len(valid_bp) * message_length
        total_errors_bp = valid_bp['Hamming Dist with bandpass'].sum()
        incomplete_bp = total_entries - len(valid_bp)
        
        # Calculate BER
        ber_no_bp = (total_errors_no_bp / total_bits_no_bp * 100) if total_bits_no_bp > 0 else None
        ber_bp = (total_errors_bp / total_bits_bp * 100) if total_bits_bp > 0 else None
        
        results.append({
            'Distance_cm': distance,
            'Bitrate': bitrate,
            'Total_Transmissions': total_entries,
            'Incomplete_No_BP': incomplete_no_bp,
            'Incomplete_BP': incomplete_bp,
            'Valid_Transmissions_No_BP': len(valid_no_bp),
            'Valid_Transmissions_BP': len(valid_bp),
            'Total_Errors_No_BP': total_errors_no_bp,
            'Total_Errors_BP': total_errors_bp,
            'Total_Bits_No_BP': total_bits_no_bp,
            'Total_Bits_BP': total_bits_bp,
            'BER_No_BP': ber_no_bp,
            'BER_With_BP': ber_bp
        })
        
        # Print results for verification
        print(f"\nDistance: {distance} cm, Bitrate: {bitrate} bps")
        print(f"Total transmissions: {total_entries}")
        print(f"Without bandpass:")
        print(f"  Incomplete transmissions: {incomplete_no_bp}")
        print(f"  Valid transmissions: {len(valid_no_bp)}")
        print(f"  Total errors: {total_errors_no_bp}")
        print(f"  Total bits: {total_bits_no_bp}")
        print(f"  BER: {ber_no_bp:.2f}%" if ber_no_bp is not None else "  BER: N/A")
        print(f"With bandpass:")
        print(f"  Incomplete transmissions: {incomplete_bp}")
        print(f"  Valid transmissions: {len(valid_bp)}")
        print(f"  Total errors: {total_errors_bp}")
        print(f"  Total bits: {total_bits_bp}")
        print(f"  BER: {ber_bp:.2f}%" if ber_bp is not None else "  BER: N/A")
        print("-" * 50)
    
    # Create DataFrame and sort by distance and bitrate
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['Distance_cm', 'Bitrate'])
    
    return results_df

def plot_ber_vs_distance_for_the_big_all_distances_files(file_path):
    """
    Create three subplots showing BER vs Distance for each bitrate,
    including bars for incomplete transmissions
    
    Args:
        file_path (str): Path to CSV file containing the BER results
    """
    # Read the results DataFrame
    results_df = pd.read_csv(file_path)
    carrier_freq = re.search(r'cf(\d+)', file_path).group(1)  # Extracts number after 'cf'

    # Create figure with three subplots and space for legend
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle(f'BER vs Distance with incomplete transmissions, carrier frequency: {carrier_freq}', fontsize=14, y=0.95)
    
    # Colors for consistency
    no_bp_color = 'red'
    bp_color = 'blue'
    
    # Store handles for legend
    legend_handles = []
    legend_labels = []
    
    # All distances to show on x-axis
    all_distances = [40, 70, 100, 150, 200, 300, 400, 500, 600]
    
    # Plot for each bitrate
    for idx, bitrate in enumerate([100, 200, 300]):
        # Filter data for this bitrate
        bitrate_data = results_df[results_df['Bitrate'] == bitrate].sort_values('Distance_cm')
        
        ax = axes[idx]
        
        # Plot BER lines
        ax.plot(bitrate_data['Distance_cm'], bitrate_data['BER_No_BP'], 
                'o-', color=no_bp_color, label='Without Bandpass', markersize=8)
        ax.plot(bitrate_data['Distance_cm'], bitrate_data['BER_With_BP'], 
                'o-', color=bp_color, label='With Bandpass', markersize=8)
        
        # Add value labels
        for x, y1, y2 in zip(bitrate_data['Distance_cm'], 
                            bitrate_data['BER_No_BP'], 
                            bitrate_data['BER_With_BP']):
            ax.annotate(f'{y1:.1f}%', (x, y1), textcoords="offset points",
                       xytext=(0, 10), ha='center', color=no_bp_color)
            ax.annotate(f'{y2:.1f}%', (x, y2), textcoords="offset points",
                       xytext=(0, -15), ha='center', color=bp_color)
        
        # Plot incomplete transmission bars
        bar_width = 15
        ax2 = ax.twinx()  # Create second y-axis for bars
        ax2.bar(bitrate_data['Distance_cm'] - bar_width/2, 
                bitrate_data['Incomplete_No_BP'],
                width=bar_width/2, alpha=0.3, color=no_bp_color, label='Incomplete (No BP)')
        ax2.bar(bitrate_data['Distance_cm'], 
                bitrate_data['Incomplete_BP'],
                width=bar_width/2, alpha=0.3, color=bp_color, label='Incomplete (BP)')
        
        # Set x-axis ticks with distances only
        ax.set_xticks(all_distances)
        ax.set_xticklabels([str(d) for d in all_distances])
        
        # Format y-axis with percentage values only
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}'))
        
        # Set labels and title
        ax.set_xlabel('Distance (cm)')
        ax.set_ylabel('BER (%)')
        ax2.set_ylabel('Incomplete Transmissions')
        ax.set_title(f'Bitrate: {bitrate} bps')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set y-axis limits for BER
        ax.set_ylim(0, 80)
        
        # Store handles and labels for legend (only from first subplot)
        if idx == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            legend_handles = lines1 + lines2
            legend_labels = labels1 + labels2

    # Add single legend below title
    fig.legend(legend_handles, legend_labels, 
              loc='upper center', 
              bbox_to_anchor=(0.5, 0.91), # detemine where the legend should be
              ncol=4)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.82) # makes space in the top for the legend
    plt.show()

def compute_ber_for_hamming_encoding_test(file_path):
    """
    Compute BER and preamble detection statistics for Hamming vs No encoding
    
    Args:
        file_path (str): Path to CSV file containing the test results
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    results = {}
    # Message lengths for different encoding types
    message_lengths = {
        'No Encoding': 88,      # Original message length
        'Hamming Encoding': 132  # Same message length, but with error correction
    }
    
    # Process each encoding type
    for encoding in ['No Encoding', 'Hamming Encoding']:
        # Filter for this encoding type
        encoding_df = df[df['Encoding'] == encoding]
        total_transmissions = len(encoding_df)
        
        if total_transmissions == 0:
            print(f"No data found for {encoding}")
            continue
            
        # Count "No preamble found" instances
        no_preamble_no_bp = encoding_df[encoding_df['Decoded without bandpass'] == 'No preamble found'].shape[0]
        no_preamble_bp = encoding_df[encoding_df['Decoded with bandpass'] == 'No preamble found'].shape[0]
        
        # Filter out "No preamble found" for BER calculation
        valid_no_bp = encoding_df[encoding_df['Decoded without bandpass'] != 'No preamble found']
        valid_bp = encoding_df[encoding_df['Decoded with bandpass'] != 'No preamble found']
        
        # Calculate BER for valid transmissions
        message_length = message_lengths[encoding]
        total_bits_no_bp = len(valid_no_bp) * message_length
        total_bits_bp = len(valid_bp) * message_length
        
        # Sum up Hamming distances (errors)
        total_errors_no_bp = valid_no_bp['Hamming Dist without bandpass'].sum()
        total_errors_bp = valid_bp['Hamming Dist with bandpass'].sum()
        
        # Calculate BER
        ber_no_bp = (total_errors_no_bp / total_bits_no_bp * 100) if total_bits_no_bp > 0 else 100
        ber_bp = (total_errors_bp / total_bits_bp * 100) if total_bits_bp > 0 else 100
        
        # Store results
        results[encoding] = {
            'total_transmissions': total_transmissions,
            'no_preamble_no_bp': no_preamble_no_bp,
            'no_preamble_bp': no_preamble_bp,
            'valid_transmissions_no_bp': len(valid_no_bp),
            'valid_transmissions_bp': len(valid_bp),
            'total_errors_no_bp': total_errors_no_bp,
            'total_errors_bp': total_errors_bp,
            'ber_no_bp': ber_no_bp,
            'ber_bp': ber_bp
        }
        
        print(f"\nResults for {encoding}:")
        print(f"Total transmissions: {total_transmissions}")
        print("\nWithout Bandpass:")
        print(f"No preamble found: {no_preamble_no_bp} ({no_preamble_no_bp/total_transmissions*100:.1f}%)")
        print(f"Valid transmissions: {len(valid_no_bp)}")
        print(f"Error ratio: {total_errors_no_bp}/{total_bits_no_bp} = {total_errors_no_bp/total_bits_no_bp:.4f}")
        print(f"BER: {ber_no_bp:.2f}%")
        
        print("\nWith Bandpass:")
        print(f"No preamble found: {no_preamble_bp} ({no_preamble_bp/total_transmissions*100:.1f}%)")
        print(f"Valid transmissions: {len(valid_bp)}")
        print(f"Error ratio: {total_errors_bp}/{total_bits_bp} = {total_errors_bp/total_bits_bp:.4f}")
        print(f"BER: {ber_bp:.2f}%")
        print("-" * 50)
    
    return results

def compute_bit_flip_tendency(reference_array, received_arrays, filter_type=""):
    """
    Analyze and print bit flip tendencies between a reference array and received arrays.
    
    Args:
        reference_array (list): Reference bit array (e.g. [1,0,1,0])
        received_arrays (list): List of received bit arrays to compare against
        filter_type (str): String indicating filter type for printing
    """
    # Initialize counters and tracking
    one_to_zero = 0
    zero_to_one = 0
    flip_positions = []
    total_errors = 0
    valid_arrays = 0
    
    # Validate inputs
    if not isinstance(reference_array, list) or not isinstance(received_arrays, list):
        print("Error: Inputs must be lists")
        return
    
    if not received_arrays:
        print("Error: No received arrays provided")
        return
        
    ref_len = len(reference_array)
    print(f"\nBit Flip Analysis for {filter_type}:")
    print(f"Reference array length: {ref_len}")
    print(f"Number of arrays to analyze: {len(received_arrays)}")
    print("-" * 50)
    
    # Process each received array
    for idx, received in enumerate(received_arrays):
        # Skip arrays with mismatched length
        if len(received) != ref_len:
            print(f"Warning: Array {idx + 1} length mismatch ({len(received)} != {ref_len}), skipping")
            continue
            
        # Initialize per-array counters
        array_one_to_zero = 0
        array_zero_to_one = 0
        array_flips = []
        array_errors = 0
            
        # Compare bits
        for i in range(ref_len):
            if reference_array[i] != received[i]:
                array_errors += 1
                total_errors += 1
                
                if reference_array[i] == 1:
                    array_one_to_zero += 1
                    one_to_zero += 1
                else:
                    array_zero_to_one += 1
                    zero_to_one += 1
                    
                if i not in flip_positions:
                    flip_positions.append(i)
                array_flips.append(i)
        
        valid_arrays += 1
        print(f"\nArray {idx + 1} Results:")
        print(f"  1->0 flips: {array_one_to_zero}")
        print(f"  0->1 flips: {array_zero_to_one}")
        # print(f"  Flip positions: {sorted(array_flips)}")
        print(f"  Total errors: {array_errors}")
        print(f"  Error rate: {array_errors/ref_len:.2%}")
    
    # Print summary statistics
    if valid_arrays > 0:
        error_rate = total_errors / (ref_len * valid_arrays)
        print(f"\nOverall Statistics for {filter_type}:")
        print(f"Total 1->0 flips: {one_to_zero}")
        print(f"Total 0->1 flips: {zero_to_one}")
        # print(f"All flip positions: {sorted(flip_positions)}")
        # print(f"Total errors: {total_errors}")
        # print(f"Average error rate: {error_rate:.2%}")
        # print(f"Valid arrays processed: {valid_arrays}")
    else:
        print(f"\nNo valid arrays were processed for {filter_type}")
    print("=" * 50)

def analyze_bit_flips_from_csv(file_path, id):
    """
    Analyze bit flips for a specific transmission ID from CSV file
    
    Args:
        file_path (str): Path to CSV file
        id (str): Transmission ID to analyze
    """
    try:
        df = pd.read_csv(file_path)
        row = df[df['ID'] == id].iloc[0]
        
        reference = eval(row['Original message in bits'])
        # Get all arrays from both columns
        data_without_bp = eval(row['Data bits without bandpass'])
        data_with_bp = eval(row['Data bits with bandpass'])
        
        print(f"Analysis for transmission ID: {id}")
        # Pass all arrays for analysis
        compute_bit_flip_tendency(reference, data_without_bp, "Without Bandpass")
        compute_bit_flip_tendency(reference, data_with_bp, "With Bandpass")
        
    except IndexError:
        print(f"Error: No transmission found with ID {id}")
    except Exception as e:
        print(f"Error analyzing transmission {id}: {str(e)}")

def analyze_ber_by_carrier_freq(file_path, test_description="Testing: testing impact of similarlity of payloads and barker 13"):
    """
    Analyze BER for each carrier frequency at 500 cm distance for a specific test description
    
    Args:
        file_path (str): Path to CSV file containing transmission data
        test_description (str): Description to filter the data by
    
    Returns:
        pd.DataFrame: DataFrame with BER results grouped by carrier frequency
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    global dist
    dist = 100
    bitrate = 500
    # if dist == 500:
    #     test_description = "Testing: At 5 m now testing for frequency sweep again for power average and BER, keeping stick exactly the same place for next test"
    # else:
    #     test_description = "Testing: Average power purely for check of interference"
        
    # Filter by test description and distance
    df = df[df['Test description'] == test_description]
    df = df[df['Distance to speaker'] == dist]
    df = df[df['Bitrate'] == bitrate]
    
    if len(df) == 0:
        print(f"No data found for test description: {test_description}")
        return None
    
    # Convert Hamming distances to integers and handle NaN values
    df['Hamming Dist without bandpass'] = pd.to_numeric(df['Hamming Dist without bandpass'], errors='coerce')
    df['Hamming Dist with bandpass'] = pd.to_numeric(df['Hamming Dist with bandpass'], errors='coerce')
    
    # Group by carrier frequency
    grouped = df.groupby('Carrier Frequency')
    
    results = []
    message_length = 96  # Length of alternating bit sequence (48 '01' pairs)
    
    for carrier_freq, group in grouped:
        # Count total transmissions
        total_entries = len(group)
        
        # Without bandpass
        valid_no_bp = group[group['Decoded without bandpass'] != 'No preamble found']
        total_bits_no_bp = len(valid_no_bp) * message_length
        total_errors_no_bp = valid_no_bp['Hamming Dist without bandpass'].sum()
        invalid_no_bp = total_entries - len(valid_no_bp)
        
        # With bandpass
        valid_bp = group[group['Decoded with bandpass'] != 'No preamble found']
        total_bits_bp = len(valid_bp) * message_length
        total_errors_bp = valid_bp['Hamming Dist with bandpass'].sum()
        invalid_bp = total_entries - len(valid_bp)
        
        # Calculate BER
        ber_no_bp = (total_errors_no_bp / total_bits_no_bp * 100) if total_bits_no_bp > 0 else None
        ber_bp = (total_errors_bp / total_bits_bp * 100) if total_bits_bp > 0 else None
        
        results.append({
            'Carrier_Frequency': carrier_freq,
            'Total_Transmissions': total_entries,
            'Invalid_No_BP': invalid_no_bp,
            'Invalid_BP': invalid_bp,
            'Valid_Transmissions_No_BP': len(valid_no_bp),
            'Valid_Transmissions_BP': len(valid_bp),
            'Total_Errors_No_BP': total_errors_no_bp,
            'Total_Errors_BP': total_errors_bp,
            'BER_No_BP': ber_no_bp,
            'BER_With_BP': ber_bp,
            'Average_Power': group['Average Power of signal'].mean()
        })
        
        # Print results for verification
        print(f"\nCarrier Frequency: {carrier_freq} Hz at 500 cm")
        print(f"Test description: {test_description}")
        print(f"Total transmissions: {total_entries}")
        print(f"Average signal power: {group['Average Power of signal'].mean():.2f}")
        print("\nWithout bandpass:")
        print(f"  Invalid transmissions: {invalid_no_bp}")
        print(f"  Valid transmissions: {len(valid_no_bp)}")
        print(f"  Total errors: {total_errors_no_bp}")
        print(f"  BER: {ber_no_bp:.2f}%" if ber_no_bp is not None else "  BER: N/A")
        print("\nWith bandpass:")
        print(f"  Invalid transmissions: {invalid_bp}")
        print(f"  Valid transmissions: {len(valid_bp)}")
        print(f"  Total errors: {total_errors_bp}")
        print(f"  BER: {ber_bp:.2f}%" if ber_bp is not None else "  BER: N/A")
        print("-" * 50)
    
    # Create DataFrame and sort by frequency
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Carrier_Frequency')
    
    # Plot results
    plot_carrier_freq_analysis(results_df, test_description)
    
    return results_df

def plot_carrier_freq_analysis(results_df, test_description):
    """
    Create plots showing BER and signal power vs carrier frequency at 500 cm
    Including all frequencies from 1000 to 29000 Hz, with N/A values shown at top
    
    Args:
        results_df (pd.DataFrame): DataFrame containing analysis results
    """
    # Create complete range of frequencies
    all_frequencies = range(1000, 30000, 1000)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Ensure all frequencies are in the DataFrame
    complete_df = pd.DataFrame({'Carrier_Frequency': all_frequencies})
    results_df = pd.merge(complete_df, results_df, on='Carrier_Frequency', how='left')
    
    # Replace NaN with 100 for plotting (will appear at top of plot)
    plot_df = results_df.copy()
    plot_df['BER_No_BP'] = plot_df['BER_No_BP'].fillna(100)
    plot_df['BER_With_BP'] = plot_df['BER_With_BP'].fillna(100)
    
    # Plot BER vs Carrier Frequency
    ax1.plot(plot_df['Carrier_Frequency'], plot_df['BER_No_BP'], 
             'ro-', label='Without Bandpass', markersize=8)
    ax1.plot(plot_df['Carrier_Frequency'], plot_df['BER_With_BP'], 
             'bo-', label='With Bandpass', markersize=8)
    
    # Add value labels
    for x, y1, y2 in zip(plot_df['Carrier_Frequency'], 
                        plot_df['BER_No_BP'], 
                        plot_df['BER_With_BP']):
        if y1 == 100:  # N/A value
            ax1.annotate('N/A', (x, y1), textcoords="offset points",
                        xytext=(0, 10), ha='center', color='red')
        elif pd.notna(y1):
            ax1.annotate(f'{y1:.1f}%', (x, y1), textcoords="offset points",
                        xytext=(0, 10), ha='center', color='red')
            
        if y2 == 100:  # N/A value
            ax1.annotate('N/A', (x, y2), textcoords="offset points",
                        xytext=(0, -15), ha='center', color='blue')
        elif pd.notna(y2):
            ax1.annotate(f'{y2:.1f}%', (x, y2), textcoords="offset points",
                        xytext=(0, -15), ha='center', color='blue')
    
    # Configure BER plot
    ax1.set_xlabel('Carrier Frequency (Hz)')
    ax1.set_ylabel('Bit Error Rate (%)')
    ax1.set_title(f'BER vs Carrier Frequency at {dist} cm')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    ax1.set_ylim(0, 105)  # Make room for N/A labels at top
    
    # Set x-axis ticks for every 1000 Hz
    ax1.set_xticks(list(all_frequencies)[::2])  # Show every other frequency to avoid crowding
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot Average Power vs Carrier Frequency
    ax2.plot(results_df['Carrier_Frequency'], results_df['Average_Power'], 
             'go-', label='Signal Power', markersize=8)
    
    # Add value labels for power where data exists
    for x, y in zip(results_df['Carrier_Frequency'], results_df['Average_Power']):
        if pd.notna(y):
            ax2.annotate(f'{y:.1f}', (x, y), textcoords="offset points",
                        xytext=(0, 10), ha='center', color='green')
        else:
            ax2.annotate('N/A', (x, 0), textcoords="offset points",
                        xytext=(0, 10), ha='center', color='green')
    
    # Configure power plot
    ax2.set_xlabel('Carrier Frequency (Hz)')
    ax2.set_ylabel('Average Signal Power')
    ax2.set_title(f'Signal Power vs Carrier Frequency at {dist} cm')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Set x-axis ticks for every 1000 Hz
    ax2.set_xticks(list(all_frequencies)[::2])  # Show every other frequency to avoid crowding
    ax2.tick_params(axis='x', rotation=45)
    
    # Adjust layout to prevent label overlap
    plt.tight_layout()
    plt.show()


def analyze_invalid_transmissions(csv_path):
    """
    Analyze and plot invalid transmissions per unique original message.
    Invalid transmissions are identified when "Decoded without bandpass" == "No preamble found".

    Args:
        csv_path (str): Path to the CSV file.
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    df = df[df["Original Message"] != ("U" * 12)]

    # Compute number of invalid transmissions per message
    invalid_transmissions = (
        df.groupby('Original Message')
        .apply(lambda x: (x['Decoded without bandpass'] == "No preamble found").sum())
        .reset_index(name='Invalid Transmission Count')
    )

    # Compute total transmissions per message
    total_transmissions = df['Original Message'].value_counts().reset_index()
    total_transmissions.columns = ['Original Message', 'Total Transmissions']

    # Merge results
    result = pd.merge(total_transmissions, invalid_transmissions, on='Original Message')

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(result['Original Message'], result['Invalid Transmission Count'])
    plt.xticks(rotation=90)
    plt.xlabel('Original Message')
    plt.ylabel('Invalid Transmission Count')
    plt.title('Invalid Transmissions per Unique Message')
    plt.tight_layout()
    plt.grid(True, axis='y')
    plt.show()

    return result


if __name__ == "__main__":  
    # # NOTE: used to combine pool sweeps into one file, this only needs to be called once
    # # combine_csv_by_carrier_freq(12000)
    # # the code below also only needs to be called once, it loops through the files
    # for cf in ["6000",  "9000", "12000"]:
    #     df = compute_ber_by_distance_bitrate(f"Code/dsp/data/pool/pool_testing_cf{cf}_all_distances.csv")
    #     df.to_csv(f"Code/dsp/data/pool/ber_results_by_distance_cf{cf}.csv", index=False)

    # # plots the scatter plots for the ones above
    # for cf in ["6000", "9000", "12000"]:
    #     results_file = f"Code/dsp/data/pool/ber_results_by_distance_cf{cf}.csv"
    #     plot_ber_vs_distance_for_the_big_all_distances_files(results_file)

    # # NOTE: compute BER for the comparison of hamming encoding, with valid tranmissions with below
    # file_path = "Code/dsp/data/plastic/csv/SG_plastic_hamming_encoding_testing_cf_6000_400bps, 5sd, 50ds.csv"
    # results = compute_ber_for_hamming_encoding_test(file_path)

    # # NOTE: these file paths are to compare how large the variance is between measurements made at the same settings
    # # one with hamming_encoding and one without. Maybe it could be used to determine some variance ???
    # file_paths = ["Code/dsp/data/plastic/plastic_testing_hamming_encoding_cf6000_200bps, 5sd, 40ds.csv", 
    #             "Code/dsp/data/plastic/plastic_testing_hamming_encoding_v2_cf6000_200bps, 5sd, 40ds.csv", 
    #             "Code/dsp/data/plastic/plastic_testing_hamming_encoding_v3_cf6000_200bps, 5sd, 40ds.csv",
    #             "Code/dsp/data/plastic/plastic_testing_hamming_encoding_v4_cf6000_200bps, 5sd, 40ds.csv"]

    # # file_paths = ["Code/dsp/data/plastic/esp_plastic_testing_v1_cf6000_100-300bps, 5sd, 5ds.csv", 
    # #             "Code/dsp/data/plastic/esp_plastic_testing_v2_cf6000_100-300bps, 5sd, 5ds.csv", 
    # #             "Code/dsp/data/plastic/esp_plastic_testing_v3_cf6000_100-300bps, 5sd, 5ds.csv",
    # #             "Code/dsp/data/plastic/esp_plastic_testing_v4_cf6000_100-300bps, 5sd, 5ds.csv"]
    # plot_ber_comparison_across_files(file_paths)

    # NOTE: use below to compute bit flip tendency of a given wav file - will compute for all_data_bits, raises an error
    # if the length of the received does not match the length of the transmitted
    # id_to_analyze = "01befded-cf21-4380-bb04-8a9f9de48385"
    # analyze_bit_flips_from_csv("Received_data_for_tests.csv", id_to_analyze)

    ## NOTE: use below to find the best carrier freq at 5 meters
    file_path = "1m_distance_payload_barker_similarity_impact.csv"
    results = analyze_ber_by_carrier_freq(file_path, test_description="Testing: testing impact of similarlity of payloads and barker 13")
    # results = analyze_invalid_transmissions(file_path)
