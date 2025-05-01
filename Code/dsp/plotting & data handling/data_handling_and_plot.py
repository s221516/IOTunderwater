import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

def analyze_ber_by_carrier_freq(file_path, dist, bitrate, transmitter, test_description="Testing: testing impact of similarlity of payloads and barker 13"):
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
    
    # if dist == 500:
    #     test_description = "Testing: At 5 m now testing for frequency sweep again for power average and BER, keeping stick exactly the same place for next test"
    # else:
    #     test_description = "Testing: Average power purely for check of interference"
        
    # Filter by test description and distance
    df = df[df['Test description'] == test_description]
    df = df[df['Distance to speaker'] == dist]
    df = df[df['Bitrate'] == bitrate]
    df = df[df['Transmitter'] == transmitter]
    
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
        print(f"\nCarrier Frequency: {carrier_freq} Hz at {dist} cm")
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

def plot_power_vs_distance_by_frequency(csv_path, min_freq_khz, max_freq_khz, test_descript):
    """
    Reads a CSV file and plots Distance vs Average Power for all carrier frequencies
    in the same plot with different colors.

    Parameters:
    - csv_path: path to the CSV file.
    - min_freq_khz: minimum carrier frequency (in kHz) to include.
    - max_freq_khz: maximum carrier frequency (in kHz) to include.
    """
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    # Filter for the specific test description
    mask = df['Test description'] == test_descript
    filtered_df = df[mask]

    # Further filter by the frequency range in kHz
    freq_filtered_df = filtered_df[
        (filtered_df['Carrier Frequency'] >= min_freq_khz) &
        (filtered_df['Carrier Frequency'] <= max_freq_khz)
    ]

    # Get all unique frequencies in that range
    frequencies = sorted(freq_filtered_df['Carrier Frequency'].unique())

    if len(frequencies) == 0:
        print(f"No carrier frequencies found in the range {min_freq_khz}–{max_freq_khz} Hz.")
        return

    # Create color map for different frequencies
    colors = plt.cm.rainbow(np.linspace(0, 1, len(frequencies)))
    
    # Create single large figure
    plt.figure(figsize=(15, 10))

    # Plot data for each frequency
    for freq, color in zip(frequencies, colors):
        subset = freq_filtered_df[freq_filtered_df['Carrier Frequency'] == freq]
        
        # Group by distance and calculate mean power
        avg_power = subset.groupby('Distance to speaker')['Average Power of signal'].agg(['mean']).reset_index()
        avg_power_sorted = avg_power.sort_values('Distance to speaker')
        
        # Print intermediate values
        print(f"\nCarrier Frequency: {freq} Hz")
        print("Distance (cm) | Mean Power")
        print("-" * 30)
        for _, row in avg_power_sorted.iterrows():
            print(f"{row['Distance to speaker']:11.0f} | {row['mean']:10.2f}")

        # Plot averaged points
        plt.plot(
            avg_power_sorted['Distance to speaker'],
            avg_power_sorted['mean'],
            'o-',
            color=color,
            label=f'{freq} Hz',
            markersize=8,
            alpha=0.9
        )

    plt.title('Average Power vs. Distance for Different Carrier Frequencies')
    plt.xlabel('Distance to speaker (cm)')
    plt.ylabel('Average Power of signal')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def compute_ber_for_different_vpps(file_path):
    """
    Compute BER for different VPP values from test data and plot the results
    
    Args:
        file_path (str): Path to CSV file
        
    Returns:
        pd.DataFrame: DataFrame with columns [VPP, BER_No_BP, BER_With_BP, Valid_Transmissions]
    """
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Extract VPP from test description using regex
    df['VPP'] = df['Test description'].str.extract(r'VPP: (\d*\.?\d+)').astype(float)    
    results = []
    message_length = 96  # Length of message in bits
    
    # Group by VPP
    for vpp, group in df.groupby('VPP'):
        total_transmissions = len(group)
        
        # Count invalid transmissions (no preamble found)
        invalid_no_bp = group[group['Decoded without bandpass'] == 'No preamble found'].shape[0]
        invalid_bp = group[group['Decoded with bandpass'] == 'No preamble found'].shape[0]
        
        # Without bandpass
        valid_no_bp = group[group['Decoded without bandpass'] != 'No preamble found']
        total_bits_no_bp = len(valid_no_bp) * message_length
        total_errors_no_bp = pd.to_numeric(valid_no_bp['Hamming Dist without bandpass'], errors='coerce').sum()
        
        # With bandpass
        valid_bp = group[group['Decoded with bandpass'] != 'No preamble found']
        total_bits_bp = len(valid_bp) * message_length
        total_errors_bp = pd.to_numeric(valid_bp['Hamming Dist with bandpass'], errors='coerce').sum()
        
        # Calculate BER
        ber_no_bp = (total_errors_no_bp / total_bits_no_bp * 100) if total_bits_no_bp > 0 else 100
        ber_bp = (total_errors_bp / total_bits_bp * 100) if total_bits_bp > 0 else 100
        
        results.append({
            'VPP': vpp,
            'BER_No_BP': ber_no_bp,
            'BER_With_BP': ber_bp,
            'Invalid_No_BP': invalid_no_bp,
            'Invalid_BP': invalid_bp,
            'Valid_Transmissions_No_BP': len(valid_no_bp),
            'Valid_Transmissions_BP': len(valid_bp),
            'Total_Transmissions': total_transmissions,
            'Average_Power': group['Average Power of signal'].mean()
        })
    
    # Create DataFrame and sort by VPP
    results_df = pd.DataFrame(results).sort_values('VPP')
    
    # Create evenly spaced positions for plotting
    vpp_positions = np.arange(len(results_df))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot BER vs VPP
    ax1.plot(vpp_positions, results_df['BER_No_BP'], 'ro-', label='Without Bandpass', markersize=8)
    ax1.plot(vpp_positions, results_df['BER_With_BP'], 'bo-', label='With Bandpass', markersize=8)
    
    # Add bars for invalid transmissions
    bar_width = 0.2
    ax1_twin = ax1.twinx()
    ax1_twin.bar(vpp_positions - bar_width/2, results_df['Invalid_No_BP'], 
                 width=bar_width, alpha=0.3, color='red', label='Invalid (No BP)')
    ax1_twin.bar(vpp_positions + bar_width/2, results_df['Invalid_BP'], 
                 width=bar_width, alpha=0.3, color='blue', label='Invalid (BP)')
    
    # Add value labels for BER
    for x, y1, y2 in zip(vpp_positions, results_df['BER_No_BP'], results_df['BER_With_BP']):
        ax1.annotate(f'{y1:.1f}%', (x, y1), textcoords="offset points",
                    xytext=(0, 10), ha='center', color='red')
        ax1.annotate(f'{y2:.1f}%', (x, y2), textcoords="offset points",
                    xytext=(0, -15), ha='center', color='blue')
    
    # Add value labels for invalid transmissions
    for x, y1, y2 in zip(vpp_positions, results_df['Invalid_No_BP'], results_df['Invalid_BP']):
        if y1 > 0:
            ax1_twin.annotate(f'{int(y1)}', (x - bar_width/2, y1), textcoords="offset points",
                            xytext=(0, 5), ha='center', color='darkred')
        if y2 > 0:
            ax1_twin.annotate(f'{int(y2)}', (x + bar_width/2, y2), textcoords="offset points",
                            xytext=(0, 5), ha='center', color='darkblue')
    
    # Set x-axis ticks with VPP values
    ax1.set_xticks(vpp_positions)
    ax1.set_xticklabels([f'{vpp}' for vpp in results_df['VPP']])
    
    ax1.set_xlabel('VPP')
    ax1.set_ylabel('Bit Error Rate (%)')
    ax1_twin.set_ylabel('Invalid Transmissions')
    ax1.set_title('BER vs VPP with Invalid Transmissions')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Plot Average Power vs VPP
    ax2.plot(vpp_positions, results_df['Average_Power'], 'go-', label='Average Power', markersize=8)
    
    # Add value labels for power
    for x, y in zip(vpp_positions, results_df['Average_Power']):
        ax2.annotate(f'{y:.1f}', (x, y), textcoords="offset points",
                    xytext=(0, 10), ha='center', color='green')
    
    # Set x-axis ticks for power plot
    ax2.set_xticks(vpp_positions)
    ax2.set_xticklabels([f'{vpp}' for vpp in results_df['VPP']])
    
    ax2.set_xlabel('VPP')
    ax2.set_ylabel('Average Power')
    ax2.set_title('Average Power vs VPP')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return results_df


if __name__ == "__main__":  
    # NOTE: use below to compute bit flip tendency of a given wav file
    # id_to_analyze = "41af8091-ff03-4f17-8567-470ce7141dd2"
    # analyze_bit_flips_from_csv("5m_dist_10kHz_unique_payloads.csv", id_to_analyze)

    ## NOTE: 
    file_path = "Average_power_of_received_signal.csv"
    # file_path = "avg_power_of_rec_signal_purely_for_check_of_interference.csv"

    df = pd.read_csv(file_path)
    print(df["Test description"].unique())
    # dist = 100, 15000 hz
    # dist = 200, 13000 hz
    # dist = 300, 16000 hz
    # dist = 400, 18000 hz
    # dist = 500, 
    dist = 400
    bitrate = 500
    transmitter = "SG"
    # "Testing: Average power purely for check of interference"
    results_df = analyze_ber_by_carrier_freq(file_path, dist, bitrate, transmitter, "Testing: average power of a signal")

    ## NOTE: change to see a subset of carrier freqs, min_freq, max_freq as inputs
    # plot_power_vs_distance_by_frequency(file_path, 1000, 15000, "Testing: average power of a signal")
    
    # # NOTE: below is for the vpp test
    # test_file = "1m_distance_carrier_freq_sg_vpp_variable.csv"
    # result_df = compute_ber_for_different_vpps(test_file)

    ## NOTE: new function
    # plot_carrier_freq_analysis(results_df, "Testing: average power of a signal")
    # results = analyze_ber_by_carrier_freq(file_path, test_description="Testing: testing impact of similarlity of payloads and barker 13")
    # results = analyze_invalid_transmissions(file_path)



    
