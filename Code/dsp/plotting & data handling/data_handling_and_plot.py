import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast


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
    all_frequencies = range(1000, 20000, 1000)
    
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
    
    # Add bars for invalid transmissions
    bar_width = 200  # Width of bars in Hz
    ax1_twin = ax1.twinx()
    ax1_twin.bar(plot_df['Carrier_Frequency'] - bar_width/2, plot_df['Invalid_No_BP'],
                width=bar_width, alpha=0.3, color='red', label='Invalid (No BP)')
    ax1_twin.bar(plot_df['Carrier_Frequency'] + bar_width/2, plot_df['Invalid_BP'],
                width=bar_width, alpha=0.3, color='blue', label='Invalid (BP)')
    
    # Add value labels for BER
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
    
    # Add value labels for invalid transmissions
    for x, y1, y2 in zip(plot_df['Carrier_Frequency'], 
                        plot_df['Invalid_No_BP'],
                        plot_df['Invalid_BP']):
        if pd.notna(y1) and y1 > 0:
            ax1_twin.annotate(f'{int(y1)}', (x - bar_width/2, y1),
                            textcoords="offset points",
                            xytext=(0, 5), ha='center', color='darkred')
        if pd.notna(y2) and y2 > 0:
            ax1_twin.annotate(f'{int(y2)}', (x + bar_width/2, y2),
                            textcoords="offset points",
                            xytext=(0, 5), ha='center', color='darkblue')
    
    # Configure BER plot
    ax1.set_xlabel('Carrier Frequency (Hz)')
    ax1.set_ylabel('Bit Error Rate (%)')
    ax1_twin.set_ylabel('Number of Invalid Transmissions')
    ax1.set_title(f'BER vs Carrier Frequency with Invalid Transmissions, Distance: {dist} cm, Bitrate: {bitrate} bps')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(0, 105)  # Make room for N/A labels at top
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
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
    ax2.set_title(f'Signal Power vs Carrier Frequency at {dist} cm, Bitrate: {bitrate} bps')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Set x-axis ticks for every 1000 Hz on both plots
    ax1.set_xticks(list(all_frequencies)[::2])
    ax2.set_xticks(list(all_frequencies)[::2])
    ax1.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='x', rotation=45)

    # Adjust layout with more space at top
    plt.subplots_adjust(top=0.95, bottom=0.1, hspace=0.4)
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
    in the same plot with different colors, including error bars.

    Parameters:
    - csv_path: path to the CSV file.
    - min_freq_khz: minimum carrier frequency (in kHz) to include.
    - max_freq_khz: maximum carrier frequency (in kHz) to include.
    - test_descript: test description to filter the data
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
        
        # Group by distance and calculate mean and std power
        stats = subset.groupby('Distance to speaker')['Average Power of signal'].agg(['mean', 'std']).reset_index()
        stats_sorted = stats.sort_values('Distance to speaker')
        
        # Print intermediate values
        print(f"\nCarrier Frequency: {freq} Hz")
        print("Distance (cm) | Mean Power | Std Dev")
        print("-" * 45)
        for _, row in stats_sorted.iterrows():
            print(f"{row['Distance to speaker']:11.0f} | {row['mean']:10.2f} | {row['std']:7.2f}")

        # Plot averaged points with error bars
        plt.errorbar(
            stats_sorted['Distance to speaker'],
            stats_sorted['mean'],
            yerr=stats_sorted['std'],
            fmt='o-',
            color=color,
            label=f'{freq} Hz',
            markersize=8,
            alpha=0.9,
            capsize=5,
            capthick=1,
            elinewidth=1
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


    """
    Analyze and plot BER vs bitrate for different transmitter types (SG and ESP)
    
    Args:
        file_path (str): Path to CSV file containing transmission data
    
    Returns:
        pd.DataFrame: DataFrame with BER results grouped by transmitter and bitrate
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Create figure with subplots for each transmitter
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    results = []
    message_length = 96  # Standard message length
    
    # Process each transmitter type
    for transmitter, transmitter_data in df.groupby('Transmitter'):
        # Group by bitrate
        for bitrate, group in transmitter_data.groupby('Bitrate'):
            # Count total transmissions
            total_entries = len(group)
            
            # Process without bandpass
            valid_no_bp = group[group['Decoded without bandpass'] != 'No preamble found']
            total_bits_no_bp = len(valid_no_bp) * message_length
            total_errors_no_bp = valid_no_bp['Hamming Dist without bandpass'].sum()
            invalid_no_bp = total_entries - len(valid_no_bp)
            
            # Process with bandpass
            valid_bp = group[group['Decoded with bandpass'] != 'No preamble found']
            total_bits_bp = len(valid_bp) * message_length
            total_errors_bp = valid_bp['Hamming Dist with bandpass'].sum()
            invalid_bp = total_entries - len(valid_bp)
            
            # Calculate BER
            ber_no_bp = (total_errors_no_bp / total_bits_no_bp * 100) if total_bits_no_bp > 0 else None
            ber_bp = (total_errors_bp / total_bits_bp * 100) if total_bits_bp > 0 else None
            
            results.append({
                'Transmitter': transmitter,
                'Bitrate': bitrate,
                'Total_Transmissions': total_entries,
                'Invalid_No_BP': invalid_no_bp,
                'Invalid_BP': invalid_bp,
                'BER_No_BP': ber_no_bp,
                'BER_With_BP': ber_bp
            })
    
    # Create DataFrame and sort by bitrate
    results_df = pd.DataFrame(results).sort_values(['Transmitter', 'Bitrate'])
    
    # Plot for each transmitter
    axes = {'SG': ax1, 'ESP': ax2}
    colors = {'No_BP': 'red', 'BP': 'blue'}
    
    for transmitter in ['SG', 'ESP']:
        transmitter_data = results_df[results_df['Transmitter'] == transmitter]
        ax = axes[transmitter]
        
        # Plot BER lines
        ax.plot(transmitter_data['Bitrate'], transmitter_data['BER_No_BP'], 
                'ro-', label='Without Bandpass', markersize=8)
        ax.plot(transmitter_data['Bitrate'], transmitter_data['BER_With_BP'], 
                'bo-', label='With Bandpass', markersize=8)
        
        # Add invalid transmission bars
        bar_width = 50  # Width of bars in bps
        ax_twin = ax.twinx()
        ax_twin.bar(transmitter_data['Bitrate'] - bar_width/2, transmitter_data['Invalid_No_BP'],
                   width=bar_width, alpha=0.3, color='red', label='Invalid (No BP)')
        ax_twin.bar(transmitter_data['Bitrate'] + bar_width/2, transmitter_data['Invalid_BP'],
                   width=bar_width, alpha=0.3, color='blue', label='Invalid (BP)')
        
        # Add labels and styling
        ax.set_xlabel('Bitrate (bps)')
        ax.set_ylabel('Bit Error Rate (%)')
        ax_twin.set_ylabel('Number of Invalid Transmissions')
        ax.set_title(f'BER vs Bitrate - {transmitter} Transmitter')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(0, 105)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Add value annotations
        for x, y1, y2, inv1, inv2 in zip(transmitter_data['Bitrate'], 
                                        transmitter_data['BER_No_BP'],
                                        transmitter_data['BER_With_BP'],
                                        transmitter_data['Invalid_No_BP'],
                                        transmitter_data['Invalid_BP']):
            if pd.notna(y1):
                ax.annotate(f'{y1:.1f}%', (x, y1), textcoords="offset points",
                           xytext=(0, 10), ha='center', color='red')
            if pd.notna(y2):
                ax.annotate(f'{y2:.1f}%', (x, y2), textcoords="offset points",
                           xytext=(0, -15), ha='center', color='blue')
            if inv1 > 0:
                ax_twin.annotate(f'{int(inv1)}', (x - bar_width/2, inv1),
                               textcoords="offset points",
                               xytext=(0, 5), ha='center', color='darkred')
            if inv2 > 0:
                ax_twin.annotate(f'{int(inv2)}', (x + bar_width/2, inv2),
                               textcoords="offset points",
                               xytext=(0, 5), ha='center', color='darkblue')
    
    plt.tight_layout()
    plt.show()
    
    return results_df

def analyze_ber_by_bitrate_and_distance(file_path, only_bandpass=False, compare_hamming=False):
    """
    Create a single plot showing BER vs distance with different lines for each bitrate
    
    Args:
        file_path (str): Path to CSV file containing transmission data
        only_bandpass (bool): If True, only show results with bandpass filtering
        compare_hamming (bool): If True, compare SG with and without Hamming encoding
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Filter for specific bitrates and sort distances
    allowed_bitrates = [300, 500, 1000, 1500, 2000]
    df = df[df['Bitrate'].isin(allowed_bitrates)]
    distances = sorted(df['Distance to speaker'].unique())
    
    # Create figure
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # Define color map for bitrates
    colors = plt.cm.rainbow(np.linspace(0, 1, len(allowed_bitrates)))
    
    # Define markers for different transmitters
    markers = {
        'SG': 'o',  # Circle for SG
        'ESP': 's',  # Square for ESP
        'SG_no_hamming': 'o',  # Circle for SG without Hamming
        'SG_hamming': 's'  # Circle for SG with Hamming
    }
    
    for idx, bitrate in enumerate(allowed_bitrates):
        bitrate_data = df[df['Bitrate'] == bitrate]
        
        # Process data based on comparison type
        if compare_hamming:
            conditions = [
                ('SG_no_hamming', bitrate_data[(bitrate_data['Transmitter'] == 'SG') & 
                                              (bitrate_data['Encoding'] == 'No Encoding')], 96),
                ('SG_hamming', bitrate_data[(bitrate_data['Transmitter'] == 'SG') & 
                                          (bitrate_data['Encoding'] == 'Hamming Encoding')], 128)
            ]
        else:
            conditions = [
                ('SG', bitrate_data[bitrate_data['Transmitter'] == 'SG'], 96),
                ('ESP', bitrate_data[bitrate_data['Transmitter'] == 'ESP'], 96)
            ]
        
        for trans_type, trans_data, msg_length in conditions:
            results = []
            for distance in distances:
                group = trans_data[trans_data['Distance to speaker'] == distance]
                if len(group) == 0:
                    continue
                
                if not only_bandpass:
                    # Calculate BER without bandpass
                    valid_no_bp = group[group['Decoded without bandpass'] != 'No preamble found']
                    total_bits_no_bp = len(valid_no_bp) * msg_length
                    total_errors_no_bp = valid_no_bp['Hamming Dist without bandpass'].sum()
                    ber_no_bp = (total_errors_no_bp / total_bits_no_bp * 100) if total_bits_no_bp > 0 else 100
                
                # Calculate BER with bandpass
                valid_bp = group[group['Decoded with bandpass'] != 'No preamble found']
                total_bits_bp = len(valid_bp) * msg_length
                total_errors_bp = valid_bp['Hamming Dist with bandpass'].sum()
                ber_bp = (total_errors_bp / total_bits_bp * 100) if total_bits_bp > 0 else 100
                
                results.append({
                    'Distance': distance,
                    'BER_BP': ber_bp,
                    'BER_No_BP': ber_no_bp if not only_bandpass else None
                })
            
            if results:
                df_results = pd.DataFrame(results)
                label_base = f'{trans_type} {bitrate}bps'
                
                # Plot bandpass results
                ax.plot(df_results['Distance'], df_results['BER_BP'], 
                       marker=markers[trans_type], linestyle='-', color=colors[idx],
                       label=f'{label_base} (BP)', markersize=8)
                
                if not only_bandpass:
                    # Plot non-bandpass results with same marker but dashed line
                    ax.plot(df_results['Distance'], df_results['BER_No_BP'], 
                           marker=markers[trans_type], linestyle='--', color=colors[idx],
                           label=f'{label_base} (No BP)', markersize=8)
    
    # Configure plot
    ax.set_xlabel('Distance (cm)')
    ax.set_ylabel('Bit Error Rate (%)')
    ax.set_title('Bit Error Rate vs Distance for Different Bitrates')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(0, 80)
    
    # Set x-axis ticks
    ax.set_xticks(distances)
    
    # Add legend with extra space on the right
    plt.subplots_adjust(right=0.85)
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    
    plt.show()

def varying_length_analysis_and_plot(file_path, only_bandpass=False):
    """
    Compute and plot average BER vs message length with error bars
    
    Args:
        file_path (str): Path to CSV file containing transmission data
        only_bandpass (bool): If True, only show results with bandpass filtering
    """
    # Read CSV file
    df = pd.read_csv(file_path)
    
    def safe_eval(x):
        try:
            return ast.literal_eval(x) if pd.notna(x) else None
        except (ValueError, SyntaxError):
            return None
    
    df['Original message in bits'] = df['Original message in bits'].apply(safe_eval)
    df['Data bits without bandpass'] = df['Data bits without bandpass'].apply(safe_eval)
    df['Data bits with bandpass'] = df['Data bits with bandpass'].apply(safe_eval)
    
    results = []
    for index, row in df.iterrows():
        original = row['Original message in bits']
        if original is None:
            continue
            
        msg_length = len(original)
        
        if not only_bandpass:
            # Calculate BER without bandpass
            decoded_without = row['Data bits without bandpass']
            if decoded_without is not None and len(decoded_without) > 0:
                decoded_without = decoded_without[0]
                errors_without = sum(a != b for a, b in zip(original, decoded_without[:len(original)]))
                ber_without = errors_without / msg_length
                invalid_without = 1 if row['Decoded without bandpass'] == 'No preamble found' else 0
            else:
                ber_without = None
                invalid_without = 1
            
        # Calculate BER with bandpass
        decoded_with = row['Data bits with bandpass']
        if decoded_with is not None and len(decoded_with) > 0:
            decoded_with = decoded_with[0]
            errors_with = sum(a != b for a, b in zip(original, decoded_with[:len(original)]))
            ber_with = errors_with / msg_length
            invalid_with = 1 if row['Decoded with bandpass'] == 'No preamble found' else 0
        else:
            ber_with = None
            invalid_with = 1
            
        result = {
            'Message Length': msg_length,
            'BER_BP': ber_with,
            'Invalid_BP': invalid_with
        }
        if not only_bandpass:
            result.update({
                'BER_No_BP': ber_without,
                'Invalid_No_BP': invalid_without
            })
        results.append(result)
    
    # Create DataFrame and calculate statistics
    df_results = pd.DataFrame(results)
    
    # Calculate statistics for each message length
    grouped_stats = df_results.groupby('Message Length').agg({
        'BER_BP': ['mean', 'std'],
        'Invalid_BP': 'sum'
    })
    
    if not only_bandpass:
        no_bp_stats = df_results.groupby('Message Length').agg({
            'BER_No_BP': ['mean', 'std'],
            'Invalid_No_BP': 'sum'
        })
        # Combine the stats
        grouped_stats = pd.concat([grouped_stats, no_bp_stats], axis=1)
    
    # Reset index to make Message Length a column
    stats = grouped_stats.reset_index()
    
    # Create plot
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax_twin = ax.twinx()
    
    x_values = stats['Message Length'].values
    
    # Plot average BER with error bars
    if not only_bandpass:
        ax.errorbar(x_values, stats[('BER_No_BP', 'mean')], 
                   yerr=stats[('BER_No_BP', 'std')],
                   fmt='ro', label='Without bandpass', markersize=8,
                   capsize=5, capthick=1, elinewidth=1)
        
        # Plot invalid transmission bars
        ax_twin.bar(x_values - 2, stats[('Invalid_No_BP', 'sum')],
                   width=3, alpha=0.3, color='red', label='Invalid (No BP)')
    
    # Plot bandpass results
    ax.errorbar(x_values, stats[('BER_BP', 'mean')],
               yerr=stats[('BER_BP', 'std')],
               fmt='bo', label='With bandpass', markersize=8,
               capsize=5, capthick=1, elinewidth=1)
    
    # Plot invalid transmission bars
    ax_twin.bar(x_values + (0 if only_bandpass else 2),
               stats[('Invalid_BP', 'sum')],
               width=3, alpha=0.3, color='blue', label='Invalid (BP)')
    
    # Add value annotations
    for x, y_bp, std_bp, inv_bp in zip(x_values,
                                      stats[('BER_BP', 'mean')],
                                      stats[('BER_BP', 'std')],
                                      stats[('Invalid_BP', 'sum')]):
        if pd.notna(y_bp):
            ax.annotate(f'{y_bp:.2f}', (x, y_bp),
                       textcoords="offset points",
                       xytext=(0, 10), ha='center', color='blue')
        if inv_bp > 0:
            ax_twin.annotate(f'{int(inv_bp)}', (x + (0 if only_bandpass else 2), inv_bp),
                           textcoords="offset points",
                           xytext=(0, 5), ha='center', color='darkblue')
        
        if not only_bandpass:
            y_no_bp = stats.loc[stats['Message Length'] == x, ('BER_No_BP', 'mean')].iloc[0]
            std_no_bp = stats.loc[stats['Message Length'] == x, ('BER_No_BP', 'std')].iloc[0]
            inv_no_bp = stats.loc[stats['Message Length'] == x, ('Invalid_No_BP', 'sum')].iloc[0]
            if pd.notna(y_no_bp):
                ax.annotate(f'{y_no_bp:.2f}', (x, y_no_bp),
                           textcoords="offset points",
                           xytext=(0, -15), ha='center', color='red')
            if inv_no_bp > 0:
                ax_twin.annotate(f'{int(inv_no_bp)}', (x - 2, inv_no_bp),
                               textcoords="offset points",
                               xytext=(0, 5), ha='center', color='darkred')
    
    # Configure plot
    ax.set_xlabel('Message Length (bits)')
    ax.set_ylabel('Bit Error Rate')
    ax_twin.set_ylabel('Invalid Transmissions')
    ax.set_title('Average Bit Error Rate vs Message Length')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(bottom=0)  # Set only bottom limit to 0
    
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.show()

def analyze_bit_flips_by_transmitter(csv_path):
    """
    Analyze bit flip tendencies for all entries grouped by transmitter type (ESP vs SG)
    
    Args:
        csv_path (str): Path to CSV file containing transmission data
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Initialize counters for each transmitter type
    results = {
        'ESP': {'one_to_zero': 0, 'zero_to_one': 0, 'total_bits': 0, 'valid_transmissions': 0},
        'SG': {'one_to_zero': 0, 'zero_to_one': 0, 'total_bits': 0, 'valid_transmissions': 0}
    }
    
    def safe_eval(x):
        try:
            return ast.literal_eval(x) if pd.notna(x) else None
        except (ValueError, SyntaxError):
            return None
    
    # Process each row
    for _, row in df.iterrows():
        transmitter = row['Transmitter']
        original = safe_eval(row['Original message in bits'])
        decoded_no_bp = safe_eval(row['Data bits without bandpass'])
        decoded_bp = safe_eval(row['Data bits with bandpass'])
        
        if original is None:
            continue
            
        # Process without bandpass
        if decoded_no_bp is not None and len(decoded_no_bp) > 0:
            for decoded in decoded_no_bp:
                if len(decoded) == len(original):
                    results[transmitter]['valid_transmissions'] += 1
                    results[transmitter]['total_bits'] += len(original)
                    
                    for orig_bit, dec_bit in zip(original, decoded):
                        if orig_bit != dec_bit:
                            if orig_bit == 1:
                                results[transmitter]['one_to_zero'] += 1
                            else:
                                results[transmitter]['zero_to_one'] += 1
        
        # Process with bandpass
        if decoded_bp is not None and len(decoded_bp) > 0:
            for decoded in decoded_bp:
                if len(decoded) == len(original):
                    results[transmitter]['valid_transmissions'] += 1
                    results[transmitter]['total_bits'] += len(original)
                    
                    for orig_bit, dec_bit in zip(original, decoded):
                        if orig_bit != dec_bit:
                            if orig_bit == 1:
                                results[transmitter]['one_to_zero'] += 1
                            else:
                                results[transmitter]['zero_to_one'] += 1
    
    # Print results
    for transmitter, data in results.items():
        if data['valid_transmissions'] > 0:
            print(f"\nResults for {transmitter}:")
            print(f"Total valid transmissions: {data['valid_transmissions']}")
            print(f"Total bits processed: {data['total_bits']}")
            print(f"1->0 flips: {data['one_to_zero']} ({data['one_to_zero']/data['total_bits']*100:.2f}%)")
            print(f"0->1 flips: {data['zero_to_one']} ({data['zero_to_one']/data['total_bits']*100:.2f}%)")
            total_flips = data['one_to_zero'] + data['zero_to_one']
            print(f"Total bit flips: {total_flips} ({total_flips/data['total_bits']*100:.2f}%)")
            print("-" * 50)

    return results

if __name__ == "__main__":  
    # NOTE: use below to compute bit flip tendency of a given wav file
    # id_to_analyze = "41af8091-ff03-4f17-8567-470ce7141dd2"
    # analyze_bit_flips_from_csv("5m_dist_10kHz_unique_payloads.csv", id_to_analyze)

    ## NOTE: 
    # file_path = "Average_power_of_received_signal.csv"
    # file_path = "avg_power_of_rec_signal_purely_for_check_of_interference.csv"

    # df = pd.read_csv(file_path)
    # print(df["Test description"].unique())
    # dist = 100
    # bitrate = 500
    # transmitter = "ESP"
    # # "Testing: Average power purely for check of interference"
    # # "Testing: average power of a signal"  
    # # "Testing: average power of a signal - ESP"
    # # "Testing: average power of a signal - ESP, reverted back to old code"
    # results_df = analyze_ber_by_carrier_freq(file_path, dist, bitrate, transmitter, "Testing: average power of a signal - ESP, reverted back to old code")

    ## NOTE: change to see a subset of carrier freqs, min_freq, max_freq as inputs
    # plot_power_vs_distance_by_frequency(file_path, 10000, 15000, "Testing: average power of a signal")
    
    # # NOTE: below is for the vpp test
    # test_file = "1m_distance_carrier_freq_sg_vpp_variable.csv"
    # result_df = compute_ber_for_different_vpps(test_file)

    # NOTE: below computes the BER for the variying payload composition test

    # NOTE: below computes BER for the max bitrate using ESP, set only_bandpass = True if you only want to compare bandpass, compare_hamming = True if you want to compare
    # SG with and without hamming. Both cant be true at the same time :) 
    # file_path = "Max_bitrate_at_different_distances_and_best_carrier_freq.csv"
    # results = analyze_ber_by_bitrate_and_distance(file_path, only_bandpass=True, compare_hamming=True)

    # # NOTE: below computes the BER for varying lengths of the message
    # file_path = "Varying_payload_sizes.csv"
    # varying_length_analysis_and_plot(file_path, only_bandpass=True)

    # NOTE: computing bitflip tendency for a given file, computes for ESP and SG
    file_path = "1m_distance_carrier_freq_sg_vpp_variable.csv"
    results = analyze_bit_flips_by_transmitter(file_path)

    # NOTE: below computes random payload composition importance
    # TODO: Figure out how to do the note above :)

    ## NOTE: new function
    # plot_carrier_freq_analysis(results_df, "Testing: average power of a signal")
    # results = analyze_ber_by_carrier_freq(file_path, test_description="Testing: testing impact of similarlity of payloads and barker 13")
    # results = analyze_invalid_transmissions(file_path)



    
