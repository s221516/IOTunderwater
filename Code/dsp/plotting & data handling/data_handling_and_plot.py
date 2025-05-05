import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects  # Add this import at the top
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

def analyze_ber_by_carrier_freq(file_path, dist, bitrate, transmitter, test_description, show_error_bars):
    """
    Analyze BER for each carrier frequency at different distances for a specific test description
    
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
        

        # Recalculate standard deviations properly
        if total_bits_no_bp > 0:
            errors_array_no_bp = valid_no_bp['Hamming Dist without bandpass'].values
            ber_array_no_bp = (errors_array_no_bp / message_length)  # Convert to percentage
            ber_no_bp_std = np.std(ber_array_no_bp) / np.sqrt(len(errors_array_no_bp))  # Standard error
        else:
            ber_no_bp_std = None
            
        if total_bits_bp > 0:
            errors_array_bp = valid_bp['Hamming Dist with bandpass'].values
            ber_array_bp = (errors_array_bp / message_length)  # Convert to percentage
            ber_bp_std = np.std(ber_array_bp) / np.sqrt(len(errors_array_bp))  # Standard error
        else:
            ber_bp_std = None

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
            'Average_Power': group['Average Power of signal'].mean(),
            'Power_Std': group['Average Power of signal'].std(),
            'BER_No_BP_Std': ber_no_bp_std,
            'BER_BP_Std': ber_bp_std
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
    plot_carrier_freq_analysis(results_df, test_description, show_error_bars)
    
    return results_df

def plot_carrier_freq_analysis(results_df, test_description, show_error_bars):
    """
    Create plots showing BER and signal power vs carrier frequency at 500 cm
    Including all frequencies from 1000 to 29000 Hz, with N/A values shown at top
    
    Args:
        results_df (pd.DataFrame): DataFrame containing analysis results
    """
    # Create complete range of frequencies
    all_frequencies = range(1000, 20000, 1000)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))
    
    # Ensure all frequencies are in the DataFrame
    complete_df = pd.DataFrame({'Carrier_Frequency': all_frequencies})
    results_df = pd.merge(complete_df, results_df, on='Carrier_Frequency', how='left')
    
    # Replace NaN with 100 for plotting (will appear at top of plot)
    plot_df = results_df.copy()
    plot_df['BER_No_BP'] = plot_df['BER_No_BP'] / 100
    plot_df['BER_With_BP'] = plot_df['BER_With_BP'] / 100
    
    # Plot BER vs Carrier Frequency with optional error bars
    if show_error_bars:
        ax1.errorbar(plot_df['Carrier_Frequency'], plot_df['BER_No_BP'], 
                    yerr=plot_df['BER_No_BP_Std'],
                    fmt='ro-', label='Without Bandpass', markersize=8,
                    capsize=5, capthick=1, elinewidth=1)
        ax1.errorbar(plot_df['Carrier_Frequency'], plot_df['BER_With_BP'], 
                    yerr=plot_df['BER_BP_Std'],
                    fmt='bo-', label='With Bandpass', markersize=8,
                    capsize=5, capthick=1, elinewidth=1)
    else:
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
        if y1 == 1:  # N/A value
            ax1.annotate('N/A', (x, y1), textcoords="offset points",
                        xytext=(0, 10), ha='center', color='red')
        elif pd.notna(y1):
            ax1.annotate(f'{y1:.2f}', (x, y1), textcoords="offset points",
                        xytext=(0, 10), ha='center', color='red')
            
        if y2 == 1:  # N/A value
            ax1.annotate('N/A', (x, y2), textcoords="offset points",
                        xytext=(0, -15), ha='center', color='blue')
        elif pd.notna(y2):
            ax1.annotate(f'{y2:.2f}', (x, y2), textcoords="offset points",
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
    ax1.set_ylim(-0.05, 1.05)  # Make room for N/A labels at top
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Set x-axis ticks for every 1000 Hz
    ax1.set_xticks(list(all_frequencies))  # Show every other frequency to avoid crowding
    ax1.tick_params(axis='x', rotation=45)

    
    if show_error_bars:
        ax2.errorbar(results_df['Carrier_Frequency'], results_df['Average_Power'],
                    yerr=results_df['Power_Std'],
                    fmt='go-', label='Signal Power', markersize=8,
                    capsize=5, capthick=1, elinewidth=1)
    else:
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

    ymin, ymax = ax2.get_ylim()
    ax2.set_ylim(ymin, ymax * 1.1)
    
    # Set x-axis ticks for every 1000 Hz on both plots
    ax1.set_xticks(list(all_frequencies))
    ax2.set_xticks(list(all_frequencies))
    ax1.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='x', rotation=45)

    # Adjust layout with more space at top
    plt.subplots_adjust(top=0.95, bottom=0.1, hspace=0.4)  # Increased hspace from 0.4 to 0.5
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
    plt.figure(figsize=(14, 10))

    # Increase font size for all text elements
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })

    for freq, color in zip(frequencies, colors):
        subset = freq_filtered_df[freq_filtered_df['Carrier Frequency'] == freq]
        
        # Group by distance and calculate statistics
        stats = (subset.groupby('Distance to speaker')['Average Power of signal']
                .agg(['mean', 'std', 'count'])
                .reset_index())
        stats_sorted = stats.sort_values('Distance to speaker')
        
        # Print intermediate values including count
        print(f"\nCarrier Frequency: {freq} Hz")
        print("Distance (cm) | Mean Power | Std Dev  | Data Points")
        print("-" * 60)
        for _, row in stats_sorted.iterrows():
            print(f"{row['Distance to speaker']:11.0f} | {row['mean']:10.2f} | {row['std']:7.2f} | {row['count']:>11f}")

        # Plot averaged points with error bars
        plt.errorbar(
            stats_sorted['Distance to speaker'],
            stats_sorted['mean'],
            yerr=stats_sorted['std'],
            fmt='o-',
            color=color,
            label=f'{freq} Hz',
            markersize=8,
            alpha=1,
            capsize=5,
            capthick=1,
            elinewidth=1.5
        )
    

    # Get current axis limits
    ax = plt.gca()
    
    # Add 50 to y-ticks if not already present
    xticks = list(ax.get_xticks())
    if 50 not in xticks:
        xticks.append(50)
        xticks.sort()
        ax.set_xticks(xticks)
    ax.set_xlim(0,650)

        # Set labels and title with increased font sizes
    plt.title('Average Power vs. Distance for Different Carrier Frequencies', 
              fontsize=16, pad=20)
    plt.xlabel('Distance to speaker (cm)', fontsize=14, labelpad=10)
    plt.ylabel('Average Power of signal', fontsize=14, labelpad=10)
    
    # Configure legend with increased font size
    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left', 
              fontsize=12, frameon=True)
    
    # Add grid with specified style
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Adjust layout to prevent text overlap
    plt.tight_layout()
    plt.show()

def compute_ber_for_different_vpps(file_path):
    """
    Compute BER for different VPP values from test data and plot the results
    with error bars for both BER and power measurements.
    
    Args:
        file_path (str): Path to CSV file
        
    Returns:
        pd.DataFrame: DataFrame with analysis results
    """
    # Read CSV file and extract VPP values
    df = pd.read_csv(file_path)
    df['VPP'] = df['Test description'].str.extract(r'VPP: (\d*\.?\d+)').astype(float)    
    results = []
    message_length = 96  # Length of message in bits
    
    # Group by VPP
    for vpp, group in df.groupby('VPP'):
        total_transmissions = len(group)
        
        # Process without bandpass
        valid_no_bp = group[group['Decoded without bandpass'] != 'No preamble found']
        invalid_no_bp = group[group['Decoded without bandpass'] == 'No preamble found'].shape[0]
        
        if len(valid_no_bp) > 0:
            # Calculate individual transmission BERs for standard error
            ber_array_no_bp = []
            for _, row in valid_no_bp.iterrows():
                errors = pd.to_numeric(row['Hamming Dist without bandpass'], errors='coerce')
                if pd.notna(errors):
                    ber = (errors / message_length) * 100
                    ber_array_no_bp.append(ber)
            
            ber_no_bp = np.mean(ber_array_no_bp)
            ber_no_bp_std = np.std(ber_array_no_bp) / np.sqrt(len(ber_array_no_bp))
        else:
            ber_no_bp = 100
            ber_no_bp_std = 0
        
        # Process with bandpass
        valid_bp = group[group['Decoded with bandpass'] != 'No preamble found']
        invalid_bp = group[group['Decoded with bandpass'] == 'No preamble found'].shape[0]
        
        if len(valid_bp) > 0:
            # Calculate individual transmission BERs for standard error
            ber_array_bp = []
            for _, row in valid_bp.iterrows():
                errors = pd.to_numeric(row['Hamming Dist with bandpass'], errors='coerce')
                if pd.notna(errors):
                    ber = (errors / message_length) * 100
                    ber_array_bp.append(ber)
            
            ber_bp = np.mean(ber_array_bp)
            ber_bp_std = np.std(ber_array_bp) / np.sqrt(len(ber_array_bp))
        else:
            ber_bp = 100
            ber_bp_std = 0
        
        # Calculate power statistics
        power_mean = group['Average Power of signal'].mean()
        power_std = group['Average Power of signal'].std() / np.sqrt(len(group))
        
        results.append({
            'VPP': vpp,
            'BER_No_BP': ber_no_bp,
            'BER_With_BP': ber_bp,
            'BER_No_BP_Std': ber_no_bp_std,
            'BER_BP_Std': ber_bp_std,
            'Invalid_No_BP': invalid_no_bp,
            'Invalid_BP': invalid_bp,
            'Valid_Transmissions_No_BP': len(valid_no_bp),
            'Valid_Transmissions_BP': len(valid_bp),
            'Total_Transmissions': total_transmissions,
            'Average_Power': power_mean,
            'Power_Std': power_std
        })
    
    # Create DataFrame and sort by VPP
    results_df = pd.DataFrame(results).sort_values('VPP')
    
    # Create evenly spaced positions for plotting
    vpp_positions = np.arange(len(results_df))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot BER vs VPP with error bars
    ax1.errorbar(vpp_positions, results_df['BER_No_BP'], 
                yerr=results_df['BER_No_BP_Std'],
                fmt='ro-', label='Without Bandpass', markersize=8,
                capsize=5, capthick=1, elinewidth=1)
    ax1.errorbar(vpp_positions, results_df['BER_With_BP'], 
                yerr=results_df['BER_BP_Std'],
                fmt='bo-', label='With Bandpass', markersize=8,
                capsize=5, capthick=1, elinewidth=1)
    
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
    
    # Configure BER plot
    ax1.set_xticks(vpp_positions)
    ax1.set_xticklabels([f'{vpp}' for vpp in results_df['VPP']])
    ax1.set_xlabel('VPP')
    ax1.set_ylabel('Bit Error Rate (%)')
    ax1_twin.set_ylabel('Invalid Transmissions')
    ax1.set_title('BER vs VPP with Invalid Transmissions')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis limits with extra space for error bars
    ymax = max(results_df['BER_No_BP'].max(), results_df['BER_With_BP'].max())
    ymax_with_error = ymax + max(results_df['BER_No_BP_Std'].max(), results_df['BER_BP_Std'].max())
    ax1.set_ylim(0, min(100, ymax_with_error * 1.2))  # Add 20% margin, cap at 100%
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Plot Average Power vs VPP with error bars
    ax2.errorbar(vpp_positions, results_df['Average_Power'],
                yerr=results_df['Power_Std'],
                fmt='go-', label='Average Power', markersize=8,
                capsize=5, capthick=1, elinewidth=1)
    
    # Add value labels for power
    for x, y in zip(vpp_positions, results_df['Average_Power']):
        ax2.annotate(f'{y:.1f}', (x, y), textcoords="offset points",
                    xytext=(0, 10), ha='center', color='green')
    
    # Configure power plot
    ax2.set_xticks(vpp_positions)
    ax2.set_xticklabels([f'{vpp}' for vpp in results_df['VPP']])
    ax2.set_xlabel('VPP')
    ax2.set_ylabel('Average Power')
    ax2.set_title('Average Power vs VPP, Distance: 100 cm, Bitrate: 500')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Set y-axis limits with extra space for error bars
    ymin, ymax = ax2.get_ylim()
    ax2.set_ylim(ymin, ymax * 1.2)  # Add 20% margin
    
    plt.tight_layout()
    plt.show()
    
    return results_df

def analyze_ber_by_bitrate_and_distance(file_path, only_bandpass=False, compare_hamming=False):
    """
    Create a single plot showing BER vs distance with different lines for each bitrate
    
    Args:
        file_path (str): Path to CSV file containing transmission data
        only_bandpass (bool): If True, only show results with bandpass filtering
        compare_hamming (bool): If True, compare with and without Hamming encoding
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
    
    # Define markers for different types
    markers = {
        'No Encoding': 'o',     
        'Hamming Encoding': 's',  
        'SG': 'o',
        'ESP': 's'
    }
    
    for idx, bitrate in enumerate(allowed_bitrates):
        bitrate_data = df[df['Bitrate'] == bitrate]
        
        # Process data based on comparison type
        if compare_hamming:
            conditions = [
                ('No Encoding', bitrate_data[bitrate_data['Encoding'] == 'No Encoding']),
                ('Hamming Encoding', bitrate_data[bitrate_data['Encoding'] == 'Hamming Encoding'])
            ]
        else:
            conditions = [
                ('SG', bitrate_data[bitrate_data['Transmitter'] == 'SG']),
                ('ESP', bitrate_data[bitrate_data['Transmitter'] == 'ESP'])
            ]
        
        for trans_type, trans_data in conditions:
            results = []
            
            for distance in distances:
                group = trans_data[trans_data['Distance to speaker'] == distance]
                if len(group) == 0:
                    continue
                
                total_transmissions = len(group)
                
                # Calculate BER with bandpass
                valid_bp = group[group['Decoded with bandpass'] != 'No preamble found']
                total_bits_bp = len(valid_bp) * 96  # Using fixed message length of 96 bits
                total_errors_bp = valid_bp['Hamming Dist with bandpass'].sum()
                ber_bp = (total_errors_bp / total_bits_bp * 100) if total_bits_bp > 0 else 100
                
                if not only_bandpass:
                    # Calculate BER without bandpass
                    valid_no_bp = group[group['Decoded without bandpass'] != 'No preamble found']
                    total_bits_no_bp = len(valid_no_bp) * 96
                    total_errors_no_bp = valid_no_bp['Hamming Dist without bandpass'].sum()
                    ber_no_bp = (total_errors_no_bp / total_bits_no_bp * 100) if total_bits_no_bp > 0 else 100
                
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
                                # Add label at 600 cm with offset
                
                mask_600cm = df_results['Distance'] == 600
                if any(mask_600cm):
                    ber_value = df_results.loc[mask_600cm, 'BER_BP'].values[0]
                    # Calculate vertical offset based on index
                    # cursed setup to fix the labels
                    if ber_value > 48.5 and ber_value < 51:
                        y_offset = 0
                    elif ber_value > 51:
                        y_offset = 4
                    else: 
                        y_offset = -10
                    # Add black border to text using path effects
                    text = ax.annotate(f'{ber_value:.1f}%', 
                                    xy=(600, ber_value),
                                    xytext=(10, y_offset),
                                    textcoords='offset points',
                                    color=colors[idx],
                                    fontsize=10)
                    # Add black edge
                    text.set_path_effects([
                        path_effects.Stroke(linewidth=1, foreground='grey'),
                        path_effects.Normal()
                    ])

                if not only_bandpass:
                    # Plot non-bandpass results
                    ax.plot(df_results['Distance'], df_results['BER_No_BP'], 
                           marker=markers[trans_type], linestyle='--', color=colors[idx],
                           label=f'{label_base} (No BP)', markersize=8)
    
    # Configure plot
    ax.set_xlabel('Distance (cm)')
    ax.set_ylabel('Bit Error Rate (%)')
    title = 'Bit Error Rate vs Distance'
    if compare_hamming:
        title += ' (Comparing Hamming Encoding, Signal Generator)'
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(0, 70)
    ax.set_xlim(50, 650)
    
    # Set x-axis ticks
    ax.set_xticks(distances)
    
    # Add legend with extra space on the right
    plt.subplots_adjust(right=0.75)
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
    # file_path = "ESP_new_average_power_1_3_6meters.csv"
    # file_path = "Average_power_of_received_signal.csv"
    # file_path = "avg_power_of_rec_signal_purely_for_check_of_interference.csv"

    # df = pd.read_csv(file_path)
    # print(df["Test description"].unique())
    # dist = 200
    # bitrate = 500
    # transmitter = "SG"
    # # "Testing: Average power purely for check of interference"
    # # "Testing: average power of a signal"  
    # # "Testing: average power of a signal - ESP"
    # # "Testing: average power of a signal - ESP, reverted back to old code"
    # results_df = analyze_ber_by_carrier_freq(file_path, dist, bitrate, transmitter, "Testing: average power of a signal", show_error_bars=True)

    # # NOTE: change to see a subset of carrier freqs, min_freq, max_freq as inputs
    # plot_power_vs_distance_by_frequency(file_path, 9000, 15000, "Testing: average power of a signal")
    
    # # NOTE: below is for the vpp test
    # test_file = "1m_distance_carrier_freq_sg_vpp_variable.csv"
    # result_df = compute_ber_for_different_vpps(test_file)

    # # NOTE: below computes BER for the max bitrate using ESP, set only_bandpass = True if you only want to compare bandpass, compare_hamming = True if you want to compare
    # # SG with and without hamming. Both cant be true at the same time :) 
    # file_path = "Max_bitrate_at_different_distances_and_best_carrier_freq.csv"
    # results = analyze_ber_by_bitrate_and_distance(file_path, only_bandpass=False, compare_hamming=False)

    # NOTE: below computes the BER for varying lengths of the message
    # file_path = "Varying_payload_sizes.csv"
    # varying_length_analysis_and_plot(file_path, only_bandpass=True)

    # NOTE: computing bitflip tendency for a given file, computes for ESP and SG
    file_path = "Average_power_of_received_signal.csv"
    results = analyze_bit_flips_by_transmitter(file_path)

    # NOTE: below computes random payload composition importance
    # TODO: Figure out how to do the note above :)

    ## NOTE: new function
    # plot_carrier_freq_analysis(results_df, "Testing: average power of a signal")
    # results = analyze_ber_by_carrier_freq(file_path, test_description="Testing: testing impact of similarlity of payloads and barker 13")
    # results = analyze_invalid_transmissions(file_path)



    
