import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects  
import pandas as pd
import numpy as np
import ast
from scipy.special import erfc


# --- Theoretical Model Constants and Functions ---
# (These are from your provided example block)

# Hydrophone & conversion constants
HYDRO_SENS = 40e-6    # V_rms per Pa (adjust if your value is different)
V_REF      = 1.0      # Full-scale RMS voltage for int16 (adjust if different)
RHO        = 1000     # kg/m^3 (density of water)
C          = 1500     # m/s (speed of sound in water)import scipy.signal as signal 

def compute_ber_for_hamming_encoding_test(file_path):
    """
    Compute BER and preamble detection statistics for Hamming vs No encoding with standard deviations
    
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
        
        # Arrays to store individual BER values for std calculation
        ber_array_no_bp = []
        ber_array_bp = []
        
        # Process no bandpass results
        for _, row in valid_no_bp.iterrows():
            errors = pd.to_numeric(row['Hamming Dist without bandpass'], errors='coerce')
            if pd.notna(errors):
                ber = errors / message_length
                ber_array_no_bp.append(ber)
                
        # Process bandpass results
        for _, row in valid_bp.iterrows():
            errors = pd.to_numeric(row['Hamming Dist with bandpass'], errors='coerce')
            if pd.notna(errors):
                ber = errors / message_length
                ber_array_bp.append(ber)
        
        # Calculate mean BER and standard deviation
        ber_no_bp_mean = np.mean(ber_array_no_bp) if ber_array_no_bp else 1.0
        ber_bp_mean = np.mean(ber_array_bp) if ber_array_bp else 1.0
        
        ber_no_bp_std = np.std(ber_array_no_bp) / np.sqrt(len(ber_array_no_bp)) if ber_array_no_bp else 0
        ber_bp_std = np.std(ber_array_bp) / np.sqrt(len(ber_array_bp)) if ber_array_bp else 0
        
        # Store results
        results[encoding] = {
            'total_transmissions': total_transmissions,
            'no_preamble_no_bp': no_preamble_no_bp,
            'no_preamble_bp': no_preamble_bp,
            'valid_transmissions_no_bp': len(valid_no_bp),
            'valid_transmissions_bp': len(valid_bp),
            'ber_no_bp_mean': ber_no_bp_mean,
            'ber_bp_mean': ber_bp_mean,
            'ber_no_bp_std': ber_no_bp_std,
            'ber_bp_std': ber_bp_std
        }
        
        print(f"\nResults for {encoding}:")
        print(f"Total transmissions: {total_transmissions}")
        print("\nWithout Bandpass:")
        print(f"No preamble found: {no_preamble_no_bp} ({no_preamble_no_bp/total_transmissions*100:.1f}%)")
        print(f"Valid transmissions: {len(valid_no_bp)}")
        print(f"BER: {ber_no_bp_mean:.3f} ± {ber_no_bp_std:.3f}")
        
        print("\nWith Bandpass:")
        print(f"No preamble found: {no_preamble_bp} ({no_preamble_bp/total_transmissions*100:.1f}%)")
        print(f"Valid transmissions: {len(valid_bp)}")
        print(f"BER: {ber_bp_mean:.3f} ± {ber_bp_std:.3f}")
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
    print(grouped)
    
    results = []
    message_length = 96  # Length of alternating bit sequence
    
    for carrier_freq, group in grouped:
        total_entries = len(group)
        
        # Without bandpass
        valid_no_bp = group[group['Decoded without bandpass'] != 'No preamble found']
        total_bits_no_bp = len(valid_no_bp) * message_length
        total_errors_no_bp = valid_no_bp['Hamming Dist without bandpass'].sum()
        invalid_no_bp_rate = len(group[group['Decoded without bandpass'] == 'No preamble found']) / len(group)
        
        # With bandpass
        valid_bp = group[group['Decoded with bandpass'] != 'No preamble found']
        total_bits_bp = len(valid_bp) * message_length
        total_errors_bp = valid_bp['Hamming Dist with bandpass'].sum()
        invalid_bp_rate = len(group[group['Decoded with bandpass'] == 'No preamble found']) / len(group)
        
        # Calculate BER as decimal (0-1 scale)
        ber_no_bp = (total_errors_no_bp / total_bits_no_bp) if total_bits_no_bp > 0 else None
        ber_bp = (total_errors_bp / total_bits_bp) if total_bits_bp > 0 else None
    



        # Calculate standard deviations
        if total_bits_no_bp > 0:
            errors_array_no_bp = valid_no_bp['Hamming Dist without bandpass'].values
            ber_array_no_bp = (errors_array_no_bp / message_length)
            ber_no_bp_std = np.std(ber_array_no_bp) / np.sqrt(len(errors_array_no_bp))
        else:
            ber_no_bp_std = None
            
        if total_bits_bp > 0:
            errors_array_bp = valid_bp['Hamming Dist with bandpass'].values
            ber_array_bp = (errors_array_bp / message_length)
            ber_bp_std = np.std(ber_array_bp) / np.sqrt(len(errors_array_bp))
        else:
            ber_bp_std = None

        # Calculate power statistics in dB relative to max power
        power_linear = group['Average Power of signal'].mean()
        max_power = df['Average Power of signal'].max()  # Get maximum power across all measurements
        
        # Convert to relative dB
        power_db_rel = 10 * np.log10(power_linear / max_power) if power_linear > 0 else -100
        power_std_linear = group['Average Power of signal'].std() / np.sqrt(len(group))
        # Convert standard deviation to relative dB scale
        power_std_db = (10 * np.log10((power_linear + power_std_linear) / max_power) - 
                        10 * np.log10(power_linear / max_power)) if power_linear > 0 else 0

        

        results.append({
            'Carrier_Frequency': carrier_freq,
            'Total_Transmissions': total_entries,
            'Invalid_No_BP': invalid_no_bp_rate,
            'Invalid_BP': invalid_bp_rate,
            'Valid_Transmissions_No_BP': len(valid_no_bp),
            'Valid_Transmissions_BP': len(valid_bp),
            'Total_Errors_No_BP': total_errors_no_bp,
            'Total_Errors_BP': total_errors_bp,
            'BER_No_BP': ber_no_bp,
            'BER_With_BP': ber_bp,
            'Average_Power': group['Average Power of signal'].mean(),
            'Average_Power_DB': power_db_rel,
            'Power_Std': group['Average Power of signal'].std(),
            'Power_Std_DB': power_std_db,
            'BER_No_BP_Std': ber_no_bp_std,
            'BER_BP_Std': ber_bp_std
        })
        
        # Print results with decimal percentages
        print(f"\nCarrier Frequency: {carrier_freq} Hz at {dist} cm")
        print(f"Test description: {test_description}")
        print(f"Total transmissions: {total_entries}")
        print(f"Average signal power: {group['Average Power of signal'].mean():.2f}")
        print("\nWithout bandpass:")
        print(f"  Invalid rate: {invalid_no_bp_rate:.2f}")
        print(f"  Valid transmissions: {len(valid_no_bp)}")
        print(f"  Total errors: {total_errors_no_bp}")
        print(f"  BER: {ber_no_bp:.2f}" if ber_no_bp is not None else "  BER: N/A")
        print("\nWith bandpass:")
        print(f"  Invalid rate: {invalid_bp_rate:.2f}")
        print(f"  Valid transmissions: {len(valid_bp)}")
        print(f"  Total errors: {total_errors_bp}")
        print(f"  BER: {ber_bp:.2f}" if ber_bp is not None else "  BER: N/A")
        print("-" * 50)
    
    # Create DataFrame and sort by frequency
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Carrier_Frequency')
    
    # Plot results
    plot_carrier_freq_analysis(results_df, dist, bitrate, test_description, show_error_bars)
    
    return results_df

def plot_carrier_freq_analysis(results_df, dist, bitrate, test_description, show_error_bars):
    """
    Create plots showing BER and signal power vs carrier frequency
    Using decimal rates (0-1) for BER and invalid transmissions
    
    Args:
        results_df (pd.DataFrame): DataFrame containing analysis results
        test_description (str): Description of the test
        show_error_bars (bool): Whether to show error bars in the plots
    """
    # Create complete range of frequencies
    all_frequencies = range(1000, 20000, 1000)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))
    
    # Ensure all frequencies are in the DataFrame
    complete_df = pd.DataFrame({'Carrier_Frequency': all_frequencies})
    results_df = pd.merge(complete_df, results_df, on='Carrier_Frequency', how='left')
    
    # Replace NaN with 1.0 for plotting (will appear at top of plot)
    plot_df = results_df.copy()
    
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
    bars_no_bp = ax1_twin.bar(plot_df['Carrier_Frequency'] - bar_width/2, plot_df['Invalid_No_BP'],
                width=bar_width, alpha=0.3, color='red', label='Invalid (No BP)')
    bars_bp = ax1_twin.bar(plot_df['Carrier_Frequency'] + bar_width/2, plot_df['Invalid_BP'],
                width=bar_width, alpha=0.3, color='blue', label='Invalid (BP)')
    
    # Add value labels for BER
    for x, y1, y2 in zip(plot_df['Carrier_Frequency'], 
                        plot_df['BER_No_BP'], 
                        plot_df['BER_With_BP']):
        if y1 == 1:  # N/A value
            ax1.annotate('N/A', (x, y1), textcoords="offset points",
                        xytext=(15, 5), ha='center', color='red')
        elif pd.notna(y1):
            ax1.annotate(f'{y1:.2f}', (x, y1), textcoords="offset points",
                        xytext=(15, 5), ha='center', color='red')
            
        if y2 == 1:  # N/A value
            ax1.annotate('N/A', (x, y2), textcoords="offset points",
                        xytext=(15, 5), ha='center', color='blue')
        elif pd.notna(y2):
            ax1.annotate(f'{y2:.2f}', (x, y2), textcoords="offset points",
                        xytext=(15, 5), ha='center', color='blue')
    
    # Add labels for invalid transmission rates with offset
    for bar in bars_no_bp:
        height = bar.get_height()
        # if height > 0:
        #     ax1_twin.text(bar.get_x() + bar.get_width()/2 + 0.2, height + 0.02,
        #                  f'{height:.2f}',
        #                  ha='center', va='bottom',
        #                  color='darkred',
        #                  fontweight='bold',
        #                  fontsize=10)
    
    for bar in bars_bp:
        height = bar.get_height()
        # if height > 0:
            # ax1_twin.text(bar.get_x() + bar.get_width()/2 - 0.2, height + 0.02,
            #              f'{height:.2f}',
            #              ha='center', va='bottom',
            #              color='darkblue',
            #              fontweight='bold',
            #              fontsize=10)

    # Configure BER plot
    ax1.set_xlabel('Carrier Frequency (Hz)')
    ax1.set_ylabel('Bit Error Rate')
    ax1_twin.set_ylabel('Invalid Transmission Rate')
    ax1.set_title(f'BER vs Carrier Frequency with Invalid Transmission Rate, Distance: {dist} cm, Bitrate: {bitrate} bps')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(-0.05, 1.1)  # Make room for labels
    ax1_twin.set_ylim(0, 1.1)  # Same scale as BER
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Set x-axis ticks
    ax1.set_xticks(list(all_frequencies))
    ax1.tick_params(axis='x', rotation=45)

    # Plot power data
    if show_error_bars:
        ax2.errorbar(results_df['Carrier_Frequency'], results_df['Average_Power_DB'],
                    yerr=results_df['Power_Std_DB'],
                    fmt='go-', label='Signal Power', markersize=8,
                    capsize=5, capthick=1, elinewidth=1)
    else:
        ax2.plot(results_df['Carrier_Frequency'], 10 * np.log10(results_df['Average_Power_DB']), 
                'go-', label='Signal Power', markersize=8)
    
    # Add value labels for power
    for x, y in zip(results_df['Carrier_Frequency'], results_df['Average_Power_DB']):
        if pd.notna(y):
            ax2.annotate(f'{y:.1f}', (x, y), textcoords="offset points",
                        xytext=(0, 10), ha='center', color='green')
        else:
            ax2.annotate('N/A', (x, 0), textcoords="offset points",
                        xytext=(0, 10), ha='center', color='green')

    


    # Configure power plot
    ax2.set_xlabel('Carrier Frequency (Hz)')
    ax2.set_ylabel('Average Signal Power (dB)')
    ax2.set_title(f'Signal Power vs Carrier Frequency, Distance: {dist} cm, Bitrate: {bitrate} bps')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    # Set y-axis limits for power plot
    ymin, ymax = ax2.get_ylim()
    ax2.set_ylim(ymin, 5)
    
    # Set x-axis ticks
    ax2.set_xticks(list(all_frequencies))
    ax2.tick_params(axis='x', rotation=45)

    # Adjust layout
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
    Reads a CSV file and plots Distance vs Average Power (in dB) for all carrier frequencies
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
    plt.figure(figsize=(10,6))

    # Increase font size for all text elements
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })

    # Find maximum power for relative dB calculation
    max_power = filtered_df['Average Power of signal'].max()

    for freq, color in zip(frequencies, colors):
        subset = freq_filtered_df[freq_filtered_df['Carrier Frequency'] == freq]
        
        # Group by distance and calculate statistics
        stats = subset.groupby('Distance to speaker').agg({
            'Average Power of signal': lambda x: {
                'mean_db': 10 * np.log10(np.mean(x) / max_power) if np.mean(x) > 0 else -100,
                'std_db': (10 * np.log10((np.mean(x) + np.std(x)) / max_power) - 
                          10 * np.log10(np.mean(x) / max_power)) if np.mean(x) > 0 else 0,
                'count': len(x)
            }
        }).reset_index()
        
        # Extract values from dictionaries
        stats['mean_db'] = stats['Average Power of signal'].apply(lambda x: x['mean_db'])
        stats['std_db'] = stats['Average Power of signal'].apply(lambda x: x['std_db'])
        stats['count'] = stats['Average Power of signal'].apply(lambda x: x['count'])
        
        stats_sorted = stats.sort_values('Distance to speaker')
        
        # Print intermediate values including count
        print(f"\nCarrier Frequency: {freq} Hz")
        print("Distance (cm) | Mean Power (dB) | Std Dev (dB) | Data Points")
        print("-" * 65)
        for _, row in stats_sorted.iterrows():
            print(f"{row['Distance to speaker']:11.0f} | {row['mean_db']:13.2f} | {row['std_db']:11.2f} | {row['count']:>11d}")

        # Plot averaged points with error bars
        plt.errorbar(
            stats_sorted['Distance to speaker'],
            stats_sorted['mean_db'],
            yerr=stats_sorted['std_db'],
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
    
    # Add 50 to x-ticks if not already present
    xticks = list(ax.get_xticks())
    if 50 not in xticks:
        xticks.append(50)
        xticks.sort()
        ax.set_xticks(xticks)
    ax.set_xlim(0, 650)

    # Set labels and title with increased font sizes
    plt.title('Average Power vs. Distance for Different Carrier Frequencies', 
              fontsize=16, pad=20)
    plt.xlabel('Distance to speaker (cm)', fontsize=14, labelpad=10)
    plt.ylabel('Average Signal Power (dB)', fontsize=14, labelpad=10)
    
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
    Power is shown in dB scale.
    
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
        invalid_no_bp_rate = len(group[group['Decoded without bandpass'] == 'No preamble found']) / len(group)
        
        if len(valid_no_bp) > 0:
            # Calculate individual transmission BERs for standard error
            ber_array_no_bp = []
            for _, row in valid_no_bp.iterrows():
                errors = pd.to_numeric(row['Hamming Dist without bandpass'], errors='coerce')
                if pd.notna(errors):
                    ber = errors / message_length
                    ber_array_no_bp.append(ber)
            
            ber_no_bp = np.mean(ber_array_no_bp)
            ber_no_bp_std = np.std(ber_array_no_bp) / np.sqrt(len(ber_array_no_bp))
        else:
            ber_no_bp = 1.0
            ber_no_bp_std = 0
        
        # Process with bandpass
        valid_bp = group[group['Decoded with bandpass'] != 'No preamble found']
        invalid_bp_rate = len(group[group['Decoded with bandpass'] == 'No preamble found']) / len(group)
        
        if len(valid_bp) > 0:
            # Calculate individual transmission BERs for standard error
            ber_array_bp = []
            for _, row in valid_bp.iterrows():
                errors = pd.to_numeric(row['Hamming Dist with bandpass'], errors='coerce')
                if pd.notna(errors):
                    ber = errors / message_length
                    ber_array_bp.append(ber)
            
            ber_bp = np.mean(ber_array_bp)
            ber_bp_std = np.std(ber_array_bp) / np.sqrt(len(ber_array_bp))
        else:
            ber_bp = 1.0
            ber_bp_std = 0
        
        # Calculate power statistics in dB relative to max power
        power_linear = group['Average Power of signal'].mean()
        max_power = df['Average Power of signal'].max()  # Get maximum power across all measurements
        
        # Convert to relative dB
        power_db_rel = 10 * np.log10(power_linear / max_power) if power_linear > 0 else -100
        power_std_linear = group['Average Power of signal'].std() / np.sqrt(len(group))
        # Convert standard deviation to relative dB scale
        power_std_db = (10 * np.log10((power_linear + power_std_linear) / max_power) - 
                        10 * np.log10(power_linear / max_power)) if power_linear > 0 else 0

        
        results.append({
            'VPP': vpp,
            'BER_No_BP': ber_no_bp,
            'BER_With_BP': ber_bp,
            'BER_No_BP_Std': ber_no_bp_std,
            'BER_BP_Std': ber_bp_std,
            'Invalid_No_BP': invalid_no_bp_rate,
            'Invalid_BP': invalid_bp_rate,
            'Valid_Transmissions_No_BP': len(valid_no_bp),
            'Valid_Transmissions_BP': len(valid_bp),
            'Total_Transmissions': total_transmissions,
            'Average_Power_dB': power_db_rel,
            'Power_Std_dB': power_std_db
        })
    
    # Create DataFrame and sort by VPP
    results_df = pd.DataFrame(results).sort_values('VPP')
    
    # Create evenly spaced positions for plotting
    vpp_positions = np.arange(len(results_df))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.5)
    
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
    bars_no_bp = ax1_twin.bar(vpp_positions - bar_width/2, results_df['Invalid_No_BP'], 
                             width=bar_width, alpha=0.3, color='red', label='Invalid (No BP)')
    bars_bp = ax1_twin.bar(vpp_positions + bar_width/2, results_df['Invalid_BP'], 
                          width=bar_width, alpha=0.3, color='blue', label='Invalid (BP)')
    
    # Add value labels for BER
    for x, y1, y2 in zip(vpp_positions, results_df['BER_No_BP'], results_df['BER_With_BP']):
        ax1.annotate(f'{y1:.2f}', (x, y1), textcoords="offset points",
                    xytext=(15, 8), ha='center', color='red')
        ax1.annotate(f'{y2:.2f}', (x, y2), textcoords="offset points",
                    xytext=(15, 10), ha='center', color='blue')
    
    # Add value labels for invalid transmissions
    for bar in bars_no_bp:
        height = bar.get_height()
        if height > 0:
            ax1_twin.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                         f'{height:.2f}',
                         ha='center', va='bottom',
                         color='darkred',
                         fontweight='bold',
                         fontsize=10)
    
    for bar in bars_bp:
        height = bar.get_height()
        if height > 0:
            ax1_twin.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                         f'{height:.2f}',
                         ha='center', va='bottom',
                         color='darkblue',
                         fontweight='bold',
                         fontsize=10)
    
    # Configure BER plot
    ax1.set_xticks(vpp_positions)
    ax1.set_xticklabels([f'{vpp}' for vpp in results_df['VPP']])
    ax1.set_xlabel('Vpp (V)', labelpad=-5)
    ax1.set_ylabel('Bit Error Rate (%)')
    ax1_twin.set_ylabel('Invalid Transmission Rate (%)')
    ax1.set_title('BER vs Vpp with Invalid Transmissions, Distance: 100 cm, Bitrate: 500 bps', fontsize = 16)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(0, 1.1)
    ax1_twin.set_ylim(0, 1.1)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Plot Average Power (dB) vs VPP with error bars
    ax2.errorbar(vpp_positions, results_df['Average_Power_dB'],
                yerr=results_df['Power_Std_dB'],
                fmt='go-', label='Average Power', markersize=8,
                capsize=5, capthick=1, elinewidth=1)
    
    # Add value labels for power in dB
    for x, y in zip(vpp_positions, results_df['Average_Power_dB']):
        ax2.annotate(f'{y:.1f} dB', (x, y), textcoords="offset points",
                    xytext=(0, 10), ha='center', color='green')
    
    # Configure power plot
    ax2.set_xticks(vpp_positions)
    ax2.set_xticklabels([f'{vpp}' for vpp in results_df['VPP']])
    ax2.set_xlabel('Vpp (V)', labelpad=-5)
    ax2.set_ylabel('Average Power (dB)')
    ax2.set_title('Average Power vs Vpp, Distance: 100 cm, Bitrate: 500 bps', fontsize = 16)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_ylim(-20,5)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return results_df

def analyze_ber_by_bitrate_and_distance(file_path, only_bandpass=False, compare_hamming=False, transmitter_select="BOTH", only_hamming=False):
    """
    Create a plot showing BER vs distance with different lines for each bitrate
    
    Args:
        file_path (str): Path to CSV file
        only_bandpass (bool): If True, only show results with bandpass filtering
        compare_hamming (bool): If True, compare with and without Hamming encoding
        transmitter_select (str): "ESP", "SG", or "BOTH" to filter transmitter type
        only_hamming (bool): If True, show only Hamming encoded results
    """
    df = pd.read_csv(file_path)
    if transmitter_select not in ["ESP", "SG", "BOTH"]:
        raise ValueError("transmitter_select must be 'ESP', 'SG', or 'BOTH'")
    
    message_lengths = {
        'No Encoding': 96,
        'Hamming Encoding': 132
    }
    
    allowed_bitrates = [300, 500, 1000, 1500, 2000]
    df = df[df['Bitrate'].isin(allowed_bitrates)]
    distances = sorted(df['Distance to speaker'].unique())

    print("\n=== Detailed Analysis Results ===")
    print("=" * 80)
    
    def print_stats_table(trans_type, bitrate, distance, group_data, msg_length):
        """Helper function to print detailed statistics"""
        total_trans = len(group_data)
        valid_trans = len(group_data[group_data['Decoded with bandpass'] != 'No preamble found'])
        invalid_rate = (total_trans - valid_trans) / total_trans if total_trans > 0 else 0
        
        ber_array = []
        for _, row in group_data[group_data['Decoded with bandpass'] != 'No preamble found'].iterrows():
            errors = pd.to_numeric(row['Hamming Dist with bandpass'], errors='coerce')
            if pd.notna(errors):
                ber = errors / msg_length
                ber_array.append(ber)
        
        mean_ber = np.mean(ber_array) if ber_array else 1.0
        ber_std = np.std(ber_array) / np.sqrt(len(ber_array)) if ber_array else 0
        
        print(f"\n{trans_type} at {bitrate}bps, Distance: {distance}cm")
        print(f"  Total transmissions: {total_trans}")
        print(f"  Valid transmissions: {valid_trans}")
        print(f"  Invalid rate: {invalid_rate:.3f}")
        print(f"  BER: {mean_ber:.3f} ± {ber_std:.3f}")
    
    # Create visualization
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    ax_top = ax.twiny()
    
    def get_carrier_freqs(data, transmitter):
        return data[data['Transmitter'] == transmitter].groupby('Distance to speaker')['Carrier Frequency'].first()
    
    if transmitter_select != "BOTH":
        carrier_freqs = get_carrier_freqs(df, transmitter_select)
    else:
        carrier_freqs_esp = get_carrier_freqs(df, "ESP")
        carrier_freqs_sg = get_carrier_freqs(df, "SG")
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(allowed_bitrates)))
    markers = {
        'No Encoding': 'o',
        'Hamming Encoding': 's',
        'SG': 'o',
        'ESP': 's'
    }

    # Process data for each bitrate
    for idx, bitrate in enumerate(allowed_bitrates):
        print(f"\nAnalyzing Bitrate: {bitrate} bps")
        print("-" * 40)
        
        bitrate_data = df[df['Bitrate'] == bitrate]
        
        if compare_hamming:
            conditions = [
                ('Hamming Encoding', bitrate_data[bitrate_data['Encoding'] == 'Hamming Encoding'])
            ] if only_hamming else [
                ('No Encoding', bitrate_data[bitrate_data['Encoding'] == 'No Encoding']),
                ('Hamming Encoding', bitrate_data[bitrate_data['Encoding'] == 'Hamming Encoding'])
            ]
        else:
            if transmitter_select == "BOTH":
                conditions = [
                    ('SG', bitrate_data[bitrate_data['Transmitter'] == 'SG']),
                    ('ESP', bitrate_data[bitrate_data['Transmitter'] == 'ESP'])
                ]
            else:
                conditions = [(transmitter_select, 
                             bitrate_data[bitrate_data['Transmitter'] == transmitter_select])]
        
        # Process each condition
        for trans_type, trans_data in conditions:
            msg_length = message_lengths['Hamming Encoding' if 'Hamming' in trans_type else 'No Encoding']
            results = []
            
            all_bers = []
            all_invalid_rates = []
            
            for distance in distances:
                group = trans_data[trans_data['Distance to speaker'] == distance]
                if len(group) == 0:
                    continue
                
                # Print detailed statistics for this group
                print_stats_table(trans_type, bitrate, distance, group, msg_length)
                
                valid_bp = group[group['Decoded with bandpass'] != 'No preamble found']
                invalid_rate = (len(group) - len(valid_bp)) / len(group)
                ber_array_bp = []
                
                for _, row in valid_bp.iterrows():
                    errors = pd.to_numeric(row['Hamming Dist with bandpass'], errors='coerce')
                    if pd.notna(errors):
                        ber = errors / msg_length
                        ber_array_bp.append(ber)
                        all_bers.append(ber)
                
                all_invalid_rates.append(invalid_rate)
                
                ber_bp = np.mean(ber_array_bp) if ber_array_bp else 1.0
                ber_bp_std = np.std(ber_array_bp) / np.sqrt(len(ber_array_bp)) if ber_array_bp else 0
                
                results.append({
                    'Distance': distance,
                    'BER': ber_bp,
                    'BER_std': ber_bp_std
                })
            
            # Print summary statistics for this condition
            if all_bers:
                mean_ber = np.mean(all_bers)
                ber_std = np.std(all_bers) / np.sqrt(len(all_bers))
                mean_invalid = np.mean(all_invalid_rates)
                print(f"\nSummary for {trans_type} at {bitrate}bps:")
                print(f"Overall BER: {mean_ber:.3f} ± {ber_std:.3f}")
                print(f"Overall Invalid Rate: {mean_invalid:.3f}")
                print("-" * 40)
            
            if results:
                df_results = pd.DataFrame(results)
                label_base = f'{trans_type} {bitrate}bps'
                
                # Filter points where BER < 0.1 (10%)
                mask = df_results['BER'] < 0.1
                df_results = df_results[mask]

                ax.errorbar(df_results['Distance'], df_results['BER'],
                          yerr=df_results['BER_std'],
                          marker=markers[trans_type], linestyle='-', color=colors[idx],
                          label=f'{label_base}', markersize=8,
                          capsize=5, capthick=1, elinewidth=1)
                
                # Add value annotations for 600cm point
                mask_600cm = df_results['Distance'] == 600
                if any(mask_600cm):
                    ber_value = df_results.loc[mask_600cm, 'BER'].values[0]
                    y_offset = -5 if bitrate == 2000 else (5 if bitrate == 1500 else 0)
                    if bitrate == 1000 and msg_length == 132:
                        y_offset = -10
                    if bitrate == 1500 and msg_length == 96:
                        y_offset = -10
                    if bitrate == 2000 and msg_length == 96:
                        y_offset = 5

                    ax.annotate(f'{ber_value:.2f}',
                              xy=(600, ber_value),
                              xytext=(7, y_offset),
                              textcoords='offset points',
                              color=colors[idx],
                              fontsize=12)
    
    # Configure plot
    ax.set_xlim(30, 690)
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(distances)
    
    if transmitter_select != "BOTH":
        ax_top.set_xticklabels([f'{carrier_freqs[d]/1000:.0f}kHz' for d in distances], 
                              rotation=45, fontsize=13)
    else:
        ax_top.set_xticklabels([f'ESP:{carrier_freqs_esp[d]/1000:.0f}kHz SG:{carrier_freqs_sg[d]/1000:.0f}kHz' 
                               for d in distances], rotation=45, fontsize=13)
    
    ax_top.set_xlabel('Carrier Frequency', fontsize=14, labelpad=5)
    ax.set_xlabel('Distance (cm)', fontsize=14, labelpad=5)
    ax.set_ylabel('Bit Error Rate', fontsize=14, labelpad=5)
    title = f'Bit Error Rate vs Distance - {transmitter_select}'
    if compare_hamming:
        title += ' with Hamming Encoding'
    ax.set_title(title, fontsize=16, pad=20)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(0, 0.15)
    ax.set_xticks(distances)
    ax.tick_params(axis='x', rotation=45, labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax_top.tick_params(axis='x', labelsize=13)
    
    # Change legend placement to inside plot
    plt.subplots_adjust(right=0.95, top=0.90)  # Adjust plot margins
    ax.legend(bbox_to_anchor=(0.98, 0.98),  # Position legend in top right corner
             loc='upper right',
             fontsize=11,
             ncol=1)
    
    
    plt.tight_layout()
    plt.show()

def varying_length_analysis_and_plot(file_path, test_description=None, only_bandpass=False):
    """
    Compute and plot average BER vs message length using Hamming distance from the CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing transmission data
        test_description (str): Optional test description to filter data
        only_bandpass (bool): If True, only show results with bandpass filtering
    """
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # # Inside varying_length_analysis_and_plot function, modify the ID saving part:
    # # Save IDs to text file with quotes
    # with open('message_ids.txt', 'w') as f:
    #     ids = [f'"{str(id)}"' for id in df["ID"]]  # Wrap each ID in quotes
    #     f.write(','.join(ids))  # Join IDs with commas and write to file
    
    # Filter by test description if provided
    if test_description:
        df = df[df['Test description'] == test_description]
        if df.empty:
            print(f"No data found for test description: {test_description}")
            return
    
    # Extract message length
    df['Message Length'] = df['Original message in bits'].apply(lambda x: len(ast.literal_eval(x)) if pd.notna(x) else None)
    # Filter out message length of 976 bits
    # df = df[df['Message Length'] != 976]
    df = df[df['Message Length'] < 600]
    
    # Calculate BER using Hamming distance
    df['BER_BP'] = df['Hamming Dist with bandpass'] / df['Message Length']
    df['BER_No_BP'] = df['Hamming Dist without bandpass'] / df['Message Length']
    df['Invalid_BP'] = df['Decoded with bandpass'].apply(lambda x: 1 if x == 'No preamble found' else 0)
    df['Invalid_No_BP'] = df['Decoded without bandpass'].apply(lambda x: 1 if x == 'No preamble found' else 0)
    
    # Group by message length and calculate statistics
    df_grouped = df.groupby('Message Length').agg({
        'BER_BP': ['mean', 'std'],
        'Invalid_BP': ['sum', 'size'],
        'Hamming Dist with bandpass': 'sum'
    })
    
    # Calculate invalid transmission percentages
    df_grouped[('Invalid_BP', 'percentage')] = (
        df_grouped[('Invalid_BP', 'sum')] / df_grouped[('Invalid_BP', 'size')]
    )
    
    if not only_bandpass:
        no_bp_stats = df.groupby('Message Length').agg({
            'BER_No_BP': ['mean', 'std'],
            'Invalid_No_BP': ['sum', 'size'],
            'Hamming Dist without bandpass': 'sum'
        })
        no_bp_stats[('Invalid_No_BP', 'percentage')] = (
            no_bp_stats[('Invalid_No_BP', 'sum')] / no_bp_stats[('Invalid_No_BP', 'size')]
        )
        df_grouped = pd.concat([df_grouped, no_bp_stats], axis=1)
    
    # Reset index to make Message Length a column
    stats = df_grouped.reset_index()

    # Set global font size
    plt.rcParams.update({'font.size': 12})
    
    # Create plot
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax_twin = ax.twinx()
    
    x_values = stats['Message Length'].values
    bar_width = 0.4
    
    # Plot bandpass results
    ax.errorbar(x_values, stats[('BER_BP', 'mean')], 
                yerr=stats[('BER_BP', 'std')],
                fmt='bo-', label='With bandpass', markersize=8,
                capsize=5, capthick=1, elinewidth=1)
    
    # Add labels for bandpass BER
    for x, y in zip(x_values, stats[('BER_BP', 'mean')]):
        if pd.notna(y):
            ax.annotate(f'{y:.2f}', 
                       (x, y),
                       xytext=(15, 10), 
                       textcoords='offset points',
                       ha='center',
                       va='bottom',
                       color='blue')
    
    # Plot invalid transmission percentage bars
    bars_bp = ax_twin.bar(x_values - bar_width/2 if not only_bandpass else x_values, 
                         stats[('Invalid_BP', 'percentage')], 
                         width=bar_width*1.5,
                         alpha=1,
                         color='red', 
                         label='Invalid % (BP)',
                         edgecolor='red',
                         linewidth=2)
    
    # Inside varying_length_analysis_and_plot function, after creating the bars:
    # Add labels for invalid transmission rates
    for bar in bars_bp:
        height = bar.get_height()
        if height > 0:
            ax_twin.text(bar.get_x() + bar.get_width()/2 + 20, height + 0.01,
                        f'{height:.2f}',
                        ha='center', va='bottom',
                        color='red',
                        fontweight='bold',
                        fontsize=10)

    if not only_bandpass:
        for bar in bars_no_bp:
            height = bar.get_height()
            if height > 0:
                ax_twin.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                            f'{height:.2f}',
                            ha='center', va='bottom',
                            color='darkblue',
                            fontweight='bold',
                            fontsize=10)

    if not only_bandpass:
        # Plot non-bandpass results
        ax.errorbar(x_values, stats[('BER_No_BP', 'mean')], 
                   yerr=stats[('BER_No_BP', 'std')],
                   fmt='ro-', label='Without bandpass', markersize=8,
                   capsize=5, capthick=1, elinewidth=1)
        
        for x, y in zip(x_values, stats[('BER_No_BP', 'mean')]):
            if pd.notna(y):
                ax.annotate(f'{y:.2f}', 
                           (x, y),
                           xytext=(0, -20), 
                           textcoords='offset points',
                           ha='center',
                           va='top',
                           color='red')
        
        bars_no_bp = ax_twin.bar(x_values + bar_width/2, 
                                stats[('Invalid_No_BP', 'percentage')], 
                                width=bar_width*1.5,
                                alpha=0.7,
                                color='red', 
                                label='Invalid % (No BP)',
                                edgecolor='darkred',
                                linewidth=2)
    

    # Print statistics
    print("\n=== Detailed Statistics for Each Message Length ===")
    for _, row in stats.iterrows():
        print(f"\nMessage Length: {row['Message Length']} bits")
        print("With Bandpass:")
        print(f"BER: {row[('BER_BP', 'mean')]:.3f} ± {row[('BER_BP', 'std')]:.3f}")
        print(f"Invalid Rate: {row[('Invalid_BP', 'percentage')]:.3f}")
        
        if not only_bandpass:
            print("\nWithout Bandpass:")
            print(f"BER: {row[('BER_No_BP', 'mean')]:.3f} ± {row[('BER_No_BP', 'std')]:.3f}")
            print(f"Invalid Rate: {row[('Invalid_No_BP', 'percentage')]:.3f}")
        print("-" * 50)
    
    # Print overall statistics
    print("\n=== Overall Statistics ===")
    print("With Bandpass:")
    mean_ber_bp = stats[('BER_BP', 'mean')].mean()
    std_ber_bp = np.sqrt(np.sum(stats[('BER_BP', 'std')]**2))/len(stats)
    mean_invalid_bp = stats[('Invalid_BP', 'percentage')].mean()
    print(f"Average BER across all lengths: {mean_ber_bp:.3f} ± {std_ber_bp:.3f}")
    print(f"Average Invalid Rate: {mean_invalid_bp:.3f}")
    
    print("\n=== Mean of Standard Deviations ===")
    mean_std_bp = stats[('BER_BP', 'std')].mean()
    print(f"Mean std with bandpass: {mean_std_bp:.3f}")

    if not only_bandpass:
        mean_std_no_bp = stats[('BER_No_BP', 'std')].mean()
        print(f"Mean std without bandpass: {mean_std_no_bp:.3f}")
        
        print("\nWithout Bandpass:")
        mean_ber_no_bp = stats[('BER_No_BP', 'mean')].mean()
        std_ber_no_bp = np.sqrt(np.sum(stats[('BER_No_BP', 'std')]**2))/len(stats)
        mean_invalid_no_bp = stats[('Invalid_No_BP', 'percentage')].mean()
        print(f"Average BER across all lengths: {mean_ber_no_bp:.3f} ± {std_ber_no_bp:.3f}")
        print(f"Average Invalid Rate: {mean_invalid_no_bp:.3f}")
    print("=" * 50)

    # Configure plot
    ax.set_xlabel('Message Length (bits)')
    ax.set_ylabel('Bit Error Rate (%)')
    ax_twin.set_ylabel('Invalid Transmission Rate (%)')
    title = 'Average Bit Error Rate vs Message Length, Distance: 100 cm, Bitrate: 500 bps'
    # if test_description:
    #     title += f'\n{test_description}'
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(bottom=0, top=0.7)
    ax_twin.set_ylim(0, 1.1)

    # Set x-axis ticks
    ax.set_xticks(x_values)
    ax.set_xticklabels([f'{int(x)}' for x in x_values])

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.subplots_adjust(top=0.85 if test_description else 0.9, 
                       bottom=0.15, 
                       left=0.1, 
                       right=0.9)
    plt.show()
    
def analyze_bit_flips_by_transmitter(csv_paths):
    """
    Analyze bit flip tendencies and invalid rates for ESP vs SG transmitters across multiple files
    
    Args:
        csv_paths (list): List of paths to CSV files containing transmission data
    """
    # Initialize per-file and total statistics
    results = {
        'ESP': {
            'bandpass': {
                'one_to_zero': 0,
                'zero_to_one': 0,
                'total_bits': 0,
                'valid_transmissions': 0,
                'total_transmissions': 0,
                'invalid_transmissions': 0,
                'per_file_flip_rates': [],  # Store rates for each file
                'per_file_invalid_rates': []
            },
            'no_bandpass': {
                'one_to_zero': 0,
                'zero_to_one': 0,
                'total_bits': 0,
                'valid_transmissions': 0,
                'total_transmissions': 0,
                'invalid_transmissions': 0,
                'per_file_flip_rates': [],
                'per_file_invalid_rates': []
            }
        },
        'SG': {
            'bandpass': {
                'one_to_zero': 0,
                'zero_to_one': 0,
                'total_bits': 0,
                'valid_transmissions': 0,
                'total_transmissions': 0,
                'invalid_transmissions': 0,
                'per_file_flip_rates': [],
                'per_file_invalid_rates': []
            },
            'no_bandpass': {
                'one_to_zero': 0,
                'zero_to_one': 0,
                'total_bits': 0,
                'valid_transmissions': 0,
                'total_transmissions': 0,
                'invalid_transmissions': 0,
                'per_file_flip_rates': [],
                'per_file_invalid_rates': []
            }
        }
    }
    
    def safe_eval(x):
        try:
            return ast.literal_eval(x) if pd.notna(x) else None
        except (ValueError, SyntaxError):
            return None
    
    # Process each file
    for file_path in csv_paths:
        print(f"\nProcessing file: {file_path}")
        file_stats = {
            'ESP': {'bandpass': {'flips': 0, 'bits': 0, 'invalid': 0, 'total': 0},
                   'no_bandpass': {'flips': 0, 'bits': 0, 'invalid': 0, 'total': 0}},
            'SG': {'bandpass': {'flips': 0, 'bits': 0, 'invalid': 0, 'total': 0},
                   'no_bandpass': {'flips': 0, 'bits': 0, 'invalid': 0, 'total': 0}}
        }
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
            
        for _, row in df.iterrows():
            transmitter = row['Transmitter']
            if transmitter not in results:
                print(f"Warning: Unknown transmitter type '{transmitter}' found in {file_path}")
                continue
                
            original = safe_eval(row['Original message in bits'])
            decoded_no_bp = safe_eval(row['Data bits without bandpass'])
            decoded_bp = safe_eval(row['Data bits with bandpass'])
            
            if original is None:
                continue
                
            # Process both bandpass and no_bandpass
            for filter_type in ['bandpass', 'no_bandpass']:
                decoded = decoded_bp if filter_type == 'bandpass' else decoded_no_bp
                decoded_field = 'Decoded with bandpass' if filter_type == 'bandpass' else 'Decoded without bandpass'
                
                # Update total transmissions
                results[transmitter][filter_type]['total_transmissions'] += 1
                file_stats[transmitter][filter_type]['total'] += 1
                
                if row[decoded_field] == 'No preamble found':
                    results[transmitter][filter_type]['invalid_transmissions'] += 1
                    file_stats[transmitter][filter_type]['invalid'] += 1
                elif decoded is not None and len(decoded) > 0:
                    results[transmitter][filter_type]['valid_transmissions'] += 1
                    
                    for dec in decoded:
                        if len(dec) == len(original):
                            results[transmitter][filter_type]['total_bits'] += len(original)
                            file_stats[transmitter][filter_type]['bits'] += len(original)
                            
                            for orig_bit, dec_bit in zip(original, dec):
                                if orig_bit != dec_bit:
                                    if orig_bit == 1:
                                        results[transmitter][filter_type]['one_to_zero'] += 1
                                        file_stats[transmitter][filter_type]['flips'] += 1
                                    else:
                                        results[transmitter][filter_type]['zero_to_one'] += 1
                                        file_stats[transmitter][filter_type]['flips'] += 1
        
        # Calculate and store per-file rates
        for transmitter in ['ESP', 'SG']:
            for filter_type in ['bandpass', 'no_bandpass']:
                stats = file_stats[transmitter][filter_type]
                if stats['total'] > 0:
                    invalid_rate = stats['invalid'] / stats['total']
                    results[transmitter][filter_type]['per_file_invalid_rates'].append(invalid_rate)
                
                if stats['bits'] > 0:
                    flip_rate = stats['flips'] / stats['bits']
                    results[transmitter][filter_type]['per_file_flip_rates'].append(flip_rate)
    
    # Print combined results with standard deviations
    print("\n=== Combined Results Across All Files ===")
    for transmitter in ['ESP', 'SG']:
        print(f"\n=== Results for {transmitter} ===")
        
        for filter_type in ['bandpass', 'no_bandpass']:
            data = results[transmitter][filter_type]
            total_trans = data['total_transmissions']
            valid_trans = data['valid_transmissions']
            total_bits = data['total_bits']
            
            print(f"\n-- {filter_type.replace('_', ' ').title()} Results --")
            print(f"Total transmissions: {total_trans}")
            print(f"Valid transmissions: {valid_trans}")
            
            if total_trans > 0:
                invalid_rate = data['invalid_transmissions']/total_trans * 100
                invalid_std = np.std(data['per_file_invalid_rates'])*100 if data['per_file_invalid_rates'] else 0
                print(f"Invalid rate: {invalid_rate:.2f}% ± {invalid_std:.2f}%")
                
                if valid_trans > 0:
                    print(f"Total bits processed: {total_bits}")
                    flip_rate = (data['one_to_zero'] + data['zero_to_one'])/total_bits * 100
                    flip_std = np.std(data['per_file_flip_rates'])*100 if data['per_file_flip_rates'] else 0
                    
                    print(f"1->0 flips: {data['one_to_zero']} ({data['one_to_zero']/total_bits*100:.2f}%)")
                    print(f"0->1 flips: {data['zero_to_one']} ({data['zero_to_one']/total_bits*100:.2f}%)")
                    print(f"Total bit flips: {data['one_to_zero'] + data['zero_to_one']} ({flip_rate:.2f}% ± {flip_std:.2f}%)")
                else:
                    print("No valid transmissions to analyze bit flips")
            else:
                print("No transmissions found")
        print("=" * 50)

    return results

def varying_correlation_value_and_plot(file_path, test_description, only_bandpass=False):
    """
    Analyze BER and invalid transmission rates for messages with different Barker code correlations.
    
    Args:
        file_path (str): Path to CSV file
        test_description (str): Test description to filter data
        only_bandpass (bool): If True, only show results with bandpass filtering
    """
    # Read and filter data
    df = pd.read_csv(file_path)
    df = df[df['Test description'] == test_description]
    
    # Calculate correlations
    BINARY_BARKER = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]
    
    # Group messages by correlation value
    correlation_groups = {}
    for message, group in df.groupby('Original Message'):
        original_bits = ast.literal_eval(group['Original message in bits'].iloc[0])
        correlation = round(np.max(signal.correlate(original_bits, BINARY_BARKER, mode="valid")), 2)
        
        if correlation not in correlation_groups:
            correlation_groups[correlation] = []
        correlation_groups[correlation].append((message, group))
    
    # Process each correlation group
    results = []
    for correlation, message_groups in correlation_groups.items():
        total_transmissions = 0
        ber_array_bp = []
        ber_array_no_bp = []
        invalid_bp_array = []
        invalid_no_bp_array = []
        
        # Process all messages with this correlation value
        for message, group in message_groups:
            original_bits = ast.literal_eval(group['Original message in bits'].iloc[0])
            message_length = len(original_bits)
            group_transmissions = len(group)
            total_transmissions += group_transmissions
            
            # Calculate invalid rate for this message
            invalid_bp = len(group[group['Decoded with bandpass'] == 'No preamble found']) / group_transmissions
            invalid_bp_array.append(invalid_bp)
            
            # Process bandpass results
            valid_bp = group[group['Decoded with bandpass'] != 'No preamble found']
            if len(valid_bp) > 0:
                for _, row in valid_bp.iterrows():
                    errors = pd.to_numeric(row['Hamming Dist with bandpass'], errors='coerce')
                    if pd.notna(errors):
                        ber = errors / message_length
                        ber_array_bp.append(ber)
            
            # Process non-bandpass results if needed
            if not only_bandpass:
                invalid_no_bp = len(group[group['Decoded without bandpass'] == 'No preamble found']) / group_transmissions
                invalid_no_bp_array.append(invalid_no_bp)
                
                valid_no_bp = group[group['Decoded without bandpass'] != 'No preamble found']
                if len(valid_no_bp) > 0:
                    for _, row in valid_no_bp.iterrows():
                        errors = pd.to_numeric(row['Hamming Dist without bandpass'], errors='coerce')
                        if pd.notna(errors):
                            ber = errors / message_length
                            ber_array_no_bp.append(ber)
        
        # Calculate statistics
        ber_bp = np.mean(ber_array_bp) if ber_array_bp else 1.0
        ber_bp_std = np.std(ber_array_bp) / np.sqrt(len(ber_array_bp)) if ber_array_bp else 0
        invalid_bp_mean = np.mean(invalid_bp_array)
        invalid_bp_std = np.std(invalid_bp_array) / np.sqrt(len(invalid_bp_array))
        
        # Print detailed statistics
        print(f"\nCorrelation value: {correlation:.2f}")
        print(f"Number of unique messages: {len(message_groups)}")
        print(f"Total transmissions: {total_transmissions}")
        print("\nWith Bandpass:")
        print(f"BER: {ber_bp:.3f} ± {ber_bp_std:.3f}")
        print(f"Invalid rate: {invalid_bp_mean:.3f} ± {invalid_bp_std:.3f}")
        
        result = {
            'Correlation': correlation,
            'BER_With_BP': ber_bp,
            'BER_BP_Std': ber_bp_std,
            'Invalid_BP': invalid_bp_mean,
            'Invalid_BP_Std': invalid_bp_std,
            'Message_Count': len(message_groups)
        }
        
        if not only_bandpass:
            ber_no_bp = np.mean(ber_array_no_bp) if ber_array_no_bp else 1.0
            ber_no_bp_std = np.std(ber_array_no_bp) / np.sqrt(len(ber_array_no_bp)) if ber_array_no_bp else 0
            invalid_no_bp_mean = np.mean(invalid_no_bp_array)
            invalid_no_bp_std = np.std(invalid_no_bp_array) / np.sqrt(len(invalid_no_bp_array))
            
            print("\nWithout Bandpass:")
            print(f"BER: {ber_no_bp:.3f} ± {ber_no_bp_std:.3f}")
            print(f"Invalid rate: {invalid_no_bp_mean:.3f} ± {invalid_no_bp_std:.3f}")
            
            result.update({
                'BER_No_BP': ber_no_bp,
                'BER_No_BP_Std': ber_no_bp_std,
                'Invalid_No_BP': invalid_no_bp_mean,
                'Invalid_No_BP_Std': invalid_no_bp_std
            })
        
        print("-" * 50)
        results.append(result)
    
    # Create DataFrame and sort by correlation
    results_df = pd.DataFrame(results).sort_values('Correlation')
    
    # Create plot
    plt.figure(figsize=(8, 4))
    x = np.arange(len(results_df))
    width = 0.15
    
    ax = plt.gca()
    ax_twin = ax.twinx()
    
    # Plot BER points with error bars
    if not only_bandpass:
        ax.errorbar(x - width/2, results_df['BER_No_BP'],
                   yerr=results_df['BER_No_BP_Std'],
                   fmt='ro', label='BER No BP', markersize=8,
                   capsize=5, capthick=1, elinewidth=1)
        invalid_bars_no_bp = ax_twin.bar(x - width/2, results_df['Invalid_No_BP'],
                   width=width, alpha=0.5, color='red',
                   label='Invalid No BP')
    
    # Plot bandpass results
    ax.errorbar(x + (0 if only_bandpass else width/2), results_df['BER_With_BP'],
                yerr=results_df['BER_BP_Std'],
                fmt='bo', label='BER With BP', markersize=8,
                capsize=5, capthick=1, elinewidth=1)
    
    invalid_bars_bp = ax_twin.bar(x + (0 if only_bandpass else width/2), results_df['Invalid_BP'],
                width=width, alpha=0.5, color='red',
                label='Invalid With BP')
    
    # Configure plot
    ax.set_ylabel('Bit Error Rate', fontsize=12)
    ax_twin.set_ylabel('Invalid Transmission Rate', fontsize=12)
    title = 'BER and Invalid Rate by Unique Correlation Values'
    if only_bandpass:
        title += ' (Bandpass Only)'
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{corr:.2f}" for corr in results_df['Correlation']],
                       rotation=45, ha='right')
    ax.set_xlabel('Correlation with Barker Code', fontsize=12)
    
    # Set y-axis limits
    ax.set_ylim(0, 1.1)
    ax_twin.set_ylim(0, 1.1)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return results_df


if __name__ == "__main__":  
    # below is the three functions to call for figure 24, 25, 32, and 33.

    # This is for figure 24, using the commented filepath
    # # NOTE: change to see a subset of carrier freqs, min_freq, max_freq as inputs
    # file_path = "Average_power_of_received_signal.csv"
    # plot_power_vs_distance_by_frequency(file_path, 9000, 15000, "Testing: average power of a signal")

    
    ## This is for figure 25, where the varibles is to defined the different bitrates and distances. Keep the bitrate constant, but change the dist 
    ## depending on the distance you want. There is most datat for the signal generator, and this is also the only one plotted in the report
    # dist = 500
    # bitrate = 500
    # transmitter = "SG"
    # "Testing: average power of a signal" 
    # file_path = "Average_power_of_received_signal.csv" 
    # results_df = analyze_ber_by_carrier_freq(file_path, dist, bitrate, transmitter, "Testing: average power of a signal", show_error_bars=True)

    
    # This is for figure 32 and 33. Set the variables for what you want to see. To reduce cluttering and also what is shown in the thesis 
    # # NOTE: below computes BER for the max bitrate using ESP, set only_bandpass = True if you only want to compare bandpass, compare_hamming = True if you want to compare
    # # SG with and without hamming. Both cant be true at the same time :) 
    # file_path = "Max_bitrate_at_different_distances_and_best_carrier_freq.csv"
    # results = analyze_ber_by_bitrate_and_distance(file_path, only_bandpass=True, compare_hamming=True, transmitter_select="SG", only_hamming=True)



    # no functions to use below here :)










    # file_path = "Code/dsp/data/plastic/SG_plastic_hamming_encoding_testing_cf_6000_400bps, 5sd, 50ds.csv"
    # compute_ber_for_hamming_encoding_test(file_path)




    # # NOTE: below is for the vpp test
    # test_file = "1m_distance_carrier_freq_sg_vpp_variable.csv"
    # result_df = compute_ber_for_different_vpps(test_file)



    # # NOTE: below computes the BER for varying lengths of the message
    # test_descrip = None
    # # file_path = "Varying_payload_sizes.csv"
    # # file_path = "Random_payloads.csv"
    # file_path = "Random_payloads_CORRECT_BARKER.csv"
    # test_descrip = "Testing with correct barker13 implementaion"
    # varying_length_analysis_and_plot(file_path, test_description = test_descrip, only_bandpass=True)

    # # NOTE: computing bitflip tendency for a given file, computes for ESP and SG
    # file_paths = ["1m_distance_payload_barker_similarity_impact.csv",
    #               "1m_distance_carrier_freq_sg_vpp_variable.csv", 
    #               "Max_bitrate_at_different_distances_and_best_carrier_freq.csv",
    #               "Average_power_of_received_signal.csv", 
    #               "5m_dist_10kHz_unique_payloads.csv",
    #               "Random_payloads.csv",
    #               "Varying_payload_sizes.csv",
    #               "avg_power_of_rec_signal_purely_for_check_of_interference.csv", 
    #               "Received_data_for_tests.csv"
    #               ]
    # results = analyze_bit_flips_by_transmitter(file_paths)

    # NOTE: below computes random payload composition importance
    # TODO: Figure out how to do the note above :)

    # # # NOTE: below compute PPC test
    # test_description = "Testing: payload similarity with barker 13 on 1m distance at 500 bit rate and 11000Hz carrier frequency"
    # file_path = "1m_distance_payload_barker_similarity_impact.csv"
    # varying_correlation_value_and_plot(file_path, test_description, only_bandpass=True)

    