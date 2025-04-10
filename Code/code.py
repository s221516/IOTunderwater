import csv

# Open the existing CSV file for reading and writing
input_file = 'data.csv'

# Read the data from the CSV
with open(input_file, mode='r', newline='') as file:
    reader = csv.reader(file)
    header = next(reader)  # Read the header row
    rows = [row for row in reader]  # Read all data rows

# Add the new header column for 'ActualBitrate'
header.append("ActualBitrate")

# Process each row to calculate the actual bitrate
for row in rows:
    try:
        # Extract values from the row
        duration_per_bit = float(row[0])  # Duration per bit (in microseconds)

        # Calculate the actual bit rate (1 / duration_per_bit)
        actual_bitrate = 1 / (duration_per_bit * 10**-6)  # Convert microseconds to seconds

        # Add the calculated actual bitrate to the row
        row.append(actual_bitrate)
    
    except ValueError:
        # Handle the case where conversion fails (skip invalid rows)
        print(f"Skipping invalid row: {row}")

# Write the updated data back to the same CSV file
with open(input_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)  # Write header row
    writer.writerows(rows)   # Write data rows

print(f"Updated data with actual bitrate added to {input_file}")
