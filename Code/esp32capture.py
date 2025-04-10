import serial
import csv
import time

# Set up serial connection
ser = serial.Serial('COM5', 115200)
time.sleep(2)  # Wait for the serial connection to initialize

# Open CSV file for writing
with open('data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["SampleRate", "BitRate", "Duration", "SamplesPerSymbol"])  # Header

    # Continuously read serial data
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            data = line.split(',')

            # Check if the line contains exactly 3 numbers
            if len(data) == 3:
                try:
                    sample_rate = float(data[0])
                    bit_rate = float(data[1])
                    duration = float(data[2])

                    samples_per_symbol = sample_rate / bit_rate
                    writer.writerow([sample_rate, bit_rate, duration, samples_per_symbol])
                    file.flush()  # 💥 Force data to be written to disk immediately
                    print(f"Data written: {sample_rate}, {bit_rate}, {duration}, {samples_per_symbol}")
                except ValueError:
                    print(f"Ignored (not numeric): {line}")
            else:
                print(f"Ignored (bad format): {line}")
