import serial
import csv
import time
import re

# Set up serial connection
ser = serial.Serial('COM5', 115200)
time.sleep(2)  # Allow time for the serial port to settle

# Prepare CSV file
with open('data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "SampleRate_Before", "SampleRate_After", "SamplesPerSymbol",
        "CarrierFrequency", "BitRate", "MessageLength", "Message", "MessageBits"
    ])

    # Buffer to collect multi-line message
    buffer = []

    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            print(f"Serial: {line}")  # For debug

            buffer.append(line)

            # Detect end of message block (based on last line being Message bits)
            if line.startswith("Message bits:"):
                try:
                    # Extract values from the buffer
                    data = {}

                    for l in buffer:
                        if "Sample rate before allocations" in l:
                            data["SampleRate_Before"] = float(re.findall(r"[-+]?\d*\.\d+|\d+", l)[0])
                        elif "Sample rate after allocations" in l:
                            data["SampleRate_After"] = float(re.findall(r"[-+]?\d*\.\d+|\d+", l)[0])
                        elif "Samples per symbol" in l:
                            data["SamplesPerSymbol"] = int(re.findall(r"\d+", l)[0])
                        elif "Carrier frequency" in l:
                            data["CarrierFrequency"] = int(re.findall(r"\d+", l)[0])
                        elif "Bit rate" in l:
                            data["BitRate"] = int(re.findall(r"\d+", l)[0])
                        elif "Message length" in l:
                            data["MessageLength"] = int(re.findall(r"\d+", l)[0])
                        elif "Message:" in l:
                            data["Message"] = l.split("Message:")[1].strip()
                        elif "Message bits:" in l:
                            bits = l.split("Message bits:")[1].strip()
                            data["MessageBits"] = bits

                    # Write to CSV
                    writer.writerow([
                        data.get("SampleRate_Before"),
                        data.get("SampleRate_After"),
                        data.get("SamplesPerSymbol"),
                        data.get("CarrierFrequency"),
                        data.get("BitRate"),
                        data.get("MessageLength"),
                        data.get("Message"),
                        data.get("MessageBits")
                    ])
                    file.flush()

                    print(f"✅ Data written: {data}")
                except Exception as e:
                    print(f"⚠️ Failed to parse buffer: {e}")
                finally:
                    buffer.clear()  # Clear for the next block
