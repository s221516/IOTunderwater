import csv
import time
import pandas as pd
import uuid
from receiver.DSP import NonCoherentDemodulator
from receiver.Recorder import Recorder
from Transmitter import Transmitter
import os
from utilityFunctions import string_to_bin_array, generate_payload
from encoding.hamming_codes import hamming_distance
from config import (
    USE_ESP,
    PAYLOAD_SIZES,
    BITRATES,
    CARRIERFREQS,
    IS_ID_SPECIFIED,
    LOG_DATA_USING_IDS,
    set_bitrate,
    set_carrierfreq,
    LOG
)



def startNewTest():
    for payload in PAYLOAD_SIZES:
        for bitrate in BITRATES:
            set_bitrate(bitrate)
            for carrier_frequency in CARRIERFREQS:
                set_carrierfreq(carrier_frequency)
                LOG["bitrate"] = bitrate
                LOG["carrier_frequency"] = carrier_frequency

                #Generate the message to transmit (string)
                message = generate_payload(payload)
                LOG["message"] = message
                LOG["message_in_bits"] = string_to_bin_array(LOG["message"])
                
                
                #Start transmitting
                # transmitter = Transmitter(USE_ESP)
                # transmitter.transmit(message, carrier_frequency, bitrate)
                
                #Record
             
                recorder = Recorder() 
                record_seconds = recorder.calculate_recording_time(message)
                record_seconds = 10
                if USE_ESP:
                    time.sleep(4.5931)  #TODO idk why we sleep here
                else:
                    time.sleep(1)
                
                id = str(uuid.uuid4()) #Create unique id for each recording
                LOG["id"] = id 
                recorder.record(record_seconds, name=id)
                
                # #Stop transmission 
                # transmitter.stopTransmission()

                demodulation()
                log_to_csv()

def redoOldTest():
    
    for id in IS_ID_SPECIFIED:
        LOG["id"] = "Using existing ID" 
        df = pd.read_csv(NAME_OF_DATA_FILE, sep=",")
        LOG["message"] = df[df["id"] == id]["message"].values[0]
        LOG["estimated_message"] = df[df["id"] == id]["estimated_message"].values[0]
        LOG["estimated_message_bandpass"] = df[df["id"] == id]["estimated_message_bandpass"].values[0]
        LOG["bitrate"] = df[df["id"] == id]["bitrate"].values[0]
        LOG["carrier_freq"] = df[df["id"] == id]["carrier_freq"].values[0]
        set_bitrate(LOG["bitrate"]) 
        set_carrierfreq(LOG["carrier_freq"])

        demodulation()
        
        if LOG_DATA_USING_IDS:
            LOG["id"] = str(uuid.uuid4())  # Generate a new ID for the log entry
            LOG["specified_id"] = id
            log_to_csv()

def demodulation():
    
    #Demodulation of the recorded signal
    #Using no bandpass filter

    nonCoherentDemodulator = NonCoherentDemodulator(LOG["id"], band_pass = False) 
    nonCoherentDemodulator.set_len_of_data_bits(LOG["message"])
    estimated_message, debug_log = nonCoherentDemodulator.demodulate()
    LOG["estimated_message"] = estimated_message
    LOG["data_bits_between_peaks"] = debug_log["data_bits_between_peaks"]
    LOG["hamming_distance"] = hamming_distance(
                                            debug_log["estimated_message_in_bits"], 
                                            LOG["message_in_bits"])


    #Using bandpass filter
    nonCoherentDemodulatorWithBandpass = NonCoherentDemodulator(LOG["id"], band_pass = True)
    nonCoherentDemodulatorWithBandpass.set_len_of_data_bits(LOG["message"])
    estimated_message_bandpass, debug_log_bandpass = nonCoherentDemodulatorWithBandpass.demodulate()
    LOG["estimated_message_bandpass"] = estimated_message_bandpass
    LOG["data_bits_between_peaks_bandpass"] = debug_log_bandpass["data_bits_between_peaks"]
    LOG["hamming_distance_bandpass"] = hamming_distance(
                                            debug_log_bandpass["estimated_message_in_bits"], 
                                            LOG["message_in_bits"])
    

    avg_power_of_signal = nonCoherentDemodulator.compute_average_power_of_signal()
    LOG["average_power_of_signal"] = avg_power_of_signal

def log_to_csv():
    """Write the current LOG to CSV file."""

    # Build the full path dynamically
    csv_file_path = f"Code/dsp/data/csv_logs/Logging_Data.csv"
    
    # Create the csv_logs directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    headers = list(LOG.keys())
    
    # Check if file exists
    try:
        file_exists = open(csv_file_path).readline()
    except FileNotFoundError:
        file_exists = False

    with open(csv_file_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        
        # Write headers only if file is empty
        if not file_exists:
            writer.writerow(headers)
        
        # Write the log entry
        writer.writerow(list(LOG.values()))

if __name__ == "__main__":   
    IsNewTest = IS_ID_SPECIFIED == None
    if IsNewTest:
        startNewTest()
    else:
        redoOldTest()
