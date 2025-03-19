from receiver.receiverClass import NonCoherentReceiver, CoherentReceiver
import config
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from receiver.record_audio import create_wav_file_from_recording
from transmitterPhysical import transmitPhysical, stopTransmission

import csv

def logInCsv(id, bitrate, carrierfreq, original_message, decoded_message, hamming_distance, filename="hamming_code_implementation_v3.csv"):

    headers = ["ID", "Bitrate", "Carrier Frequency", "Original Message", "Decoded Message", "Hamming Distance", "Hamming Encoding"]

    # Check if the file exists to determine if we need to write headers
    try:
        file_exists = open(filename).readline()
    except FileNotFoundError:
        file_exists = False
    
    with open(filename, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        
        # Write headers only if file is empty
        if not file_exists:
            writer.writerow(headers)
        
        # Write the log entry
        writer.writerow([id, bitrate, carrierfreq, original_message, decoded_message, hamming_distance, config.HAMMING_CODING])

def testing():
    #test words
    # all_letters = "the quick brown fox jumps over the lazy dog while vexd zebras fight for joy! @#$%^&()_+[]{}|;:,.<>/?~`.>*"
    all_letters = "the quick brown fox jumps over the lazy dog, test test test, suggma"
    
    messages = ["hello world"]
    
    # test bitrates  
    # 16 * 9 * 15 * (2)
    # bitrates = np.arange(200, 1000, 50)

    bitrates = [100] * 10

    #test carrier frequencies
    # carrierfreqs = np.arange(2000, 12000, 1000)
    carrierfreqs = [6000]

    id = 0
    for message in messages:
        for bitrate in bitrates:
            for carrierfreq in carrierfreqs:
                transmitPhysical(message, carrierfreq, bitrate)

                time.sleep(1.5)

                # Start recording
                if (config.MAKE_NEW_RECORDING):
                    create_wav_file_from_recording(config.RECORD_SECONDS)

                time.sleep(0.1)
                
                stopTransmission()
                # Non-Coherent demodulation
                nonCoherentReceiver = NonCoherentReceiver(bitrate, carrierfreq)
                
                try:
                    message_nc, debug_nc = nonCoherentReceiver.decode()
                    print("Decoded message: ", message_nc)
                    hamming_dist = config.hamming_distance(config.string_to_bin_array(message_nc), config.string_to_bin_array(message))
                    print("Hamming distance of msgs: ", hamming_dist) 
                except TypeError:
                    message_nc = "No preamble found"
   

                logInCsv(id, bitrate, carrierfreq, message, message_nc, hamming_dist)
                id+=1


def main():

    transmitPhysical(config.MESSAGE, config.CARRIER_FREQ, config.BIT_RATE)

    time.sleep(2)

    # Start recording
    if (config.MAKE_NEW_RECORDING):
       create_wav_file_from_recording(config.RECORD_SECONDS)

    time.sleep(0.1)
    
    stopTransmission()
    # Non-Coherent demodulation
    receiver_non_coherent = NonCoherentReceiver.from_wav_file(config.PATH_TO_WAV_FILE)
    
    message_nc, debug_nc = receiver_non_coherent.decode()
    print(f"Non-Coherent Decoded: {message_nc}")
    receiver_non_coherent.plot_simulation_steps()
        
    # # Coherent demodulation
    # receiver_coherent = CoherentReceiver.from_wav_file(PATH_TO_WAV_FILE)
    # message_c, debug_c = receiver_coherent.decode()
    # print(f"Coherent Decoded: {message_c}")
    # receiver_coherent.plot_simulation_steps()





if __name__ == "__main__":
    testing()
