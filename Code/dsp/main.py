from receiver.receiverClass import NonCoherentReceiver, CoherentReceiver
from config_values import PATH_TO_WAV_FILE, CARRIER_FREQ, BIT_RATE, MAKE_NEW_RECORDING, RECORD_SECONDS, MESSAGE
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from receiver.record_audio import create_wav_file_from_recording
from transmitterPhysical import transmitPhysical, stopTransmission

import csv

def logInCsv(id, bitrate, carrierfreq, original_message, decoded_message, filename="log.csv"):
    headers = ["ID", "Bitrate", "Carrier Frequency", "Original Message", "Decoded Message"]

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
        writer.writerow([id, bitrate, carrierfreq, original_message, decoded_message])
def testing():
    #test words
    all_letters = "the quick brown fox jumps over the lazy dog while vexd zebras fight for joy! @#$%^&()_+[]{}|;:,.<>/?~` \ The 5 big oxen love quick daft zebras & dogs.>*"
    helloWorld = "Hello World!"
    messages =  [helloWorld]
    #test bitrates 
    bitrates = [1000]

    #test carrier frequencies
    carrierfreqs = [5000]

    id = 0
    for message in messages:
        for bitrate in bitrates:
            for carrierfreq in carrierfreqs:

                transmitPhysical(message, carrierfreq, bitrate)
                time.sleep(1)

                # Start recording
                if (MAKE_NEW_RECORDING):
                    create_wav_file_from_recording(RECORD_SECONDS)

                # Non-Coherent demodulation
                receiver_non_coherent = NonCoherentReceiver.from_wav_file(PATH_TO_WAV_FILE)

                try:
                    message_nc, debug_nc = receiver_non_coherent.decode()
                except TypeError:
                    message_nc = "No preamble found"
   
        
                logInCsv(id, bitrate, carrierfreq, message, message_nc)

    stopTransmission()




def main():

    transmitPhysical(MESSAGE, CARRIER_FREQ, BIT_RATE)
    time.sleep(1)

    # Start recording
    if (MAKE_NEW_RECORDING):
       create_wav_file_from_recording(RECORD_SECONDS)
   
    # Non-Coherent demodulation
    receiver_non_coherent = NonCoherentReceiver.from_wav_file(PATH_TO_WAV_FILE)
    
    try:
        message_nc, debug_nc = receiver_non_coherent.decode()
        print(f"Non-Coherent Decoded: {message_nc}")
        receiver_non_coherent.plot_simulation_steps()

    finally:
        stopTransmission()

    # # Coherent demodulation
    # receiver_coherent = CoherentReceiver.from_wav_file(PATH_TO_WAV_FILE)
    # message_c, debug_c = receiver_coherent.decode()
    # print(f"Coherent Decoded: {message_c}")
    # receiver_coherent.plot_simulation_steps()


if __name__ == "__main__":
    main()
