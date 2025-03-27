from receiver.receiverClass import NonCoherentReceiver, CoherentReceiver
import config
import time
import threading
import numpy as np
from receiver.record_audio import create_wav_file_from_recording
from transmitterPhysical import transmitPhysical, stopTransmission
from errors import PreambleNotFoundError

import csv

def logInCsv(id, bitrate, carrierfreq, original_message, decoded_message1, decoded_message2, decoded_message3, filename="testing_other_method.csv"):

    headers = ["ID", "Bitrate", "Carrier Frequency", "Original Message", "Decoded without bandpass", "Decoded with bandpass", "Decoded with bandpass and other method"]

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
        writer.writerow([id, bitrate, carrierfreq, original_message, decoded_message1, decoded_message2, decoded_message3])

def testing():
    #test words
    # all_letters = "the_quick_brown_fox_jumps_over_the_lazy_dog_while_vexd_zebras_fight_for_joy!>*"

    all_letters = "the quick brown fox jumps over the lazy dog while vexd zebras fight for joy!>*"
    
    messages = ["Hello_there"]

    bitrates = [350] * 4

    # test carrier frequencies
    # carrierfreqs = np.arange(2000, 12000, 1000)
    carrierfreqs = [6000]

    id = 0
    for message in messages:
        for bitrate in bitrates:
            for carrierfreq in carrierfreqs:
                transmitPhysical(message, carrierfreq, bitrate) 

                time.sleep(1.0)

                # Start recording
                if (config.MAKE_NEW_RECORDING):
                    create_wav_file_from_recording(config.RECORD_SECONDS)

                time.sleep(0.1)
                
                stopTransmission()
                # Non-Coherent demodulation
                nonCoherentReceiver = NonCoherentReceiver(bitrate, carrierfreq, False, False)            
                nonCoherentReceiverWithBandPass = NonCoherentReceiver(bitrate, carrierfreq, True, False)
                nonCoherentReceiverWithBandPassOtherMethod = NonCoherentReceiver(bitrate, carrierfreq, True, True)
                
                try:
                    message_nc, debug_nc = nonCoherentReceiver.decode()
                    message_nc_bandpass, debug_nc_bandpass = nonCoherentReceiverWithBandPass.decode()
                    message_nc_bandpass_othermethod, debug_nc_bandpass_om = nonCoherentReceiverWithBandPassOtherMethod.decode()
                    print("Decoded message, false, false: ", message_nc)
                    print("Decoded message, true, false: ", message_nc_bandpass)
                    print("Decoded message, true, true: ", message_nc_bandpass_othermethod)
                    # nonCoherentReceiver.plot_simulation_steps()
                    # nonCoherentReceiver.plot_bandpass_comparison()
                    original_message_in_bits = config.string_to_bin_array(message)
                    hamming_dist = config.hamming_distance(debug_nc["bits_without_preamble"], original_message_in_bits)
                    hamming_dist_bandpass = config.hamming_distance(debug_nc_bandpass["bits_without_preamble"], original_message_in_bits)
                    hamming_dist_bandpass_othermethod = config.hamming_distance(debug_nc_bandpass_om["bits_without_preamble"], original_message_in_bits)
                    print("Hamming distance of msgs, no pass:        ", hamming_dist) 
                    print("Hamming distance of msgs, with pass       ", hamming_dist_bandpass) 
                    print("Hamming distance of msgs, with pass and om", hamming_dist_bandpass_othermethod) 
                
                except PreambleNotFoundError:
                    message_nc = "No preamble found"
                    message_nc_bandpass = "No preamble found"
                    message_nc_bandpass_othermethod = "No preamble found"

                logInCsv(id, bitrate, carrierfreq, message, message_nc, message_nc_bandpass, message_nc_bandpass_othermethod)
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
