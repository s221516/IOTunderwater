from receiver.receiverClass import NonCoherentReceiver, CoherentReceiver
import config
import time
from receiver.record_audio import create_wav_file_from_recording

if (config.STAGE_1):
    from transmitterPhysical import transmitPhysical, stopTransmission

from errors import PreambleNotFoundError
from encoding.hamming_codes import hamming_encode

import csv

def logInCsv(id, bitrate, carrierfreq, original_message, decoded_message1, hamming_dist_without, decod_msg2, ham_dist_with, filename="large_test_hamming_encoding.csv"):

    headers = ["ID", "Bitrate", "Carrier Frequency", "Original Message", "Decoded without bandpass", "Hamming Dist without bandpass", "Decoded with bandpass", "Hamming Dist with bandpass", "Encoding"]

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
        writer.writerow([id, bitrate, carrierfreq, original_message, decoded_message1, hamming_dist_without, decod_msg2, ham_dist_with, "Hamming Encoding"])

def signal_generator_testing():

    all_letters = "the quick brown fox jumps over the lazy dog while vexd zebras fight for joy!>*"
    
    messages = ["Hello_there"]

    bitrates = [100]

    carrierfreqs = [6000]

    id = 0
    for message in messages:
        for bitrate in bitrates:
            for carrierfreq in carrierfreqs:
                transmitPhysical(message, carrierfreq, bitrate) 

                # NOTE: this is just to give the recieverClass the length of the transmitted bits, so you dont have to do it in config
                len_of_data_bits = len(message) * 8 
                len_of_preamble = len(config.BINARY_BARKER)
                if (config.CONVOLUTIONAL_CODING):
                    len_of_data_bits = len_of_data_bits * 3 + len_of_preamble
                elif (config.HAMMING_CODING):
                    len_of_data_bits = len_of_data_bits * 3/2 + len_of_preamble
                else:
                    len_of_data_bits = len_of_data_bits + len_of_preamble

                time.sleep(0.75)

                # NOTE: to make the time of recording dynamic 
                record_seconds = round((len_of_data_bits / bitrate) * 5)

                if (config.MAKE_NEW_RECORDING):
                    print(f"Recording for: {record_seconds} seconds")
                    create_wav_file_from_recording(record_seconds)

                time.sleep(0.1)
                
                stopTransmission()

                nonCoherentReceiver = NonCoherentReceiver(bitrate, carrierfreq, band_pass=False)      
                nonCoherentReceiver.set_len_of_data_bits(len_of_data_bits)    
                nonCoherentReceiverWithBandPass = NonCoherentReceiver(bitrate, carrierfreq, band_pass=True)                
                nonCoherentReceiverWithBandPass.set_len_of_data_bits(len_of_data_bits)

                try:
                    message_nc, debug_nc = nonCoherentReceiver.decode()
                    message_nc_bandpass, debug_nc_bandpass = nonCoherentReceiverWithBandPass.decode()
                    print("Decoded message: no pass    ", message_nc)
                    print("Decoded message, with pass: ", message_nc_bandpass)

                    # nonCoherentReceiver.plot_simulation_steps()
                    # nonCoherentReceiverWithBandPass.plot_simulation_steps()

                    original_message_in_bits = config.string_to_bin_array(message)
                    decoded_bits = debug_nc["bits_without_preamble"]
                    decoded_bits_bandpass = debug_nc_bandpass["bits_without_preamble"]

                    if (config.HAMMING_CODING):
                        decoded_bits = list(map(int, decoded_bits))
                        decoded_bits_bandpass = list(map(int, decoded_bits_bandpass))

                    hamming_dist = config.hamming_distance(decoded_bits, original_message_in_bits)
                    hamming_dist_bandpass = config.hamming_distance(decoded_bits_bandpass, original_message_in_bits)
                    
                    print("Hamming distance of msgs, no pass:   ", hamming_dist) 
                    print("Hamming distance of msgs, with pass  ", hamming_dist_bandpass) 
                
                except PreambleNotFoundError:
                    message_nc = "No preamble found"
                    message_nc_bandpass = "No preamble found"

                logInCsv(id, bitrate, carrierfreq, message, message_nc, hamming_dist, message_nc_bandpass, hamming_dist_bandpass)
                id+=1


def esp32_testing():
    if (config.MAKE_NEW_RECORDING):
       create_wav_file_from_recording(config.RECORD_SECONDS)

    time.sleep(0.1)

    nonCoherentReceiver = NonCoherentReceiver(config.BIT_RATE, config.CARRIER_FREQ, band_pass=False)       
    
    try:
        message_nc, debug_nc = nonCoherentReceiver.decode()
        decoded_bits = debug_nc["bits_without_preamble"]
        print(f"Bits received: {decoded_bits}")
        print(len(decoded_bits))
        print(f"Decoded message, no pass: {message_nc}")
        nonCoherentReceiver.plot_simulation_steps()
        
    except PreambleNotFoundError:
        message_nc = "No preamble found"


if __name__ == "__main__":
    signal_generator_testing()


