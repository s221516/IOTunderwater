import csv
import time
import threading

import config
from receiver.receiverClass import NonCoherentReceiver, CoherentReceiver
from receiver.record_audio import (
    create_wav_file_from_recording,
    continuous_recording_with_threshold,
    get_avg_rms_value
)
from errors import PreambleNotFoundError
from encoding.hamming_codes import hamming_encode


def transmitter_setting_to_string():
    if isTransmitterESP:
        return "ESP"
    else:
        return "SG"

def compute_len_of_bits(message):
    # NOTE: this is just to give the recieverClass the length of the transmitted bits, so you dont have to do it in config
    len_of_data_bits = len(message) * 8
    len_of_preamble = len(config.BINARY_BARKER)
    if config.CONVOLUTIONAL_CODING:
        len_of_data_bits = (len_of_data_bits * 2 + len_of_preamble + 12)  # NOTE talk with Mathias here
    elif config.HAMMING_CODING:
        len_of_data_bits = len_of_data_bits * 3 / 2 + len_of_preamble
    else:
        len_of_data_bits = len_of_data_bits + len_of_preamble

    # print("Len of data bits (receiver): ", len_of_data_bits)
    return len_of_data_bits

def logInCsv(
    id,
    bitrate,
    carrierfreq,
    original_message,
    decoded_message1,
    hamming_dist_without,
    decod_msg2,
    ham_dist_with,
    filename=None,
):

    if filename is None:
        transmitter_string = transmitter_setting_to_string()
        filename = f"{transmitter_string}_plastic_testing_cf{carrierfreq}_400bps, {speaker_depth}sd, {distance_to_speaker}ds.csv"

    headers = [
        "ID",
        "Bitrate",
        "Carrier Frequency",
        "Original Message",
        "Decoded without bandpass",
        "Hamming Dist without bandpass",
        "Decoded with bandpass",
        "Hamming Dist with bandpass",
        "Encoding",
        "Transmitter",
        "Speaker depth",
        "Distance to speaker",
    ]

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
        writer.writerow(
            [
                id,
                bitrate,
                carrierfreq,
                original_message,
                decoded_message1,
                hamming_dist_without,
                decod_msg2,
                ham_dist_with,
                config.ENCODING,
                transmitter_setting_to_string(),
                speaker_depth,
                distance_to_speaker,
            ]
        )

def transmit_signal(isTransmitterESP: bool):

    # NOTE: a bit cursed to have this here
    if isTransmitterESP:
        import esp32test
    else:
        from transmitterPhysical import transmitPhysical
    
    messages = ["Hello_there"]

    n = 100
    bitrates = [100]

    carrierfreqs = [6000]

    global speaker_depth
    speaker_depth = 5  # in cm

    global distance_to_speaker
    distance_to_speaker = 50  # in cm

    id = 0
    for message in messages:
        for bitrate in bitrates:
            for carrierfreq in carrierfreqs:

                if config.RECORD_FOR_LOOP_TESTING:
                    # ESP OR SIGNAL GENRATOR
                    if isTransmitterESP:
                        print("Transmitting to ESP...")
                        esp32test.transmit_to_esp32(message, carrierfreq, bitrate)
                    else:
                        print("Transmitting to signal generator...")
                        transmitPhysical(message, carrierfreq, bitrate)


                process_signal(message, carrierfreq, bitrate, id)

def process_signal(message, carrierfreq, bitrate, id):
    # NOTE: a bit cursed to have this here
    if not isTransmitterESP:
        from transmitterPhysical import stopTransmission

    len_of_bits = compute_len_of_bits(message)

    time.sleep(0.5)

    record_seconds = round((len_of_bits / bitrate) * 5)

    if config.RECORD_FOR_LOOP_TESTING:
        print(f"Recording for: {record_seconds} seconds")
        create_wav_file_from_recording(record_seconds)

    if bitrate == 300 or bitrate == 400:
        time.sleep(1)
    else:
        time.sleep(0.1)

    if not isTransmitterESP:
        stopTransmission()

    nonCoherentReceiver = NonCoherentReceiver(bitrate, carrierfreq, band_pass=False)
    nonCoherentReceiver.set_len_of_data_bits(len_of_bits)
    nonCoherentReceiver.set_transmitter(isTransmitterESP)
    nonCoherentReceiverWithBandPass = NonCoherentReceiver(bitrate, carrierfreq, band_pass=True)
    nonCoherentReceiverWithBandPass.set_len_of_data_bits(len_of_bits)
    nonCoherentReceiverWithBandPass.set_transmitter(isTransmitterESP)

    # try:
    message_nc, debug_nc = nonCoherentReceiver.decode()
    message_nc_bandpass, debug_nc_bandpass = (nonCoherentReceiverWithBandPass.decode())
    
    if config.RECORD_FOR_LOOP_TESTING:
        logging_and_printing(message_nc, message_nc_bandpass, message, debug_nc, debug_nc_bandpass, bitrate, carrierfreq, id)
    else:
        chatting(message_nc, message_nc_bandpass)
     
def logging_and_printing(message_nc, message_nc_bandpass, message, debug_nc, debug_nc_bandpass, bitrate, carrierfreq, id):
    print("Decoded message: no pass    ", message_nc)
    print("Decoded message, with pass: ", message_nc_bandpass)

    # nonCoherentReceiver.plot_simulation_steps()
    # nonCoherentReceiverWithBandPass.plot_simulation_steps()

    original_message_in_bits = config.string_to_bin_array(message)
    decoded_bits = debug_nc["bits_without_preamble"]
    decoded_bits_bandpass = debug_nc_bandpass["bits_without_preamble"]

    if config.HAMMING_CODING:
        decoded_bits = list(map(int, decoded_bits))
        decoded_bits_bandpass = list(map(int, decoded_bits_bandpass))

    hamming_dist = config.hamming_distance(decoded_bits, original_message_in_bits)
    hamming_dist_bandpass = config.hamming_distance(decoded_bits_bandpass, original_message_in_bits)

    print("Hamming distance of msgs, no pass:   ", hamming_dist)
    print("Hamming distance of msgs, with pass  ", hamming_dist_bandpass)

    # except PreambleNotFoundError:
    #     message_nc = "No preamble found"
    #     message_nc_bandpass = "No preamble found"

    logInCsv(
        id,
        bitrate,
        carrierfreq,
        message,
        message_nc,
        hamming_dist,
        message_nc_bandpass,
        hamming_dist_bandpass,
    )
    id += 1    

def chatting(message_nc, message_nc_bandpass): 
    print(f"Mathias: {message_nc}") 


def main():
    global isTransmitterESP
    isTransmitterESP = True
    transmit_signal(isTransmitterESP)

def aqua_chat():
    print("Welcome to AquaChat!")
    print("Type a message to your aqua-friend!")
    try:
        while True:
            threshold_val = 10
            continuous_recording_with_threshold(threshold_val)
            if get_avg_rms_value() > threshold_val:
                main()
            else:
                msg_to_transmit = input("You: ")
    except KeyboardInterrupt:
        print("Left the chat...")
    

    
if __name__ == "__main__":
    if config.RECORD_FOR_LOOP_TESTING:
        main()
    else:
        main()