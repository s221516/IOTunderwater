import csv
import collections
from gc import collect
import time
import numpy as np
import generatePayload
import pandas as pd
import scipy.signal as signal
from tkinter import W

import test
import config
import uuid
from receiver.receiverClass import NonCoherentReceiver
from receiver.record_audio import create_wav_file_from_recording
from errors import PreambleNotFoundError
from Transmitter import Transmitter
import generatePayload


def transmitter_setting_to_string():
    if config.USE_ESP:
        return "ESP"
    else:
        return "SG"

def testing_water_to_string():
    if isWaterThePool:
        return "pool"
    else:
        return "plastic"

def compute_len_of_bits(message):
    # NOTE: this is just to give the recieverClass the length of the transmitted bits, so you dont have to do it in config
    len_of_data_bits = len(message) * 8
    len_of_preamble = len(config.BINARY_BARKER)
    if config.CONVOLUTIONAL_CODING:
        len_of_data_bits = (len_of_data_bits * 2 + len_of_preamble + 12)
    elif config.HAMMING_CODING:
        len_of_data_bits = len_of_data_bits * 3 / 2 + len_of_preamble
    else:
        len_of_data_bits = len_of_data_bits + len_of_preamble

    # print("Len of data bits (receiver): ", len_of_data_bits)
    return len_of_data_bits

def logInCsv(id,original_message,decoded_message1,hamming_dist_without,decod_msg2,ham_dist_with, distance_to_speaker, 
             speaker_depth, test_description, original_message_in_bits, data_bits_nc, data_bits_nc_bandpass, avg_power_of_signal):


    transmitter_string = transmitter_setting_to_string()
    water_string = testing_water_to_string()
    headers = ["ID","Bitrate","Carrier Frequency","Average Power of signal", "Original Message","Decoded without bandpass","Hamming Dist without bandpass","Decoded with bandpass","Hamming Dist with bandpass","Encoding","Transmitter","Container","Speaker depth","Distance to speaker","Test description", "Original message in bits", "Data bits without bandpass", "Data bits with bandpass"]

    # Check if the file exists to determine if we need to write headers
    try:
        file_exists = open(config.FILE_NAME_DATA_TESTS).readline()
    except FileNotFoundError:
        file_exists = False

    with open(config.FILE_NAME_DATA_TESTS, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # Write headers only if file is empty
        if not file_exists:
            writer.writerow(headers)

        # Write the log entry
        writer.writerow([id,config.BIT_RATE,config.CARRIER_FREQ,avg_power_of_signal ,original_message,decoded_message1,hamming_dist_without,decod_msg2,ham_dist_with,config.ENCODING,transmitter_string, water_string,speaker_depth,distance_to_speaker,test_description, original_message_in_bits, data_bits_nc, data_bits_nc_bandpass])

def transmit_signal():
    transmitter = Transmitter(None, config.USE_ESP)

    #messages = ["Lick_on_these_big_fat_nuts_and_tell_me_what_you_think"]
    payload_sizes = [100]
    
    n = 5
    
    bitrates = [500] * n

    carrierfreqs = np.arange(1000, 30000, 1000)
    
    global test_description
    # test_description = f"Testing : average power of a signal"
    # test_description = f"Testing: does sending a message with a low correlation < 3 to barker 13 make a diffence?"
    test_description = f"Testing: Average power purely for check of interference"

    global speaker_depth
    speaker_depth = 200  # in cm

    global distance_to_speaker
    distance_to_speaker = 50  # in cm

    for payload in payload_sizes:
        for bitrate in bitrates:
            config.set_bitrate(bitrate)
            for carrierfreq in carrierfreqs:
                config.set_carrierfreq(carrierfreq)
                # message = generatePayload.generate_payload(payload)
                message = "U" * 12
                # create unique id for each test
                id = create_id()
                transmitter.transmit(message, carrierfreq, bitrate)
                record_seconds = transmitter.calculate_transmission_time(message)

                print(f"Recording for: {record_seconds} seconds")
                create_wav_file_from_recording(record_seconds, name=id)
                print("Recording done")
                    
                # NOTE: If distance to speaker is too long add a sleep time here
                # time.sleep(2)
                transmitter.stopTransmission()

                process_signal_for_testing(message, id)

def create_id():
    """Using uuid to create a unique id for each test. This is to avoid having to remember the last id used."""    
    return str(uuid.uuid4())

def process_signal_for_chat():
    # the minimum guaranteed length between two peaks of a premable is 20
    min_len_of_data_bits = 20
    
    nonCoherentReceiver = NonCoherentReceiver(band_pass=False)
    nonCoherentReceiver.set_len_of_data_bits(min_len_of_data_bits)
    msg_nc, _ = nonCoherentReceiver.decode()
    
    nonCoherentReceiverWithBandPass = NonCoherentReceiver(band_pass=True)
    nonCoherentReceiverWithBandPass.set_len_of_data_bits(min_len_of_data_bits)
    msg_nc_bp, _ = nonCoherentReceiverWithBandPass.decode()
    return msg_nc, msg_nc_bp

def process_signal_for_testing(message, id):

    print(f"Carrier frequency: {config.CARRIER_FREQ}")
    print(f"Bitrate: {config.BIT_RATE}")
    print(f"id is specfied : {config.IS_ID_SPECIFIED == id}")
    len_of_data_bits = compute_len_of_bits(message)
    nonCoherentReceiver = NonCoherentReceiver(id, band_pass = False)
    nonCoherentReceiver.set_len_of_data_bits(len_of_data_bits)
    # doesnt matter if band pass or not, the receiver will always use the same data bits
    avg_power_of_signal = nonCoherentReceiver.compute_average_power_of_signal()

    try:
        message_nc, debug_nc = nonCoherentReceiver.decode()
        # noncoherent_receiver.plot_simulation_steps()
    except PreambleNotFoundError:
        message_nc = "No preamble found"
        debug_nc = {}

    nonCoherentReceiverWithBandPass = NonCoherentReceiver(id, band_pass = True)
    nonCoherentReceiverWithBandPass.set_len_of_data_bits(len_of_data_bits)

    try:
        message_nc_bandpass, debug_nc_bandpass = nonCoherentReceiverWithBandPass.decode()
        # nonCoherentReceiverWithBandPass.plot_simulation_steps()
    except PreambleNotFoundError:
        message_nc_bandpass = "No preamble found"
        debug_nc_bandpass = {}
    
    if config.IS_ID_SPECIFIED != None:
        print("ID specified, not logging to csv")
        return

    logging_and_printing(message_nc,message_nc_bandpass,message,debug_nc,debug_nc_bandpass,id, avg_power_of_signal)


def logging_and_printing(message_nc,message_nc_bandpass,message,debug_nc,debug_nc_bandpass,id, avg_power_of_signal):
    print("Decoded message: no pass    ", message_nc)
    print("Decoded message, with pass: ", message_nc_bandpass)
    
    original_message_in_bits = config.string_to_bin_array(message)

    decoded_bits = []
    decoded_bits_bandpass = []

    data_bits_nc = "NaN"
    data_bits_nc_bandpass = "NaN"
    
    if debug_nc != {}:
        decoded_bits = debug_nc["bits_without_preamble"]
        data_bits_nc = debug_nc["all_data_bits"]
    
    if debug_nc_bandpass != {}:
        decoded_bits_bandpass = debug_nc_bandpass["bits_without_preamble"]
        data_bits_nc_bandpass = debug_nc_bandpass["all_data_bits"]
    
    if config.HAMMING_CODING:
        decoded_bits = list(map(int, decoded_bits))
        decoded_bits_bandpass = list(map(int, decoded_bits_bandpass))

    hamming_dist = config.hamming_distance(decoded_bits, original_message_in_bits)
    hamming_dist_bandpass = config.hamming_distance(decoded_bits_bandpass, original_message_in_bits)

    print("Hamming distance of msgs, no pass:   ", hamming_dist)
    print("Hamming distance of msgs, with pass  ", hamming_dist_bandpass)

    logInCsv(id,message,message_nc,hamming_dist,message_nc_bandpass,hamming_dist_bandpass, distance_to_speaker, speaker_depth, 
             test_description, original_message_in_bits, data_bits_nc, data_bits_nc_bandpass, avg_power_of_signal)


if __name__ == "__main__":
    
    
    if config.IS_ID_SPECIFIED == None:
        
        isWaterThePool = False
        transmit_signal()

    else:
        df = pd.read_csv("Received_data_for_tests.csv", sep=",")
        print(df.columns)
        original_message = df[df["ID"] == config.IS_ID_SPECIFIED]["Original Message"].values[0]
        decoded_message_without_bandpass = df[df["ID"] == config.IS_ID_SPECIFIED]["Decoded without bandpass"].values[0]
        decoded_with_bandpass = df[df["ID"] == config.IS_ID_SPECIFIED]["Decoded with bandpass"].values[0]
        bitrate = df[df["ID"] == config.IS_ID_SPECIFIED]["Bitrate"].values[0]
        carrier_freq = df[df["ID"] == config.IS_ID_SPECIFIED]["Carrier Frequency"].values[0]
        config.set_bitrate(bitrate)
        config.set_carrierfreq(carrier_freq)
        print("Original message:                 ", original_message)
        print("Decoded message without bandpass: ", decoded_message_without_bandpass)
        print("Decoded message with bandpass:    ", decoded_with_bandpass)
        corr = signal.correlate(config.string_to_bin_array(original_message), config.BINARY_BARKER, mode='same')
        is_not_similar_to_BARKER_13 = np.all(corr != 9)
        print(f"boolean check : {is_not_similar_to_BARKER_13}")
        counter = collections.Counter(corr)
        print("Correlation of the original message: ", counter)
        
        process_signal_for_testing(original_message, config.IS_ID_SPECIFIED)