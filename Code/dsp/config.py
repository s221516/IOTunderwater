

import numpy as np
import scipy.signal as signal
import os


TEST_DESCRIPTION = "Test123"

#Testing settings
TESTING_ENVIRONMENT = "Pool"
SPEAKER_DEPTH = 1           # in meters
DISTANCE_TO_SPEAKER = 300   # in cm
USE_ESP = False             # False = signal generator, True = ESP32
REP_ESP = 5                 # Number of times the payload is transmitted from the ESP32

PAYLOAD_SIZES = [100]       #Tests all the payload sizes in this list
BITRATES = [1000]           #Tests all the bitrates in this list
CARRIERFREQS = [2000]       #Tests all the carrier frequencies in this list

#Logging settings
IS_ID_SPECIFIED = None
LOG_DATA_USING_IDS = False  #If True, a new ID will be generated for each specified ID. If False, we won't generate new logings.

#Cable settings
MIC_INDEX = 1               #Recorder.listAvaliableDevices() will print the index of the mic
TRANSMITTER_PORT = "COM11"
# TRANSMITTER_PORT = "/dev/cu.usbserial-0232D158"


#Encoding settings
HAMMING_CODING = False
CONVOLUTIONAL_CODING = False


#--- Random functions / variables ---
def set_bitrate(value):
    global BIT_RATE
    BIT_RATE = value
    
def set_carrierfreq(value):
    global CARRIER_FREQ
    CARRIER_FREQ = value

def transmitter_to_string():
    if USE_ESP:
        return "ESP"
    else:
        return "SG"

def encoding_to_string():
    if CONVOLUTIONAL_CODING:
        return "Convolutional Encoding"
    elif HAMMING_CODING:
        return "Hamming Encoding"
    else:
        return "No Encoding"

SAMPLE_RATE = 96000  # this capped by the soundcard, therefore, this is non-changeable
BIT_RATE = 500 #Dont change, 
CARRIER_FREQ = 11000 #Dont change
SAMPLES_PER_SYMBOL = int(SAMPLE_RATE / BIT_RATE)
CUT_OFF_FREQ = (CARRIER_FREQ + BIT_RATE) // 2
BINARY_BARKER = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]
BIPOLAR_BARKER = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]


#--- Loging stuff ---
LOG = {
    "id": None,
    # Data related to the message
    "message": None,
    "message_in_bits": None,

    #Without bandpass
    "estimated_message": None,
    "data_bits_between_peaks": None,
    "hamming_distance": None,

    #With bandpass
    "estimated_message_bandpass": None,
    "data_bits_between_peaks_bandpass": None,
    "hamming_distance_bandpass": None,

    # Metadata    
    "specified_id": None,
    "bitrate": None,
    "carrier_frequency": None,
    "average_power_of_signal": None,
    "speaker_depth": SPEAKER_DEPTH,
    "distance_to_speaker": DISTANCE_TO_SPEAKER,
    "transmitter": transmitter_to_string(),
    "test_description": TEST_DESCRIPTION,
    "encoding": encoding_to_string(),
    "testing_environment": TESTING_ENVIRONMENT
}

