def message_toBitArray(message: str): 
    message_binary = ''.join(format(ord(i), '08b') for i in message)
    # TODO: determine the exact number of samples per bit that makes sense in relation to our sample rate 
    # and bits per second and also how this is done in the signal generator
    
    square_wave = []
    for bit in message_binary:
        if bit == '0':
            square_wave += [0]
        elif bit == '1':
            square_wave += [1] 

    return square_wave

def hamming_distance(received, expected):
    """Computes Hamming distance between received bits and expected bits."""
    if (len(received) != len(expected)):
        return 
    else:
        return sum(r != e for r, e in zip(received, expected))

def bytes_to_bin_array(byte_array):
    """Converts a byte array into a binary array."""
    bin_array = []
    for byte in byte_array:
        bin_array.extend([int(bit) for bit in format(byte, '08b')])
    return bin_array

def string_to_bin_array(string):
    """Converts a string into a binary array."""
    byte_array = string.encode('utf-8')
    return bytes_to_bin_array(byte_array)

def set_bitrate(value):
    global BIT_RATE
    BIT_RATE = value

def set_carrierfreq(value):
    global CARRIER_FREQ
    CARRIER_FREQ = value


# nyquist = 30400
# 31650 / 15200
import numpy as np
SAMPLE_RATE = 96000  # this capped by the soundcard, therefore, this is non-changeable

BIT_RATE = 200
CARRIER_FREQ = 6000  #  15200 Hz
SAMPLES_PER_SYMBOL = int(SAMPLE_RATE / BIT_RATE) 
CUT_OFF_FREQ = (CARRIER_FREQ + BIT_RATE) // 2  # TODO: check this value


THRESHOLD_BINARY_VAL = 170  # defines when a pixel should be black or white when converting from RGB to black-white
# with 200 samples per bit and 0.45 noise amplitude -> about upper limit, we get a valid picture 1/3 times
NOISE_AMPLITUDE = 0.0  # noise
PATH_TO_WAV_FILE = "Code/dsp/data/recording.wav"
PATH_TO_PICTURE = "./data/doge.jpg"
PREAMBLE_BASE = message_toBitArray("G")
BINARY_BARKER = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]
# BINARY_BARKER = [1,1,1,-1,-1,1,-1]
REPETITIONS = 3
PREAMBLE_PATTERN = PREAMBLE_BASE * REPETITIONS

EXPECTED_LEN_OF_DATA_BITS = 145

CONVOLUTIONAL_CODING = False 
HAMMING_CODING = True

APPLY_BAKER_PREAMBLE = True
APPLY_AVERAGING_PREAMBLE = False

# SAMPLE_RATE_FOR_WAV_FILE = 44100  # Hz
RECORD_SECONDS = 4
MAKE_NEW_RECORDING = True

dark_horse_lyrics = """AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABBBCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"""


# with open("picture_in_binary.txt", "r") as file:
#     picture_in_binary = file.read()


# picture_in_binary_with_prefix = "p" + picture_in_binary



all_letters = "the quick brown fox jumps over the lazy dog while vexd zebras fight for joy! The 5 big oxen love quick daft zebras & dogs.>* there"
small_test = "This is: 14"
A = "A"

MESSAGE = "Hello_there"
PORT = "COM11"
