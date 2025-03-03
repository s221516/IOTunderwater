
# nyquist = 30400
# 31650 / 15200
SAMPLE_RATE =  20000000  # 50 Khz
CARRIER_FREQ = 15200  #  15200 Hz
BIT_RATE = 1000  # 1 Khz
SAMPLES_PER_BIT = int(SAMPLE_RATE / BIT_RATE)
CUT_OFF_FREQ = BIT_RATE * 3
ACTIVATION_ENERGY_THRESHOLD = 0.4 # lower bound for start of 
THRESHOLD_BINARY_VAL = 170 # defines when a pixel should be black or white when converting from RGB to black-white
# with 200 samples per bit and 0.45 noise amplitude -> about upper limit, we get a valid picture 1/3 times
NOISE_AMPLITUDE = 0.0 # noise
PATH_TO_WAV_FILE = "./data/recording.wav"
PATH_TO_PICTURE = "./data/doge.jpg"

SAMPLE_RATE_FOR_WAV_FILE = 44100 # Hz
RECORD_SECONDS = 5

with open("picture_in_binary.txt", "r") as file:
    picture_in_binary = file.read()

all_letters = "the quick brown fox jumps over the lazy dog while vexd zebras fight for joy! @#$%^&()_+[]{}|;:,.<>/?~` \ The 5 big oxen love quick daft zebras & dogs.>*"
small_test = "This is: 14"

picture_in_binary_with_prefix = "p" + picture_in_binary
text_with_prefix = small_test

MESSAGE = "t" + text_with_prefix
