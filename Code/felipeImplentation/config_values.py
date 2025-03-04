


SAMPLE_RATE =  96000  # this capped by the soundcard, therefore, this is non-changeable
CARRIER_FREQ = 15200  #  15200 Hz
BIT_RATE = 1000  # 1 Khz
SAMPLES_PER_SYMBOL = 16 # TODO investigate this ratio int(SAMPLE_RATE / BIT_RATE)

PATH_TO_WAV_FILE = "Code/dsp/data/recording.wav"
#MESSAGE = "the quick brown fox jumps over the lazy dog while vexd zebras fight for joy! @#$%^&()_+[]{}|;:,.<>/?~` \ The 5 big oxen love quick daft zebras & dogs.>*"
MESSAGE = "A"

CUTOFF = 10000 #48000-1 #if our sample rate is 96000, then the spectrum of waves we can see is from 0 to 0.5 * sample rate, thus the cutoff should be here.
ORDER  = 5




