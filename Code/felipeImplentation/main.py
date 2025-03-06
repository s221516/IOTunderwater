from transmitterOOK import transmit
from recieveSimple import recieve
from config_values import PATH_TO_WAV_FILE, SAMPLE_RATE, MESSAGE, RECORD_SECONDS
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from record_audio import create_wav_file_from_recording




def main():

    #transmit(MESSAGE)
    create_wav_file_from_recording(RECORD_SECONDS)
    recieve()
    

if __name__ == "__main__":
    main()
