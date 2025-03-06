import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from initPorts import initPort
from config_values import MESSAGE, SAMPLES_PER_SYMBOL, SAMPLE_RATE, CARRIER_FREQ, PATH_TO_WAV_FILE, PORT
np.set_printoptions(precision=4, suppress=True)

import time
def make_square_wave(message: str): #ON OFF KEYING
    message_binary = ''.join(format(ord(i), '08b') for i in message)
    print(f"Message in binary: {message_binary}")
    # TODO: determine the exact number of samples per bit that makes sense in relation to our sample rate 
    # and bits per second and also how this is done in the signal generator
    
    square_wave = []
    for bit in message_binary:
        if bit == '0':
            square_wave += [-1]
        elif bit == '1':
            square_wave += [1] 

    duration_of_wav_signal = len(square_wave) / SAMPLE_RATE

    time_array = np.linspace(0, duration_of_wav_signal, len(square_wave))
    square_wave = np.array(square_wave)

    return square_wave, time_array

def make_carrier_wave(time_array) -> np.array:
    # NOTE: to add noise check the bottom of freq domain section of pysdr
    carrier_wave = np.sin(2 * np.pi * CARRIER_FREQ * time_array) 
    
    return carrier_wave

def create_modulated_wave(square_wave: np.array, carrier_wave: np.array) -> np.array:
    return square_wave * carrier_wave

def save_wave_file(modulated_wave):
    wav.write(PATH_TO_WAV_FILE, SAMPLE_RATE, modulated_wave)
    
ser = initPort(PORT)

def send_command(command):
    """Send a command and wait for a response if it contains '?'."""
    commands = command.strip().split("\n")
    
    for cmd in commands:
        cmd = cmd.strip()
        if not cmd:
            continue
        
        ser.write((cmd + "\r\n").encode())  # Send normal command
        print(f"Sent: {cmd}")  

        if cmd[0:8]== "DATA:DAC":
            time.sleep(0.01)

        
        if "?" in cmd:  # If it's a query, wait for a response
            response = ser.readline().decode().strip()
            print(f"Response: {response}")
            time.sleep(0.1)  # Small delay to avoid overloading the buffer
        else:
            time.sleep(0.05)  # Short delay for non-query commands

def transmit(message=MESSAGE):
    """Run virtual transmission process - send commands to signal generator"""
    square_wave, time_array = make_square_wave(message)
 
    arb_wave_form_command = "DATA:DAC VOLATILE, " + ", ".join(map(str, square_wave * 2047))
 
    name = "COCK"
    command = "" \
    "FREQ 1 \n \r"\
    "VOLTage 10 \n \r" \
    "VOLTage:OFFSet 0.0 \n \r" \
    + arb_wave_form_command + "\n \r" \
    "DATA:COPY " + name + "\n \r" \
    "FUNC:USER " + name + "\n \r" \
    "FUNC USER\n \r"\
    "FUNC:USER?\n\r" \
    "OUTPut ON\n\r" \
    
   
    command = "" \
    "APPL:SIN 1, 100000, 0\n\r"
        
    send_command(command)
