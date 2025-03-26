from initPorts import initPort
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from encoding.convolutional_encoding import conv_encode
from encoding.hamming_codes import hamming_encode

from config import (
    PORT, 
    CONVOLUTIONAL_CODING,
    PREAMBLE_PATTERN,
    BINARY_BARKER,
    APPLY_BAKER_PREAMBLE,
    HAMMING_CODING

)

ser = initPort(PORT)
def send_command(command):
    """Send a command and wait for a response if it contains '?'."""
    commands = command.strip().split("\n")
    
    for cmd in commands:
        cmd = cmd.strip()
        if not cmd:
            continue
        
        ser.write((cmd + "\r\n").encode())  # Send normal command
        # print(f"Sent: {cmd}")  

        if cmd[0:8]== "DATA:DAC":
            time.sleep(0.01)

        
        if "?" in cmd:  # If it's a query, wait for a response
            response = ser.readline().decode().strip()
            print(f"Response: {response}")
            time.sleep(0.1)  # Small delay to avoid overloading the buffer
        else:
            time.sleep(0.05)  # Short delay for non-query commands
            
def message_toBitArray(message: str): #ON OFF KEYING
    message_binary = ''.join(format(ord(i), '08b') for i in message)
    # print(f"Message in binary: {message_binary}")
    # TODO: determine the exact number of samples per bit that makes sense in relation to our sample rate 
    # and bits per second and also how this is done in the signal generator
    
    square_wave = []
    for bit in message_binary:
        if bit == '0':
            square_wave += [0]
        elif bit == '1':
            square_wave += [1] 

    return square_wave
    
def stopTransmission():
    command = f"""
    OUTPut OFF
    """
    send_command(command)

def transmitPhysical(message, carrier, bitrate):    
    
    if HAMMING_CODING:
        square_wave = []
        bits = hamming_encode(message)
        for bit in bits:
            if bit == '0':
                square_wave += [0]
            elif bit == '1':
                square_wave += [1] 
        bits =  square_wave
    else:
        bits = message_toBitArray(message)
   

    if CONVOLUTIONAL_CODING:
        bits = conv_encode(bits)

    if (APPLY_BAKER_PREAMBLE):
        bits = BINARY_BARKER + bits
    else:
        bits = PREAMBLE_PATTERN + bits

    #change bits to -1 for signal generator scipy commands
    for i in range(0, len(bits),1):
        if bits[i] == 0:
            bits[i]= -1

    bits = np.array(bits)
    arb_wave_form_command = "DATA:DAC VOLATILE, " + ", ".join(map(str, bits * 2047))
    
    freq = bitrate/len(bits)
    global len_of_bits 
    len_of_bits = len(bits)
    print("Transmitted bits: ", len_of_bits)
    

    name = "COCK"
    command = f"""
    {arb_wave_form_command}
    DATA:COPY {name}
    FUNC:USER {name}
    FUNC USER
    """
    send_command(command)

    time.sleep(0.5)

    command = f"""
    APPLy:SIN {carrier}, 10, 0
    AM:SOUR INT
    AM:INTernal:FUNCtion USER
    AM:INT:FREQuency {freq}
    AM:DEPT 120
    AM:STAT ON
    """

    send_command(command)
