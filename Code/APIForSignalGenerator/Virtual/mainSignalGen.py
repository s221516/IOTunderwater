import serial
import time
from initPorts import initPort
from transmitter import create_modulated_wave

MESSAGE = "Hello, World!"


ser = initPort('COM3')
def write(command):
    ser.write(command.encode('utf-8'))
    
    #also read response:)
    response = ser.read(ser.in_waiting).decode('utf-8')
    print(f"Received: {response.strip()}\n")
    
def main():

    #take transmitter wave 

    modulated_wave = create_modulated_wave(MESSAGE)
    print(modulated_wave)
    #convert to values between 2047 and -2047
    #convert to command DATA:DAC VOLATILE
    #store command in volatile memory
    #

    command = "" \
    "APPly:USER 1, 10, 0 \r\n" \
    "DATA:DAC VOLATILE, 2047,-2047 \r\n" 
    "SYStem:ERRor?\r\n"
    write(command)


    command = "" \
    "APPly:USER 1, 10, 0 \r\n" \
    "DATA:DAC VOLATILE, 2047,-2047 \r\n" 
    write(command)


    command = "" \
    

if __name__ == "__main__":
    main()