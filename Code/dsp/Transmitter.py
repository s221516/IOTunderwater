import time
import config
import threading
import serial
from utilityFunctions import message_toBitArray
import numpy as np
from config import (
    BINARY_BARKER,
    CONVOLUTIONAL_CODING,
    HAMMING_CODING
)
from encoding.hamming_codes import hamming_encode
from encoding.conv_encoding_scikit import conv_encode


class Transmitter():
    def __init__(self, isESP32):
        self.isESP32 = isESP32
        self._initPorts()

    def transmit(self, message, carrierfreq, bitrate):
        if self.isESP32:
            self._send_command("FREQ" + str(carrierfreq))
            self._send_command("BITRATE" + str(bitrate))
            self._send_command("REP" + str(config.REP_ESP))
            # message last so we put the specs of the wave first
            self._send_command(message)
        
        else:
            if HAMMING_CODING:
                square_wave = []
                bits = hamming_encode(message)
                for bit in bits:
                    if bit == "0":
                        square_wave += [0]
                    elif bit == "1":
                        square_wave += [1]
                bits = square_wave
            else:
                bits = message_toBitArray(message)

            if CONVOLUTIONAL_CODING:
                bits = conv_encode(bits)
                bits = bits.tolist()

            bits = BINARY_BARKER + bits

            for i in range(0, len(bits), 1):
                if bits[i] == 0:
                    bits[i] = -1

            bits = np.array(bits)
            arb_wave_form_command = "DATA:DAC VOLATILE, " + ", ".join(map(str, bits * 2047))

            len_of_bits = len(bits)

            freq = bitrate / len_of_bits

            name = "ARB1"
            command = f"""
            {arb_wave_form_command}
            DATA:COPY {name}
            FUNC:USER {name}
            FUNC USER
            """
            self._send_command(command)

            time.sleep(0.2)

            command = f"""
            APPLy:SIN {carrierfreq}, 3.3, 0
            AM:SOUR INT
            AM:INTernal:FUNCtion USER
            AM:INT:FREQuency {freq}
            AM:DEPT 120
            AM:STAT ON
            """
            self._send_command(command)

    def _send_command(self, command):

        if self.isESP32:
            self.ser.write((command + "\r\n").encode())
            time.sleep(0.3)

        else:
            commands = command.strip().split("\n")

            for cmd in commands:
                cmd = cmd.strip()
                if not cmd:
                    continue

                self.ser.write((cmd + "\r\n").encode())  # Send normal command
                # print(f"Sent: {cmd}")

                if cmd[0:8] == "DATA:DAC":
                    time.sleep(0.01)

                if "?" in cmd:  # If it's a query, wait for a response
                    response = self.ser.readline().decode().strip()
                    print(f"Response: {response}")
                    time.sleep(0.1)  # Small delay to avoid overloading the buffer
                else:
                    time.sleep(0.05)  # Short delay for non-query commands

    def stopTransmission(self):
        if self.isESP32:
            pass   
        else:
            command = """
            OUTPut OFF
            """
            self._send_command(command)

    def _initPorts(self):
        if self.isESP32:
            self.ser = serial.Serial(config.TRANSMITTER_PORT, 115200, timeout=1)
        else:
            self.ser = self.initPort(config.TRANSMITTER_PORT)

    def initPort(self, portName):
        try:
            # Initialize serial port
            ser = serial.Serial(
                port=portName,  
                baudrate=57600,  # Default baudrate
                bytesize=serial.EIGHTBITS,  # 8 bits per byte
                parity=serial.PARITY_NONE,  # No parity
                stopbits=serial.STOPBITS_ONE,  # 1 stop bit
                timeout=1,  # 1 second timeout for read/write operations
                dsrdtr=True,  # Enable DSR/DTR hardware handshaking
                rtscts=False,  # Disable RTS/CTS flow control
                xonxoff=False  # Disable software flow control (XON/XOFF)
            )
            
            # Check if the port was opened successfully
            if ser.is_open:
                # print(f"Port {portName} opened successfully.")
                pass
            else:
                print(f"Failed to open {portName}.")
                exit()

            # Return the serial connection object
            return ser

        except serial.SerialException as e:
            # Exception handling for serial port issues
            print(f"Failed to open port {portName}: {e}")
            exit()

        except Exception as e:
            # Catch any other exception not related to serial issues
            print(f"An unexpected error occurred: {e}")
            exit()
