import time
import config
import threading
import serial


import numpy as np
from config import (
    APPLY_BAKER_PREAMBLE,
    BINARY_BARKER,
    CONVOLUTIONAL_CODING,
    HAMMING_CODING
)

from encoding.hamming_codes import hamming_encode
from encoding.conv_encoding_scikit import conv_encode
from initPorts import initPort

class Transmitter(threading.Thread):
    def __init__(self, shared_state, isESP32):
        super().__init__(name="MessageSenderThread")
        
        self.isESP32 = isESP32
        self.shared_state = shared_state
        if self.isESP32:
            self.ser = serial.Serial(config.TRANSMITTER_PORT, 115200, timeout=1)
        else:
            self.ser = initPort(config.TRANSMITTER_PORT)

    def message_toBitArray(self, message: str):  # ON OFF KEYING
        message_binary = "".join(format(ord(i), "08b") for i in message)
        # print(f"Message in binary: {message_binary}")
        # TODO: determine the exact number of samples per bit that makes sense in relation to our sample rate
        # and bits per second and also how this is done in the signal generator

        square_wave = []
        for bit in message_binary:
            if bit == "0":
                square_wave += [0]
            elif bit == "1":
                square_wave += [1]

        return square_wave

    def send_command(self, command):

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

    def transmit(self, message, carrierfreq, bitrate):

        if self.isESP32:
            self.send_command("FREQ" + str(carrierfreq))
            self.send_command("BITRATE" + str(bitrate))
            self.send_command("REP" + str(config.REP_ESP))
            # message last so we put the specs of the wave first
            self.send_command(message)
        
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
                bits = self.message_toBitArray(message)

            if CONVOLUTIONAL_CODING:
                bits = conv_encode(bits)
                bits = bits.tolist()

            if APPLY_BAKER_PREAMBLE:
                bits = BINARY_BARKER + bits

            # change bits to -1 for signal generator scipy commands
            for i in range(0, len(bits), 1):
                if bits[i] == 0:
                    bits[i] = -1

            bits = np.array(bits)
            arb_wave_form_command = "DATA:DAC VOLATILE, " + ", ".join(map(str, bits * 2047))

            len_of_bits = len(bits)

            freq = bitrate / len_of_bits

            # print("Transmitted bits (transmitter): ", len_of_bits)

            name = "ARB1"
            command = f"""
            {arb_wave_form_command}
            DATA:COPY {name}
            FUNC:USER {name}
            FUNC USER
            """
            self.send_command(command)

            time.sleep(0.2)

            command = f"""
            APPLy:SIN {carrierfreq}, 3.3, 0
            AM:SOUR INT
            AM:INTernal:FUNCtion USER
            AM:INT:FREQuency {freq}
            AM:DEPT 120
            AM:STAT ON
            """
            self.send_command(command)

    def stopTransmission(self):

        if self.isESP32:
            pass
            
        else:
            command = """
            OUTPut OFF
            """
            self.send_command(command)

    def calculate_transmission_time(self, message):
        len_of_bits = len(message) * 8 + 13
        if self.isESP32:
            transmission_time = round((len_of_bits / config.BIT_RATE) * config.REP_ESP)
        else:
            transmission_time = round((len_of_bits / config.BIT_RATE) * config.REP_ESP)

        if transmission_time < 1:
            transmission_time = 1
            
        return transmission_time
    
    def chat_transmit(self, message):
        # For 1 computer setup and testing set it to False otherwise True
        self.shared_state["is_transmitting"] = False

        try:
            
            self.transmit(message, config.CARRIER_FREQ, config.BIT_RATE)
            
            transmission_time = self.calculate_transmission_time(message)
            
            # Wait for the transmission to complete
            time.sleep(transmission_time)

            #The signal generator will not stop transmitting until the next message is sent, so we need to stop it manually
            if not config.USE_ESP:
                self.stopTransmission()

        except Exception as e:
            print(f"Transmission error: {e}")
        finally:
            self.shared_state["is_transmitting"] = False
            # print("Transmitting: ", self.shared_state["is_transmitting"])

    def run(self):
        while True:
            message = self.shared_state["msg"]
            if message is not None:
                self.chat_transmit(self.shared_state["msg"])
                self.shared_state["msg"] = None
            time.sleep(1)
