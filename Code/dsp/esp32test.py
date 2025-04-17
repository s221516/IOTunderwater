import serial
import time
import config

# Set up the UART connection (adjust 'COM5' as needed)

ser = serial.Serial("/dev/cu.usbserial-0232D158", 115200, timeout=1)
# ser = serial.Serial("COM12", 115200, timeout=1)


def read_line():
    """Read bytes until a newline is encountered or timeout occurs."""
    line = b""
    while True:
        byte = ser.read(1)
        if byte:
            line += byte
            if byte in b"\r\n":  # end when a newline or CR is received
                break
    return line.decode("utf-8", errors="ignore")


def send_command(command):
    ser.write((command + "\r\n").encode())

    time.sleep(0.3)  # Give it a moment for the ESP32 to process and respond

    response = read_line()
    # RESPONSE IS NONSENCE RIGHT NOW


def transmit_to_esp32(message, carrierfreq, bitrate):

    send_command("FREQ" + str(carrierfreq))
    send_command("BITRATE" + str(bitrate))
    send_command("REP" + str(config.REP_ESP))

    # message last so we put the specs of the wave first
    send_command(message)
