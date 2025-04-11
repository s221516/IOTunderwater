import serial
import time

# Set up the UART connection (adjust 'COM5' as needed)
ser = serial.Serial('COM5', 115200, timeout=1)


def read_line():
    """Read bytes until a newline is encountered or timeout occurs."""
    line = b""
    while True:
        byte = ser.read(1)
        if byte:
            line += byte
            if byte in b'\r\n':  # end when a newline or CR is received
                break
    return line.decode('utf-8', errors='ignore')

def send_command(command):
    ser.write((command + "\r\n").encode())

    time.sleep(0.5)  # Give it a moment for the ESP32 to process and respond
    
    response = read_line()
    #RESPONSE IS NONSENCE RIGHT NOW 

def main():

    print("""
Available commands:
  FREQ <value>     - Set the carrier frequency (default is 6000)
  BITRATE <value>  - Set the bitrate (default is 100)
  REP <value>      - Set wave repetitions (default is 10)
  <text>           - Send a text message to transmit
""")
    while True:
    
        command = input("Enter command: ")
        if command.strip().upper() == "EXIT":
            print("Exiting...")
            break
        send_command(command)

if __name__ == "__main__":
    main()
