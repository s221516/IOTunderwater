import serial
import time

from initPorts import initPort

def main():

    writer = initPort('COM8')
    reader = initPort('COM7')

    message = "wassap"

    writer.write(message.encode('utf-8'))

    time.sleep(1)

    if reader.in_waiting > 0:
        response = reader.read(reader.in_waiting).decode('utf-8')
        print(f"Received: {response.strip()}\n")
    else:
        print("No response received\n")

if __name__ == "__main__":
    main()