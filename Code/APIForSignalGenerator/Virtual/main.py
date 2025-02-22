import serial
import time

from Virtual.communicate import initPort, testConnectionVirtual, testConnectionPhysical, communicateVirtual, \
    communicatePhysical
from Virtual.createSquareWave import string_to_binary
from display_menu import display_menu, startMenu
import Ports

def convert_to_scientific_notation_from_string(number_str):
    # Convert the string to a float and format it to scientific notation
    try:
        number = float(number_str)
        return f"{number:.1E}"
    except ValueError:
        return "Invalid input"


def main():
    print("\n--- Starting Program ---\n")


    answer = input("Physical (1) or virtual cables (2): ")

    if answer == "1":
        startPhysicalMain()
    else:
        startVirtualMain()




def startVirtualMain():
    print("\nInitializing serial ports...\n")
    # change this to inputs
    # portWrite = input("Enter write port: ")
    portWrite = 'COM3'
    Ports.ser_write = initPort(portWrite)

    # portWrite = input("Enter read port: ")
    portRead = 'COM4'
    Ports.ser_read = initPort(portRead)

    while True:
        print("\n---------------\n")
        choice = display_menu(startMenu)
        print("\n---------------\n")

        if choice == "Test connection":
            testConnectionVirtual()

        elif choice == "Change frequency of carrier wave":
            number = input("Enter the frequency: ")
            formatted_number = convert_to_scientific_notation_from_string(number)
            if formatted_number != "Invalid input":
                print(f"Formatted frequency: {formatted_number}")
                communicateVirtual(f"APPL:SIN {formatted_number}, 3.0, -2.5")
            else:
                print("Invalid frequency input.\n")

        # elif choice == "Modulate carrier wave with file"
        else:
            break;

    # Close both ports
    Ports.ser_write.close()
    Ports.ser_read.close()


def startPhysicalMain():

    print("\nInitializing serial ports...\n")
    # change this to inputs
    # portWrite = input("Enter port: ")
    portWrite = 'COM3'
    Ports.ser = initPort(portWrite)

    while True:
        print("\n---------------\n")
        choice = display_menu(startMenu)
        print("\n---------------\n")

        if choice == "Test connection":
            testConnectionPhysical()

        elif choice == "Change frequency of carrier wave":
            number = input("Enter the frequency: ")
            formatted_number = convert_to_scientific_notation_from_string(number)
            if formatted_number != "Invalid input":
                print(f"Formatted frequency: {formatted_number}")
                communicatePhysical(f"APPL:SIN {formatted_number}, 3.0, -2.5")
            else:
                print("Invalid frequency input.\n")

        # elif choice == "Modulate carrier wave with file"
        else:
            break;

    # Close both ports
    Ports.ser.close()

if __name__ == "__main__":
    main()


