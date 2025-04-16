from concurrent.futures import thread
from email.mime import audio
from multiprocessing import process
from pdb import run
from re import M
import threading
import time
from audio_reciever import AudioReceiver
from interactive_sender import MessageSender
from main import process_signal
import config

# Configuration variables - modify these as needed
OPERATION_MODE = "both"  # Options: "receive", "send", "both"
USE_ESP = True  # True for ESP32, False for signal generator
MIC_INDEX = 1  # Audio input device index
global is_transmitting
is_transmitting = False


def run_chat(shared_state):
    print("Enter messages to transmit or commands:")
    print("  'exit' to quit")
    print("  'cf=<freq>' to set carrier frequency (e.g., 'cf=6000')")
    print("  'br=<rate>' to set bit rate (e.g., 'br=100') \n")

    try:
        print("Write a message below!")
        while True:

            user_input = input("You: ")

            if user_input.lower() == "exit":
                break

            elif user_input.lower().startswith("cf="):
                try:
                    freq = int(user_input[3:])
                    config.CARRIER_FREQ = freq
                    print(f"Carrier frequency set to {freq} Hz")
                except ValueError:
                    print("Invalid frequency format. Use 'cf=6000' format.")

            elif user_input.lower().startswith("br="):
                try:
                    rate = int(user_input[3:])
                    config.BIT_RATE = rate
                    print(f"Bit rate set to {rate} bps")
                except ValueError:
                    print("Invalid bit rate format. Use 'br=100' format.")
            elif user_input.lower().startswith("rep="):
                try:
                    reps = int(user_input[4:])
                    config.REP_ESP = reps
                    print(f"Repetitions set to {reps}")
                except ValueError:
                    print("Invalid repetitions format. Use 'rep=3' format.")

            else:
                # Transmit the message
                ## replace spaces with underscores
                user_input = user_input.replace(" ", "_")
                ## Essure user input is always 32 characters long if less than append '_'
                max_len = 32
                if len(user_input) < max_len:
                    user_input = user_input.ljust(max_len, "_")
                elif len(user_input) > max_len:
                    user_input = user_input[:max_len]
                shared_state["msg"] = user_input

    except KeyboardInterrupt:
        print("\nExiting interactive mode...")


def main():
    shared_state = {"is_transmitting": False, "msg": None}
    receiver = AudioReceiver(shared_state)
    # Create sender
    sender = MessageSender(shared_state, use_esp=USE_ESP)
    # Start the threads
    receiver.start()
    sender.start()

    run_chat(shared_state)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
