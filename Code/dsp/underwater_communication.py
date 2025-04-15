import threading
import time
from audio_reciever import monitor_audio, list_audio_devices
from interactive_sender import MessageSender

# Configuration variables - modify these as needed
OPERATION_MODE = "both"    # Options: "receive", "send", "both"
USE_ESP = True            # True for ESP32, False for signal generator
THRESHOLD = 10          # Audio detection threshold
MIC_INDEX = 1            # Audio input device index

def run_receiver():
    """Run the receiver in a separate thread"""
    try:
        monitor_audio(THRESHOLD)
    except KeyboardInterrupt:
        print("\nReceiver stopped")

def run_sender(sender):
    """Run the sender in the main thread"""
    try:
        sender.run_interactive_mode()
    except KeyboardInterrupt: 
        print("\nSender stopped")

def main():
    if OPERATION_MODE in ["receive", "both"]:
        if OPERATION_MODE == "both":
            # Run receiver in a separate thread
            receiver_thread = threading.Thread(target=run_receiver)
            receiver_thread.daemon = True
            receiver_thread.start()
        else:
            # Run receiver in main thread
            run_receiver()
            return

    if OPERATION_MODE in ["send", "both"]:
        sender = MessageSender(use_esp=USE_ESP)
        run_sender(sender)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")