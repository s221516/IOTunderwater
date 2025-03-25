import threading
from record_audio import continuous_recording_with_preamble_detection


def main():
    recording_thread = threading.Thread(target=continuous_recording_with_preamble_detection)
    recording_thread.daemon = True
    recording_thread.start()
    
    try:
        # Keep main thread alive
        while True:
            input()  # Wait for keyboard interrupt
    except KeyboardInterrupt:
        print("Exiting chat receiver...")

if __name__ == "__main__":
    main()