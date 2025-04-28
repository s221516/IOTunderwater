from Code.dsp.receiver.audio_receiver import AudioReceiver
from Code.dsp.Transmitter import Transmitter
import config


def run_chat(shared_state):
    print("Enter messages to transmit or commands:")
    print("  'exit' to quit")
    print(f"  'cf=<freq>' to set carrier frequency (e.g., 'cf={config.CARRIER_FREQ}')")
    print(f"  'br=<rate>' to set bit rate (e.g., 'br={config.BIT_RATE}') \n")

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
                # if len(user_input) < max_len:
                #     user_input = user_input.ljust(max_len, "_")
                if len(user_input) > max_len:
                    user_input = user_input[:max_len]
                shared_state["msg"] = user_input

    except KeyboardInterrupt:
        print("\nExiting interactive mode...")

def main():
    shared_state = {"is_transmitting": False, "msg": None}
    receiver = AudioReceiver(shared_state)
    # Create sender
    sender = Transmitter(shared_state)
    
    # Start the threads
    receiver.start()
    sender.start()

    run_chat(shared_state)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
