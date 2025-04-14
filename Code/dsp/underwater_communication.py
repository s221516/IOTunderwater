import threading
import time
import argparse
from audio_reciever import AudioReceiver
from interactive_sender import MessageSender


def run_receiver(receiver):
    """Run the receiver in a separate thread"""
    receiver.start_monitoring()


def run_sender(sender):
    """Run the sender in the main thread"""
    sender.run_interactive_mode()


def main():
    parser = argparse.ArgumentParser(description="Underwater Communication System")
    parser.add_argument(
        "--mode",
        choices=["receive", "send", "both"],
        default="both",
        help="Operation mode: receive, send, or both (default)",
    )
    parser.add_argument(
        "--esp",
        action="store_true",
        default=True,
        help="Use ESP32 for transmission (default)",
    )
    parser.add_argument(
        "--sg",
        action="store_false",
        dest="esp",
        help="Use signal generator for transmission",
    )

    args = parser.parse_args()

    if args.mode == "receive" or args.mode == "both":
        receiver = AudioReceiver()
        if args.mode == "both":
            # Run receiver in a separate thread
            receiver_thread = threading.Thread(target=run_receiver, args=(receiver,))
            receiver_thread.daemon = True
            receiver_thread.start()
        else:
            # Run receiver in main thread
            try:
                receiver.start_monitoring()
            except KeyboardInterrupt:
                pass
            finally:
                receiver.stop()
                receiver.p.terminate()
                return

    if args.mode == "send" or args.mode == "both":
        sender = MessageSender(use_esp=args.esp)
        try:
            run_sender(sender)
        except KeyboardInterrupt:
            pass

    # If we're in 'both' mode, make sure to clean up the receiver
    if args.mode == "both":
        receiver.stop()
        receiver.p.terminate()


if __name__ == "__main__":
    main()
