import inputs
import threading


# Lookup table that converts a Hyperkin 6-button genesis controller's
# inputs from the `inputs` package to a dictionary
lookup = {
    'ABS_Y': {
        0: {"UP": True, "DOWN": False},
        127: {"UP": False, "DOWN": False},
        255: {"UP": False, "DOWN": True}
    },
    'ABS_X': {
        0: {"LEFT": True, "RIGHT": False},
        127: {"LEFT": False, "RIGHT": False},
        255: {"LEFT": False, "RIGHT": True}
    },
    'BTN_TRIGGER': {
        1: {"A": True},
        0: {"A": False},
    },
    'BTN_THUMB': {
        1: {"B": True},
        0: {"B": False},
    },
    'BTN_THUMB2': {
        1: {"C": True},
        0: {"C": False},
    },
    'BTN_BASE4': {
        1: {"START": True},
        0: {"START": False},
    },
    'BTN_TOP': {
        1: {"X": True},
        0: {"X": False},
    },
    'BTN_TOP2': {
        1: {"Y": True},
        0: {"Y": False},
    },
    'BTN_PINKIE': {
        1: {"Z": True},
        0: {"Z": False},
    },
}


class ButtonState:
    def __init__(self):
        """
        Create an object that keeps track of button state across threads.
        """
        # The button state, as a dictionary
        self.state = {'LEFT': False, "RIGHT": False, 'UP': False, 'DOWN': False,
                      'A': False, 'B': False, 'C': False,
                      'X': False, 'Y': False, 'Z': False,
                      'START': False}
        # The lock to use when looking at button state
        self.lock = threading.Lock()
        # Set to false to kill the background thread gracefully
        self.running = True
        # Comms between two threads indicating when a new button press is in.
        # Background thread will SET it, and it's the foreground thread's job
        # to clear as needed.
        self.event = threading.Event()

def maintain_button_state(button_state: ButtonState):
    """
    Entry point for background thread that monitors state. Does not return until button_state.running set to False
    :param button_state: The shared button state between background thread and foreground thread
    """
    while button_state.running:

        # Note: This blocks until a button is pressed or released. So the outer while look will hang until
        # a button is pressed
        events = inputs.get_gamepad()

        # Grab the mutex for all current events
        with button_state.lock:
            for event in events:
                button = lookup.get(event.code, {}).get(event.state)
                if button:
                    button_state.state.update(button)
                    button_state.event.set()


def _print_button_state_changes():
    """
    Example of monitoring button presses using another thread
    """
    # Comms between this thread and that thread
    button_state = ButtonState()

    # Create and start thread
    button_thread = threading.Thread(target=maintain_button_state, args=(button_state,))
    button_thread.start()

    while 1:
        # Wait for a button change
        button_state.event.wait()

        # Grab the mutex, clear the button change event, and print
        with button_state.lock:
            button_state.event.clear()
            print(button_state.state)

if __name__ == "__main__":
    _print_button_state_changes()
