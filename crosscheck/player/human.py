import inputs


lookup = {
    'ABS_Y': {
        0: {"Up": "1"},
        127: {"Up": "0"},
        255: {"Up": "-1"}
    },
    'ABS_X': {
        0: {"Left": "1"},
        127: {"Left": "0"},
        255: {"Left": "-1"}
    },
    'BTN_TRIGGER': {
        1: {"A": 1},
        0: {"A": 0},
    },
    'BTN_THUMB': {
        1: {"B": 1},
        0: {"B": 0},
    },
    'BTN_THUMB2': {
        1: {"C": 1},
        0: {"C": 0},
    },
    'BTN_BASE4': {
        1: {"Start": 1},
        0: {"Start": 0},
    },
    'BTN_TOP': {
        1: {"X": 1},
        0: {"X": 0},
    },
    'BTN_TOP2': {
        1: {"Y": 1},
        0: {"Y": 0},
    },
    'BTN_PINKIE': {
        1: {"Z": 1},
        0: {"Z": 0},
    },
}

while 1:
    events = inputs.get_gamepad()
    for event in events:
        button = lookup.get(event.code, {}).get(event.state)

        if button:
            print(button)
