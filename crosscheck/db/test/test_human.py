from ..human import encode, decode


def test_empty():
    """
    Test empty set
    """
    # Arrange
    orig = []

    # Act
    actual_e = encode(orig)
    actual_d = decode(actual_e)

    # Assert
    assert actual_e == b''
    assert actual_d == orig


def test_basic():
    """
    Test a handful of basic situations, encode and decode
    """
    # Arrange
    orig = [
        {"UP": True, "DOWN": True, "LEFT": True, "RIGHT": True, "A": True, "B": True, "C": True},
        {"UP": False, "DOWN": False, "LEFT": False, "RIGHT": True, "A": False, "B": False, "C": False},
        {"UP": True, "DOWN": True, "LEFT": False, "RIGHT": False, "A": False, "B": True, "C": False},
        {"UP": False, "DOWN": True, "LEFT": True, "RIGHT": True, "A": False, "B": True, "C": True},
        {"UP": True, "DOWN": True, "LEFT": True, "RIGHT": False, "A": True, "B": False, "C": False},
        {"UP": False, "DOWN": False, "LEFT": False, "RIGHT": False, "A": False, "B": False, "C": False},
    ]

    # Act
    actual_e = encode(orig)
    actual_d = decode(actual_e)

    # Assert
    assert actual_e == b'\x7F\x08\x62\x3B\x74\x00'
    assert actual_d == orig
