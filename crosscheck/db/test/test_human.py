from ..human import encode, decode, add_round
import pytest
import pathlib
import datetime
from crosscheck import definitions

@pytest.fixture(name="basic_controls")
def _basic_controls():
    return [
        {"UP": True, "DOWN": True, "LEFT": True, "RIGHT": True, "A": True, "B": True, "C": True},
        {"UP": False, "DOWN": False, "LEFT": False, "RIGHT": True, "A": False, "B": False, "C": False},
        {"UP": True, "DOWN": True, "LEFT": False, "RIGHT": False, "A": False, "B": True, "C": False},
        {"UP": False, "DOWN": True, "LEFT": True, "RIGHT": True, "A": False, "B": True, "C": True},
        {"UP": True, "DOWN": True, "LEFT": True, "RIGHT": False, "A": True, "B": False, "C": False},
        {"UP": False, "DOWN": False, "LEFT": False, "RIGHT": False, "A": False, "B": False, "C": False},
    ]

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


def test_basic(basic_controls):
    """
    Test a handful of basic situations, encode and decode
    """
    # Arrange
    orig = basic_controls

    # Act
    actual_e = encode(orig)
    actual_d = decode(actual_e)

    # Assert
    assert actual_e == b'\x7F\x08\x62\x3B\x74\x00'
    assert actual_d == orig


def test_db_write(basic_controls, tmpdir):
    # Arrange
    db_filename = pathlib.Path(str(tmpdir / "tmp.db"))
    sample_scenario = definitions.SAVE_STATE_FOLDER / 'foo' / 'my_scenario.state'
    sample_time = datetime.datetime(year=2020, month=2, day=20, hour=21, minute=32, second=29)

    # Act
    add_round(sample_scenario, False, basic_controls, sample_time, db_filename)
    add_round(sample_scenario, False, basic_controls, sample_time, db_filename)

    # Assert
    # Success if it doesn't blow up
    pass