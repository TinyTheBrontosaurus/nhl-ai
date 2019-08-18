
import pytest
import pytest_steps
from training.info_utils import InfoWrapper, InfoAccumulator, TEAMS, POSITIONS, DIMS, TIME_PER_FRAME


@pytest.fixture(name='info')
def _info():
    info = {}
    for teami, team in enumerate(TEAMS):
        for positioni, position in enumerate(POSITIONS):
            for dimi, dim in enumerate(DIMS):
                info['player-{}-{}-{}'.format(team, position, dim)] = (teami + 1) * 100 + \
                                                                      (positioni + 1) * 10 + \
                                                                      dimi + 1

    for dim in DIMS:
        info['player-w-puck-ice-{}'.format(dim)] = 0
        info['puck-ice-{}'.format(dim)] = 0

    return info

@pytest.mark.parametrize('team,pos', [(team, pos) for team in TEAMS for pos in POSITIONS])
def test_player_has_puck(info, team, pos):
    """
    Test player_w_puck can detect each player
    """
    # Arrange
    object_under_test = InfoWrapper()
    object_under_test.info = info

    info['player-{}-{}-x'.format(team, pos)] = 100
    info['player-{}-{}-y'.format(team, pos)] = 200
    info['player-w-puck-ice-x'] = 100
    info['player-w-puck-ice-y'] = 200

    # Act
    actual = object_under_test.player_w_puck

    # Assert
    assert actual == {'team': team, 'pos': pos}


def test_no_possession(info):
    """
    Test player_w_puck an unpossessed puck
    """
    # Arrange
    object_under_test = InfoWrapper()
    object_under_test.info = info

    info['player-w-puck-ice-x'] = 100
    info['player-w-puck-ice-y'] = 200

    # Act
    actual = object_under_test.player_w_puck

    # Assert
    assert actual == {}


def test_feature(info):
    """
    Snapshot on feature vector of all player positions
    """

    # Arrange
    object_under_test = InfoWrapper()
    object_under_test.info = info

    for index, key in enumerate(info.keys()):
        info[key] = (index + 1) * 10

    # Act
    actual = object_under_test.players_and_puck_feature

    # Assert
    # Note: This is basically a snapshot
    expected = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                210, 220, 230, 240, 250, 270, 260, 280, 1]
    assert actual == expected


@pytest_steps.test_steps('ctor', 'first_possession', 'first_pass_direct', 'held', 'in_transit', 'received')
def test_accumulator(info):

    def move_puck_to(team, pos):
        info['player-w-puck-ice-x'] = info['player-{}-{}-x'.format(team, pos)]
        info['player-w-puck-ice-y'] = info['player-{}-{}-y'.format(team, pos)]

    def clear_puck():
        info['player-w-puck-ice-x'] = 0
        info['player-w-puck-ice-y'] = 0

    # Arrange
    object_under_test = InfoAccumulator()
    object_under_test.info = info

    # Act
    object_under_test.accumulate()

    # Assert
    assert object_under_test.pass_attempts == {'home': 0, 'away': 0}
    assert object_under_test.pass_count == {'home': 0, 'away': 0}
    assert object_under_test.steal_count == {'home': 0, 'away': 0}

    assert object_under_test.time_puck == {'home': pytest.approx(0.0),
                                           None: pytest.approx(TIME_PER_FRAME),
                                           'away': pytest.approx(0.0)}
    assert object_under_test.consecutive_passes == {'consecutive': {'home': [0], 'away': [0]},
                                                    'unique':  {'home': [0], 'away': [0]},}
    yield


    # Arrange
    move_puck_to('home', 'LW')

    # Act
    object_under_test.accumulate()

    # Assert
    assert object_under_test.pass_attempts == {'home': 0, 'away': 0}
    assert object_under_test.pass_count == {'home': 0, 'away': 0}
    assert object_under_test.steal_count == {'home': 0, 'away': 0}

    assert object_under_test.time_puck == {'home': pytest.approx(TIME_PER_FRAME * 1),
                                           None: pytest.approx(TIME_PER_FRAME * 1),
                                           'away': pytest.approx(0.0)}
    assert object_under_test.consecutive_passes == {'consecutive': {'home': [0], 'away': [0]},
                                                    'unique':  {'home': [0], 'away': [0]},}
    yield

    # Arrange
    move_puck_to('home', 'RW')

    # Act
    object_under_test.accumulate()

    # Assert
    assert object_under_test.pass_attempts == {'home': 1, 'away': 0}
    assert object_under_test.pass_count == {'home': 1, 'away': 0}
    assert object_under_test.steal_count == {'home': 0, 'away': 0}

    assert object_under_test.time_puck == {'home': pytest.approx(TIME_PER_FRAME * 2),
                                           None: pytest.approx(TIME_PER_FRAME * 1),
                                           'away': pytest.approx(0.0)}
    assert object_under_test.consecutive_passes == {'consecutive': {'home': [1], 'away': [0]},
                                                    'unique':  {'home': [1], 'away': [0]},}
    yield

    # Arrange

    # Act
    object_under_test.accumulate()

    # Assert
    assert object_under_test.pass_attempts == {'home': 1, 'away': 0}
    assert object_under_test.pass_count == {'home': 1, 'away': 0}
    assert object_under_test.steal_count == {'home': 0, 'away': 0}

    assert object_under_test.time_puck == {'home': pytest.approx(TIME_PER_FRAME * 3),
                                           None: pytest.approx(TIME_PER_FRAME * 1),
                                           'away': pytest.approx(0.0)}
    assert object_under_test.consecutive_passes == {'consecutive': {'home': [1], 'away': [0]},
                                                    'unique': {'home': [1], 'away': [0]}, }
    yield

    # In transit
    # Arrange
    clear_puck()

    # Act
    object_under_test.accumulate()

    # Assert
    assert object_under_test.pass_attempts == {'home': 2, 'away': 0}
    assert object_under_test.pass_count == {'home': 1, 'away': 0}
    assert object_under_test.steal_count == {'home': 0, 'away': 0}

    assert object_under_test.time_puck == {'home': pytest.approx(TIME_PER_FRAME * 3),
                                           None: pytest.approx(TIME_PER_FRAME * 2),
                                           'away': pytest.approx(0.0)}
    assert object_under_test.consecutive_passes == {'consecutive': {'home': [1], 'away': [0]},
                                                    'unique': {'home': [1], 'away': [0]}, }
    yield

    # Received
    # Arrange
    move_puck_to('home', 'LW')

    # Act
    object_under_test.accumulate()

    # Assert
    assert object_under_test.pass_attempts == {'home': 2, 'away': 0}
    assert object_under_test.pass_count == {'home': 2, 'away': 0}
    assert object_under_test.steal_count == {'home': 0, 'away': 0}

    assert object_under_test.time_puck == {'home': pytest.approx(TIME_PER_FRAME * 4),
                                           None: pytest.approx(TIME_PER_FRAME * 2),
                                           'away': pytest.approx(0.0)}
    assert object_under_test.consecutive_passes == {'consecutive': {'home': [2], 'away': [0]},
                                                    'unique': {'home': [1], 'away': [0]}, }
    yield

