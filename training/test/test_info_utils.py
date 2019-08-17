
import pytest
from training.info_utils import InfoWrapper, InfoAccumulator

TEAMS = ('home', 'away')
POSITIONS = ('LW', 'C', 'RW', 'LD', 'RD', 'G')
DIMS = ('x', 'y')

@pytest.fixture(name='info')
def _info():
    info = {}
    for team in TEAMS:
        for position in POSITIONS:
            for dim in DIMS:
                info['player-{}-{}-{}'.format(team, position, dim)] = 0

    for dim in DIMS:
        info['player-w-puck-ice-{}'.format(dim)] = 0

    return info

@pytest.mark.parametrize('team,pos', [(team, pos) for team in TEAMS for pos in POSITIONS])
def test_player_has_puck(info, team, pos):
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
    # Arrange
    object_under_test = InfoWrapper()
    object_under_test.info = info

    info['player-w-puck-ice-x'] = 100
    info['player-w-puck-ice-y'] = 200

    # Act
    actual = object_under_test.player_w_puck

    # Assert
    assert actual == {}

