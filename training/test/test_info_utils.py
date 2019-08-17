

from training.info_utils import InfoWrapper, InfoAccumulator


def test_player_w_puck():
    # Arrange
    object_under_test = InfoWrapper()

    for team in ('home', 'away'):
        for position in ('LW', 'C', 'RW', 'LD', 'RD', 'G'):
            for dim in ('x', 'y'):
                object_under_test.info['player-{}-{}-{}'.format(team, position, dim)] = 0

    for dim in ('x', 'y'):
        object_under_test.info['player-w-puck-ice-{}'.format(dim)] = 0

    object_under_test.info['player-home-LW-x'] = 100
    object_under_test.info['player-home-LW-y'] = 200
    object_under_test.info['player-w-puck-ice-x'] = 100
    object_under_test.info['player-w-puck-ice-y'] = 200

    # Act
    actual = object_under_test.player_w_puck

    # Assert
    assert actual == {'team': 'home', 'pos': 'LW'}


