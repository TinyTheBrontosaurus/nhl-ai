
def get_player_w_puck(info):
    player_w_puck = {}

    puck_x = info['player-w-puck-ice-x']
    puck_y = info['player-w-puck-ice-y']

    if puck_x == info['player-home-7-x'] and puck_y == info['player-home-7-y']:
        player_w_puck['number'] = 7
        player_w_puck['team'] = 'home'
        player_w_puck['pos'] = 'RD'
    elif puck_x == info['player-home-10-x'] and puck_y == info['player-home-10-y']:
        player_w_puck['number'] = 10
        player_w_puck['team'] = 'home'
        player_w_puck['pos'] = 'LW'
    elif puck_x == info['player-home-16-x'] and puck_y == info['player-home-16-y']:
        player_w_puck['number'] = 16
        player_w_puck['team'] = 'home'
        player_w_puck['pos'] = 'C'
    elif puck_x == info['player-home-89-x'] and puck_y == info['player-home-89-y']:
        player_w_puck['number'] = 89
        player_w_puck['team'] = 'home'
        player_w_puck['pos'] = 'RW'
    elif puck_x == info['player-home-8-x'] and puck_y == info['player-home-8-y']:
        player_w_puck['number'] = 8
        player_w_puck['team'] = 'home'
        player_w_puck['pos'] = 'LD'
    elif puck_x == info['player-home-goalie-ice-x'] and puck_y == info['player-home-goalie-ice-y']:
        player_w_puck['number'] = 31
        player_w_puck['team'] = 'home'
        player_w_puck['pos'] = 'G'

    return player_w_puck
