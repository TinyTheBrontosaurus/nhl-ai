
def get_player_w_puck(info):
    player_w_puck = {}

    puck_x = info['player-w-puck-ice-x']
    puck_y = info['player-w-puck-ice-y']

    if puck_x == info['player-home-RD-x'] and puck_y == info['player-home-RD-y']:
        player_w_puck['number'] = 7
        player_w_puck['team'] = 'home'
        player_w_puck['pos'] = 'RD'
    elif puck_x == info['player-home-LW-x'] and puck_y == info['player-home-LW-y']:
        player_w_puck['number'] = 10
        player_w_puck['team'] = 'home'
        player_w_puck['pos'] = 'LW'
    elif puck_x == info['player-home-C-x'] and puck_y == info['player-home-C-y']:
        player_w_puck['number'] = 16
        player_w_puck['team'] = 'home'
        player_w_puck['pos'] = 'C'
    elif puck_x == info['player-home-RW-x'] and puck_y == info['player-home-RW-y']:
        player_w_puck['number'] = 89
        player_w_puck['team'] = 'home'
        player_w_puck['pos'] = 'RW'
    elif puck_x == info['player-home-LD-x'] and puck_y == info['player-home-LD-y']:
        player_w_puck['number'] = 8
        player_w_puck['team'] = 'home'
        player_w_puck['pos'] = 'LD'
    elif puck_x == info['player-home-G-x'] and puck_y == info['player-home-G-y']:
        player_w_puck['number'] = 31
        player_w_puck['team'] = 'home'
        player_w_puck['pos'] = 'G'

    return player_w_puck
