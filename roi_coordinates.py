'''
This dictionary describes the pixel coordinates of specific info that needs to be extracted.
It is specific to screenshots from a given monitor resolution
This config is from my 2560x1440 display.
To configure for a given machine, one must find the coordinates [y start, y end, x start, x end] for each text box
This only has to be done once.
An example can be found in
'''

stats_extract = {
    'match_type': {
        # y start, y end, x start, x end
        'roi': [111, 184, 850, 1681],
        'img_id': 'mission_sum_scrnshot'
    },
    'match_timer_secs': {
        'roi': [860, 900, 1840, 1950],
        'img_id': 'last_match_scrnshot'
    },
    'cash_xp': {
        'roi': [940, 1000, 1980, 2200],
        'img_id': 'mission_sum_scrnshot'
    },
    'my_kills': {
        'roi': [700, 770, 1230, 1330],
        'img_id': 'last_match_scrnshot'
    },
    'assists': {
        'roi': [1135, 1220, 1550, 1650],
        'img_id': 'last_match_scrnshot'
    },
    'team_kills': {
        'roi': [1140, 1210, 900, 1000],
        'img_id': 'last_match_scrnshot'
    },
    'hunter_name': {
        'roi': [370, 430, 900, 1800],
        'img_id': 'last_match_scrnshot'
    },
    'partner_1': {
        'roi': [400, 450, 44, 430],
        'img_id': 'last_match_scrnshot'
    },
    'partner_2': {
        'roi': [400, 450, 2100, 2540],
        'img_id': 'last_match_scrnshot'
    }
}

# Convert pixel coordinates to relative position on the screen. Hopefully this would work for different resolutions..