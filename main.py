import easyocr
import os
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from cv2 import imread
import numpy as np

def crop_matchtype_roi(reader, img_array):
    results = reader.readtext(img_array, text_threshold=0.3)
    return [results[x][1] for x in range(len(results))]

def imread_crop_dm(screenshot_png):
    '''Crop full res screenshot in case dual monitor is used.'''
    img_array = imread(str(screenshot_png))
    if img_array.shape == (3840, 4720, 3):
        img_array = img_array[1270:2709, 2160:]
    return img_array

def find_image_pairs(folder_path, time_threshold=30):
    image_pairs = []
    files = sorted(os.listdir(folder_path))
    for i in range(len(files)):
        for j in range(i+1, len(files)):
            image1_path = os.path.join(folder_path, files[i])
            image2_path = os.path.join(folder_path, files[j])
            image1_created = datetime.fromtimestamp(os.path.getmtime(image1_path))
            image2_created = datetime.fromtimestamp(os.path.getmtime(image2_path))
            time_difference = abs((image2_created - image1_created).total_seconds())
            if time_difference <= time_threshold:
                image_pairs.append((files[i], files[j]))
    return image_pairs

def rounded_minutes(time_str):
    # Split the time string into minutes and seconds
    split = '.' if '.' in time_str else ':'
    minutes, seconds = map(int, time_str.split(split))
    # Convert seconds to minutes and round to the nearest minute
    total_minutes = minutes + round(seconds / 60)
    return total_minutes


# Dict with info on how to extract details, which image to extract from, etc
stats_extract = {
    'matchtype': {
            # y start, y end, x start, x end
        'roi': [111, 184, 850,  1681],
        'img_id': 'mission_sum_scrnshot'
    },
    'matchtime': {
        'roi': [860, 900, 1840, 1950],
        'img_id': 'last_match_scrnshot'
    },
    'cash_xp': {
        'roi': [950, 995, 2000, 2200],
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
        'roi': [370, 430, 900, 1800 ],
        'img_id': 'last_match_scrnshot'
    }
}

dir_with_pairs = Path(r'C:\Users\giles\PycharmProjects\hunt_ocr_playtime_vs_xp\_images')
pairs = find_image_pairs(dir_with_pairs)
details = []
debug_rois = True
reader = easyocr.Reader(['en'], gpu=True)

for pair in tqdm(pairs):


    pairs_fp = [Path(dir_with_pairs) / x for x in pair]
    pair_id  = '_'.join([p.name[-9:-4] for p in pairs_fp])  # eg '(141)_(142)'
    im1, im2 = [imread_crop_dm(str(x)) for x in pairs_fp]

    # determine which image is which (match timer vs. match details)
    # mission details is #1, timer is #2
    im1b, im2b = [im[300:900, 0:200,:].mean() for im in [im1, im2]]
    # reverse order if images are backwards
    if im1b > im2b:
        im1, im2 = im2, im1

    # Get OCR raw data for each stat
    for stat, info in stats_extract.items():
        if info['img_id'] == 'last_match_scrnshot':
            img_arr = im2
        elif info['img_id'] == 'mission_sum_scrnshot':
            img_arr = im1

        roi = info['roi']
        img_cropped = img_arr[roi[0]:roi[1],roi[2]:roi[3],:]
        info['img_cropped'] = img_cropped
        ocr = crop_matchtype_roi(reader, img_cropped)
        info['ocr_raw'] = ocr

    # Process and save OCR data in useful format
    ocr_raw_dict = {stat: info['ocr_raw'] for stat, info in stats_extract.items()}
    processed = {
        'matchtype': 'Bounty Hunt' if ocr_raw_dict['matchtype'][0] == 'MISSION SUMMARY' else 'Soul Survivor',
        'matchtime': rounded_minutes(ocr_raw_dict['matchtime'][0]),
        'cash': int(ocr_raw_dict['cash_xp'][1].replace(',', '')),
        'xp': int(ocr_raw_dict['cash_xp'][0].replace(',', '')),
        'my_kills': int(ocr_raw_dict['my_kills'][0]) if (ocr_raw_dict['my_kills']) else 0,
        'assists': int(ocr_raw_dict['assists'][0]) if (ocr_raw_dict['assists']) else 0,
        'team_kills': int(ocr_raw_dict['team_kills'][0]) if (ocr_raw_dict['team_kills']) else 0,
        'hunter_name': ocr_raw_dict['hunter_name'][0]
    }
    details.append(processed)

    if debug_rois:
        debug_save_loc = Path('debug_imgs')
        if not debug_save_loc.exists():
            debug_save_loc.mkdir()

        fig, axs = plt.subplots(1, len(stats_extract.items()), figsize=(20, 5))
        for i, (stat, info) in enumerate(stats_extract.items()):
            label = f'{stat}: {ocr_raw_dict[stat]}'
            axs[i].imshow(info['img_cropped'])
            axs[i].set_title(label)
            # axs[i].set_title(stat)
            axs[i].set_axis_off()
        # Adjusting layout
        plt.tight_layout()
        # Display the plot
        plotname = f'{pair_id}_dbg.jpg'
        save_path = debug_save_loc / plotname
        plt.savefig(save_path)
        plt.close()

    # break
for d in details:
    print(d)

cache = [{'matchtype': 'Soul Survivor', 'matchtime': 13, 'cash': 640, 'xp': 4180, 'my_kills': 4, 'assists': 0, 'team_kills': 4, 'hunter_name': 'Miss Natalie Hall'}, {'matchtype': 'Bounty Hunt', 'matchtime': 16, 'cash': 300, 'xp': 2882, 'my_kills': 1, 'assists': 2, 'team_kills': 3, 'hunter_name': 'Kaiden Lockheart'}, {'matchtype': 'Bounty Hunt', 'matchtime': 6, 'cash': 1100, 'xp': 1562, 'my_kills': 0, 'assists': 0, 'team_kills': 0, 'hunter_name': 'The Prodigal Daughter'}, {'matchtype': 'Bounty Hunt', 'matchtime': 11, 'cash': 300, 'xp': 2816, 'my_kills': 0, 'assists': 1, 'team_kills': 2, 'hunter_name': 'The Rat'}, {'matchtype': 'Bounty Hunt', 'matchtime': 4, 'cash': 50, 'xp': 957, 'my_kills': 0, 'assists': 0, 'team_kills': 1, 'hunter_name': 'Deston Jacquet'}, {'matchtype': 'Bounty Hunt', 'matchtime': 18, 'cash': 742, 'xp': 4417, 'my_kills': 0, 'assists': 2, 'team_kills': 1, 'hunter_name': 'The Prodigal Daughter'}, {'matchtype': 'Bounty Hunt', 'matchtime': 22, 'cash': 2708, 'xp': 8892, 'my_kills': 2, 'assists': 2, 'team_kills': 7, 'hunter_name': 'Felis'}, {'matchtype': 'Bounty Hunt', 'matchtime': 23, 'cash': 807, 'xp': 5511, 'my_kills': 2, 'assists': 0, 'team_kills': 6, 'hunter_name': 'Darchelle Black'}, {'matchtype': 'Bounty Hunt', 'matchtime': 14, 'cash': 625, 'xp': 2981, 'my_kills': 2, 'assists': 0, 'team_kills': 2, 'hunter_name': 'Darchelle Black'}, {'matchtype': 'Bounty Hunt', 'matchtime': 36, 'cash': 1321, 'xp': 13742, 'my_kills': 4, 'assists': 1, 'team_kills': 9, 'hunter_name': 'Wilhelm Storch'}, {'matchtype': 'Bounty Hunt', 'matchtime': 14, 'cash': 500, 'xp': 7007, 'my_kills': 1, 'assists': 0, 'team_kills': 3, 'hunter_name': 'Herbert Lenz'}, {'matchtype': 'Bounty Hunt', 'matchtime': 14, 'cash': 675, 'xp': 4070, 'my_kills': 0, 'assists': 0, 'team_kills': 1, 'hunter_name': 'Wilhelm Storch'}]


def sum_from_dicts(lst_dct, key):
    return np.sum([m[key] for m in lst_dct])

batch_summary = {}
for ss_or_bh in ['Soul Survivor', 'Bounty Hunt']:

    matches = [m for m in cache if m['matchtype'] == ss_or_bh]
    batch_summary[ss_or_bh] = {}
    batch_for_mytype = batch_summary[ss_or_bh]  # pointer

    # skips keys can can't be summed
    skip_keys = ['matchtype', 'hunter_name', 'matchtype']

    keys_iter = matches[0].keys()
    for st in keys_iter:  # matchtime, my_kills, etc
        # Skip keys that can't be summed
        if st in skip_keys:
            continue
        total = sum_from_dicts(matches, st)
        batch_for_mytype[st] = total

    # per minute calculation can only be done after time is summed
    for st in keys_iter:
        if st in skip_keys:
            continue
        per_min_key = st + '_per_min'
        batch_for_mytype[per_min_key] = batch_for_mytype[st] / batch_for_mytype['matchtime']