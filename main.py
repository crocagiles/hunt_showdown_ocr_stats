import easyocr
import os
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm
from cv2 import imread
def crop_matchtype_roi(img_array):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(img_array)
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
    minutes, seconds = map(int, time_str.split('.'))
    # Convert seconds to minutes and round to the nearest minute
    total_minutes = minutes + round(seconds / 60)
    return total_minutes


#               y start, y end, x start, x end
roi_matchtype = [111, 184, 850,  1681]
roi_matchtime = [860, 900, 1840, 1950]
roi_cash_xp = [950, 995, 2000, 2200]
roi_my_kills = [715, 760, 1200, 1400]
roi_assists = [1150, 1200, 1550, 1650]
roi_team_kills = [1150, 1200, 900, 1000]
roi_hunter_name = [380, 425, 800, 1600 ]

# Dict with info on how to extract details, which image to extract from, etc
stats_extract = {
    'matchtype': {
        'roi': [111, 184, 850,  1681],
        'img_id': 'xp_cash_screenshot'
    },
    'matchtime': {
        'roi': [860, 900, 1840, 1950],
        'img_id': 'pentagram_screenshot'
    },
    'cash_xp': {
        'roi': [950, 995, 2000, 2200],
        'img_id': 'xp_cash_screenshot'
    },
    'my_kills': {
        'roi': [715, 760, 1200, 1400],
        'img_id': 'pentagram_screenshot'
    },
    'assists': {
        'roi': [1150, 1200, 1550, 1650],
        'img_id': 'pentagram_screenshot'
    },
    'team_kills': {
        'roi': [1150, 1200, 900, 1000],
        'img_id': 'pentagram_screenshot'
    },
    'hunter_name': {
        'roi': [380, 425, 900, 1800 ],
        'img_id': 'pentagram_screenshot'
    }
}

dir_with_pairs = r'C:\Users\giles\PycharmProjects\hunt_ocr_playtime_vs_xp\_images'
pairs = find_image_pairs(dir_with_pairs)
details = []
debug = False
for pair in tqdm(pairs):
    pairs_fp = [Path(dir_with_pairs) / x for x in pair]
    im1, im2 = [imread_crop_dm(str(x)) for x in pairs_fp]

    # determine which image is which (match timer vs. match details)
    # mission details is #1, timer is #2
    im1b, im2b = [im[300:900, 0:200,:].mean() for im in [im1, im2]]
    # reverse order if images are backwards
    if im1b > im2b:
        im1, im2 = im2, im1

    # Get OCR raw data for each stat
    for stat, info in stats_extract.items():
        if info['img_id'] == 'pentagram_screenshot':
            img_arr = im2
        elif info['img_id'] == 'xp_cash_screenshot':
            img_arr = im1

        roi = info['roi']
        img_cropped = img_arr[roi[0]:roi[1],roi[2]:roi[3],:]
        info['img_cropped'] = img_cropped
        ocr = crop_matchtype_roi(img_cropped)
        info['ocr_raw'] = ocr

    # Process and save OCR data in useful format
    ocr_raw_dict = {stat: info['ocr_raw'] for stat, info in stats_extract.items()}
    processed = {
        'matchtype': 'Bounty Hunt' if ocr_raw_dict['matchtype'] == 'MISSION SUMMARY' else 'Soul Survivor',
        'matchtime': rounded_minutes(ocr_raw_dict['matchtime'][0]),
        'cash': int(ocr_raw_dict['cash_xp'][1].replace(',', '')),
        'xp': int(ocr_raw_dict['cash_xp'][0].replace(',', '')),
        'my_kills': int(ocr_raw_dict['my_kills'][0])

    }

    if debug:
        fig, axs = plt.subplots(1, len(stats_extract.items()), figsize=(20, 5))
        for i, (stat, info) in enumerate(stats_extract.items()):
            axs[i].imshow(info['img_cropped'])#, label=f'Subplot {i + 1}')
            axs[i].set_title(stat)
            axs[i].set_axis_off()
        # Adjusting layout
        plt.tight_layout()
        # Display the plot
        plt.show()


        # ocr = crop_matchtype_roi(img_cropped)


    # type = 'Bounty Hunt' if matchtype[0] == 'MISSION SUMMARY' else 'Soul Survivor'
    # details.append(f'''
    # ID:         {[x.name[-8:-3] for x in pairs_fp]}
    # Match type: {type}
    # Match Time: {rounded_minutes(matchtime[0])} Minutes
    # Total XP:   {cash_xp[0]}
    # Total $:    {cash_xp[1]}
    # ''')

for d in details:
    print(d)


