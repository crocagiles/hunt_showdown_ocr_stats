# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from cv2 import imread
import easyocr
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
def crop_matchtype_roi(img_array, roi = [111, 184, 850,  1681]):

    # if img_array.shape == (3840, 4720, 3):
    #     img_array = img_array[1270:2709, 2160:]  # 2160 1270 xy upper left
    
    crop = img_array[roi[0]:roi[1],roi[2]:roi[3],: ]
    reader = easyocr.Reader(['en'])
    results = reader.readtext(crop)
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

# Read an image
# image_mis_sum = r"C:\Users\giles\Pictures\Screenshots\Screenshot (141).png"
# image_timer = r"C:\Users\giles\Pictures\Screenshots\Screenshot (142).png"
#                 # y start, y end start, x end, x end
roi_matchtype = [111, 184, 850,  1681]
roi_matchtime = [860, 900, 1840, 1950]
roi_cash_xp = [950, 995, 2000, 2200]


dir_with_pairs = r'C:\Users\giles\PycharmProjects\hunt_ocr_playtime_vs_xp\_images'
pairs = find_image_pairs(dir_with_pairs)
details = []
for pair in tqdm(pairs):
    pairs_fp = [Path(dir_with_pairs) / x for x in pair]
    im1, im2 = [imread_crop_dm(str(x)) for x in pairs_fp]

    # determine which image is which (match timer vs. match details)
    # mission details is #1, timer is #2
    im1b, im2b = [im[300:900, 0:200,:].mean() for im in [im1, im2]]
    # reverse order
    if im1b > im2b:
        im1, im2 = im2, im1

    # plt.imshow(im1)
    # plt.imshow(im2)
    # plt.show()
    matchtype = crop_matchtype_roi(im1, roi=roi_matchtype)
    matchtime = crop_matchtype_roi(im2, roi=roi_matchtime)
    cash_xp = crop_matchtype_roi(im1, roi=roi_cash_xp)

    type = 'Bounty Hunt' if matchtype[0] == 'MISSION SUMMARY' else 'Soul Survivor'
    details.append(f'''
    ID:         {[x.name[-8:-3] for x in pairs_fp]}
    Match type: {type}
    Match Time: {rounded_minutes(matchtime[0])} Minutes
    Total XP:   {cash_xp[0]}
    Total $:    {cash_xp[1]}
    ''')

for d in details:
    print(d)


