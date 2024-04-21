import cv2
import numpy as np
import time
import pyautogui
import matplotlib.pyplot as plt
import cv2
from skimage import metrics
import winsound
from pathlib import Path
from datetime import datetime


def take_screenshot():
    screenshot = pyautogui.screenshot()
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
    return screenshot


def take_screenshot_and_detect(template, threshold=0.7, sound='SystemHand'):
    # Take a screenshot of the entire screen

    screenshot = take_screenshot()
    # Calculate SSIM
    ssim_score = metrics.structural_similarity(template[::4, ::4, 1], screenshot[::4, ::4, 1], full=True)
    print(f"SSIM Score: ", round(ssim_score[0], 2))

    if ssim_score[0] > threshold:
        winsound.PlaySound(sound, winsound.SND_ALIAS)
        return True, screenshot

    return False, False


def main():
    sleep_time = 4
    template_summary = cv2.cvtColor(cv2.imread('example_matchsum.png'), cv2.COLOR_BGR2RGB)
    template_last_match = cv2.cvtColor(cv2.imread('example_lastmatch.png'), cv2.COLOR_BGR2RGB)

    ms_cache = []  # cache multiple screenshots and save the best one.
    ms_first_detection, lm_is_found = False, False
    counter = 0
    while True:
        if not ms_first_detection:
            ms_is_found, scrn = take_screenshot_and_detect(template_summary)
            if ms_is_found:
                print('1st detection of match summary')
                ms_first_detection = True
                ms_cache.append(scrn)
                sleep_time = 1
        # Triggers after match summary screen is found, waiting for last match details
        elif ms_first_detection and not lm_is_found:  # if match summary is seen
            print('Waiting for last match scrnshot')
            # keep playing sound to let user know prgm is waiting..
            counter += 1
            if counter % 3 == 0:
                winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
            # start caching screenshots to save only the last good one for match summary
            # this is because ms can be found during the animation before all info is available on the screen.
            ms_cache.append(take_screenshot())
            lm_is_found, last_match_ar = take_screenshot_and_detect(template_last_match, sound='SystemExit',
                                                                    threshold=.85)

        # Triggers if both screenshots are found.
        elif ms_first_detection and lm_is_found:
            print('Found both screenshots!')
            break
        else:
            print('this should never trigger!')

        time.sleep(sleep_time)

    # get best match summary image (last image with good match to template)
    match_summary_ar = \
    [c for c in ms_cache if metrics.structural_similarity(template_summary[:, :, 1], c[:, :, 1], full=True)[0] > .7][-1]

    path_screenshots = Path('auto_detected_screenshots')
    if not path_screenshots.exists():
        path_screenshots.mkdir()

    current_datetime = datetime.now()
    datetime_stamp = current_datetime.strftime("%y%m%d%H%M")

    arrays_out = [match_summary_ar, last_match_ar]
    names_out = [path_screenshots / f'{datetime_stamp}_ms.png', path_screenshots / f'{datetime_stamp}_lm.png']

    for i, n in enumerate(names_out):
        cv2.imwrite(str(n), arrays_out[i])

    return


if __name__ == '__main__':

    while True:
        main()
        time.sleep(200)  # Wait after both screenshots are found, it'll be a while until the next match finishes
