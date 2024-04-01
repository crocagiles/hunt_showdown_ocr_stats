import easyocr
import os
from datetime import datetime, timedelta
from pathlib import Path
import platform
from functools import lru_cache
import pprint
from copy import deepcopy

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from cv2 import imread
import numpy as np


def crop_match_type_roi(reader, img_array):
    results = reader.readtext(img_array, text_threshold=0.3)
    return [results[x][1] for x in range(len(results))]

def imread_crop_dm(screenshot_png):
    '''Crop full res screenshot in case dual monitor is used.'''
    img_array = imread(str(screenshot_png))
    if img_array.shape == (3840, 4720, 3):
        img_array = img_array[1270:2709, 2160:]
    return img_array

def find_image_pairs(folder_path, time_threshold=30, tonight= False):
    image_pairs = []
    files = sorted(os.listdir(folder_path))
    for i in range(len(files)):
        for j in range(i+1, len(files)):
            image1_path = os.path.join(folder_path, files[i])
            image2_path = os.path.join(folder_path, files[j])
            image1_created = datetime.fromtimestamp(os.path.getmtime(image1_path))
            image2_created = datetime.fromtimestamp(os.path.getmtime(image2_path))
            time_difference = abs((image2_created - image1_created).total_seconds())

            current_datetime = datetime.now()
            since_game = current_datetime - image1_created
            if tonight and since_game > timedelta(hours=12):
                continue
            if time_difference <= time_threshold:
                image_pairs.append((files[i], files[j]))
    return image_pairs
def get_creation_time_windows(path):
    if platform.system() == 'Windows':
        creation_time = os.path.getmtime(path)
        creation_date = datetime.fromtimestamp(creation_time)#.strftime('%Y-%m-%d')
        # creation_time = datetime.fromtimestamp(creation_time)#.strftime('%H:%M:%S')
        return creation_date
    else:
        raise NotImplementedError("This function is only implemented for Windows.")

def match_timer_to_secs(time_str):
    # Split the time string into minutes and seconds
    split = '.' if '.' in time_str else ':'
    minutes, seconds = map(int, time_str.split(split))
    # Convert seconds to minutes and round to the nearest minute
    total_seconds = (minutes * 60) + seconds
    return total_seconds

@lru_cache(maxsize=None)
def get_data_from_image_pairs(dir_with_pairs, debug_rois=False, tonight=False):
    '''
    :param dir_with_pairs: Image directory containing pairs of hunt screenshots
    :param debug_rois: Bool for saving _debug_imgs directory with diagnostic images
    :return:
    '''
    # Dict with info on how to extract details, which image to extract from, etc
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
            'roi': [370, 430, 900, 1800],
            'img_id': 'last_match_scrnshot'
        }
    }

    pairs = find_image_pairs(dir_with_pairs, tonight=tonight)
    details = []
    reader = easyocr.Reader(['en'])  # init ocr reader only once

    for pair in tqdm(pairs):
        pairs_fp = [Path(dir_with_pairs) / x for x in pair]
        stmp_date = get_creation_time_windows(pairs_fp[0])
        pair_id = '_'.join([p.name[-9:-4] for p in pairs_fp])  # eg '(141)_(142)'
        im1, im2 = [imread_crop_dm(str(x)) for x in pairs_fp]

        # determine which image is which (mission summary vs. last match)
        # mission summary is #1, last match is #2
        im1b, im2b = [im[300:900, 0:200, :].mean() for im in [im1, im2]]
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
            img_cropped = img_arr[roi[0]:roi[1], roi[2]:roi[3], :]
            info['img_cropped'] = img_cropped
            ocr = crop_match_type_roi(reader, img_cropped)
            info['ocr_raw'] = ocr

        # Process and save OCR data in useful format
        ocr_raw_dict = {stat: info['ocr_raw'] for stat, info in stats_extract.items()}
        try:
            processed = {
                'match_type': 'Bounty Hunt' if ocr_raw_dict['match_type'][0] == 'MISSION SUMMARY' else 'Soul Survivor',
                'match_datetime': stmp_date,
                'match_timer_secs': match_timer_to_secs(ocr_raw_dict['match_timer_secs'][0]),
                'match_timer_mins': round(match_timer_to_secs(ocr_raw_dict['match_timer_secs'][0]) / 60),
                'cash': int(ocr_raw_dict['cash_xp'][1].replace(',', '')),
                'xp': int(ocr_raw_dict['cash_xp'][0].replace(',', '')),
                'my_kills': int(ocr_raw_dict['my_kills'][0]) if (ocr_raw_dict['my_kills']) else 0,
                'assists': int(ocr_raw_dict['assists'][0]) if (ocr_raw_dict['assists']) else 0,
                'team_kills': int(ocr_raw_dict['team_kills'][0]) if (ocr_raw_dict['team_kills']) else 0,
                'hunter_name': ocr_raw_dict['hunter_name'][0]
            }
        except IndexError:
            processed = {}
            gen_ocr_diagnostirc(stats_extract, ocr_raw_dict, f'{pair_id}_dbg.jpg')
        # per minute calculations
        skip_keys = ['match_type', 'hunter_name', 'match_timer_secs', 'match_timer_mins', 'match_datetime']
        for key, val in list(processed.items()):
            if key in skip_keys:
                continue
            processed[f'{key}_per_sec'] = val / processed['match_timer_secs']
            processed[f'{key}_per_min'] = val / (processed['match_timer_secs'] / 60)
            processed[f'{key}_per_hour'] = val / (processed['match_timer_secs'] / 60 / 60)

        details.append(processed)

        # if debug_rois:
        #     debug_save_loc = Path('debug_imgs')
        #     if not debug_save_loc.exists():
        #         debug_save_loc.mkdir()



    return details

def gen_ocr_diagnostirc(stats_extract, ocr_raw_dict, fname):
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
    plotname = fname
    save_path = debug_save_loc / plotname
    plt.savefig(save_path)
    plt.close()

    return

def get_summary_from_data(gamedata_list):
    batch_summary = {}
    for ss_or_bh in ['Soul Survivor', 'Bounty Hunt']:

        games_list = [m for m in gamedata_list if m['match_type'] == ss_or_bh]
        batch_summary[ss_or_bh] = {}
        batch_for_mytype = batch_summary[ss_or_bh]  # pointer

        batch_for_mytype['data_by_match'] = games_list  # save individual game data for plotting later

        # skips keys can can't be summed
        skip_keys = ['match_type', 'hunter_name', 'match_type', 'match_datetime']

        # Run Calculations across batch of games
        keys_iter = games_list[0].keys()
        for stat in keys_iter:  # match_timer, my_kills, etc
            # Skip keys that can't be summed
            if stat in skip_keys:
                continue
            func = np.mean if 'per_' in stat else np.sum
            total = func([m[stat] for m in games_list])
            batch_for_mytype[stat] = total

        # range of dates and times that this batch covers
        all_date_time = [m['match_datetime'] for m in games_list]
        first, last = min(all_date_time), max(all_date_time)
        fmt = '%Y-%m-%d | %H:%M'
        batch_for_mytype['date_time_range'] = f'''Datetime Range: {first.strftime(fmt)}\nLast datetime: {last.strftime(fmt)}'''

    return batch_summary

def plot_lines(summary_dict, to_plot):
    # all_data = [y['data_by_match'] for x, y in summary_dict.items()]  # list with length of two
    # df_all_data = pd.DataFrame([item for sublist in all_data for item in sublist])  # combines both match type data
    fig, axs = plt.subplots(len(to_plot), figsize=(6, 12))
    fig.suptitle('Session Summary', fontsize=20, fontweight='bold')
    for i, stat in enumerate(to_plot):
        for i_mtype, mtype in enumerate(list(summary_dict.keys())):
            if mtype == 'Soul Survivor':
                continue
            is_multi_axis = True if type(stat) == list else False
            df = pd.DataFrame(summary_dict[mtype]['data_by_match'])
            x_range = range(1, df.shape[0] + 1)
            if not is_multi_axis:
                sns.lineplot(data=df, x=x_range, y=stat, label=stat, ax=axs[i])
                axs[i].set_ylabel(stat.replace('per_sec', ' Per Hour '))
            else:
                sns.lineplot(data=df, x=x_range, y=stat[0], label=stat[0], ax=axs[i])
                if not stat[1] == 'my_kills':
                    ax2 = axs[i].twinx()
                else: ax2 = axs[i]
                sns.lineplot(data=df, x=x_range, y=stat[1], label=stat[1], ax=ax2, ls='--', color='green')
                axs[i].set_ylabel(stat[0])
                ax2.set_ylabel(stat[1].replace('per_sec', ' Per Hour '))
                ax2.legend()


            # Add labels and title
            stat_str = stat.replace('per_sec', ' Per Hour ') if not is_multi_axis else f'{stat[0]} | {stat[1].replace('per_sec', ' Per Hour')}'
            axs[i].set_title(f'{stat_str} by game')
            axs[i].set_xlabel('Match Number')
            axs[i].legend()

            axs[i].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Customize grid and ticks
        # plt.grid(True, linestyle='--', alpha=0.7)

        # Show plot
    plt.tight_layout()
    plt.show()

    return
def plot_hists(summary_dict, to_hist_plot ):
    plot_config = {}

    colors = ['blue', 'red']
    all_data = [y['data_by_match'] for x, y in summary_dict.items()]  # list with length of two
    df_all_data = pd.DataFrame([item for sublist in all_data for item in sublist])  # combines both match type data
    for stat in to_hist_plot:

        for i_mtype, mtype in enumerate(list(summary_dict.keys())):
            # data_by_match = summary_dict[mtype]
            df = pd.DataFrame(summary_dict[mtype]['data_by_match'])
            # Plot histogram
            # if 'per_sec' in stat:
            #     df[stat] = df[stat].apply(lambda x: x * 60 * 60)  # kps -> kph
            #     df_all_data[stat] = df_all_data[stat].apply(lambda x: x * 60 * 60)  # kps -> kph
            # if 'match_timer_secs' in stat:
            #     df[stat] = df[stat].apply(lambda x: x / 60)  # secs -> mins
            #     df_all_data[stat] = df_all_data[stat].apply(lambda x: x / 60)  # secs -> mins

            n_bins = 10
            bin_range = np.linspace(min(df_all_data[stat]), max(df_all_data[stat]), n_bins)

            sns.histplot(data=df, x=stat, kde=True, bins=bin_range, label=mtype, color=colors[i_mtype])

        # Add labels and title
        stat_str = stat.replace('per_sec', ' Per Hour ')
        plt.title(f'Distribution of {stat_str}')
        plt.xlabel(stat_str)
        plt.ylabel('Frequency')
        plt.legend()

        # Customize grid and ticks
        # plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)

        # Show plot
        plt.show()
    return
def plot_summary_data(summary_dict):


    to_hist_plot= ['xp_per_min', 'my_kills_per_hour', 'team_kills_per_hour', 'cash_per_min', 'cash',
       'match_timer_mins', 'team_kills']

    # plot_hists(summary_dict, to_hist_plot)
    to_line_plot = [['team_kills', 'my_kills'], ['xp', 'xp_per_min'], ['cash', 'cash_per_min'], 'match_timer_mins']
    plot_lines(summary_dict, to_line_plot)

    # plot correlations

    # TODO corrolation of match time to dollar/min, xp / min,

    return
def filter_key(d, key_to_ignore):
    return {k: v for k, v in d.items() if k != key_to_ignore}
def main(dir_with_pairs, debug=False):


    game_data_list = get_data_from_image_pairs(dir_with_pairs, debug_rois=debug, tonight=True)
    summary = get_summary_from_data(game_data_list)

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(filter_key(summary, 'data_by_match'))

    plot_summary_data(summary)

    return summary

if __name__ == '__main__':
    dir_with_pairs = Path(r'C:\Users\giles\Pictures\Screenshots')
    main(dir_with_pairs)