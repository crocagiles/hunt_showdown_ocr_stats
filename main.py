import easyocr
import os
from datetime import datetime, timedelta
from pathlib import Path
import platform
from functools import lru_cache
import pprint
from copy import deepcopy
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from cv2 import imread
import numpy as np
import cv2

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

    if not image_pairs:
        error = f'No Image Pairs Found. Screenshot pair must be captured within {time_threshold} seconds of eachother.'
        error += ' Tonight keyword is enabled. only screenshots from last 12 hours are considered.' if tonight else ''
        raise IndexError(error)

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


# @lru_cache(maxsize=None)
def get_data_from_image_pairs(dir_with_pairs, debug_rois=False, tonight=False):
    '''
    :param dir_with_pairs: Image directory containing pairs of hunt screenshots
    :param debug_rois: Bool for saving _debug_imgs directory with diagnostic images
    :return:
    '''

    # read from cached data if it exists
    path_cache = Path('_cache.pickle')
    cache_exists = path_cache.exists()
    if cache_exists:
        with open(str(path_cache), 'rb') as f:
            loaded_cache = pickle.load(f)
    else:
        loaded_cache = None

    list_of_new_ocr_dicts = []
    ocr_reader_initiated = False  # Only turn it on if the cache is not available or it not fully up to date
    from roi_coordinates import stats_extract  # here is the dict with the hard coded pixel values for ROIs

    pairs = find_image_pairs(dir_with_pairs, tonight=tonight)
    for i, pair in enumerate(tqdm(pairs)):
        pairs_fp = [Path(dir_with_pairs) / x for x in pair]
        stmp_date = get_creation_time_windows(pairs_fp[0])
        pair_id = '_'.join([p.name[-9:-4] for p in pairs_fp])  # eg '(141)_(142)'

        # Skip image pair if it already exists in the cache
        try:
            already_processed = (loaded_cache['match_id'] == pair_id).any()
        except TypeError:
            already_processed = False
            pass
        if already_processed:
            print(f'Skipping {pair_id}, loaded from cache')
            continue

        # init ocr reader only once
        if ocr_reader_initiated == False:
            reader = easyocr.Reader(['en'])
            ocr_reader_initiated = True

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
            expand_roi = 0
            er = expand_roi

            img_cropped = cv2.cvtColor(img_arr[roi[0]-er:roi[1]+er, roi[2]-er:roi[3]+er, :], cv2.COLOR_BGR2RGB)

            # Pre processing
            scale_factor = 2
            upscaled = cv2.resize(img_cropped, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            blur = cv2.blur(upscaled, (5, 5))
            no_preprocess = ['hunter_name', 'match_type']
            final_thumbnail = blur if not stat in no_preprocess else img_cropped

            info['img_cropped'] = final_thumbnail
            ocr = crop_match_type_roi(reader, final_thumbnail)



            #
            # import pytesseract
            # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            # custom_config = r'--psm 6'  # Set the page segmentation mode here
            # reader = easyocr.Reader(['en'], gpu=True, recog_network='english', download_enabled=True, detector=True,
            #                         recognizer=True)

            if not ocr:
                print(f'No Char Detected: {pair_id} | {stat}')
            #     # Convert the image to grayscale
            #     gray_image = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
            #
            #     # Apply thresholding
            #     _, binary_image = cv2.threshold(gray_image, 165, 255, cv2.THRESH_BINARY)
            info['ocr_raw'] = ocr


        # Process and save OCR data in useful format
        ocr_raw_dict = {stat: info['ocr_raw'] for stat, info in stats_extract.items()}

        #handle weird ocr corner case? If XP is over 30K, something is probably not right.
        # xp_str =
        # if int(xp_str) > 30000:
        #     xp_str = xp_str[0] + xp_str[2:]

        try:
            processed = {
                'match_id': pair_id,
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
        except:
            gen_ocr_diagnostic(stats_extract, ocr_raw_dict, f'{pair_id}_dbg.jpg')
            continue
        if debug_rois:
            gen_ocr_diagnostic(stats_extract, ocr_raw_dict, f'{pair_id}_dbg.jpg')

        # per minute calculations
        skip_keys = ['match_id', 'match_type', 'hunter_name', 'match_timer_secs', 'match_timer_mins', 'match_datetime']
        for key, val in list(processed.items()):
            if key in skip_keys:
                continue
            processed[f'{key}_per_sec'] = val / processed['match_timer_secs']
            processed[f'{key}_per_min'] = val / (processed['match_timer_secs'] / 60)
            processed[f'{key}_per_hour'] = val / (processed['match_timer_secs'] / 60 / 60)

        list_of_new_ocr_dicts.append(processed)

    # add any newly extracted information to the dataframe

    if loaded_cache is not None and list_of_new_ocr_dicts:  # if there is a loaded cache, append any newly extracted info to it.
        df = loaded_cache
        df_to_append = pd.DataFrame(list_of_new_ocr_dicts)
        print(f'Adding to Cache: {df_to_append['match_id']}')
        df = pd.concat([df, df_to_append], ignore_index=True)
        update_cache = True
    elif loaded_cache is None and list_of_new_ocr_dicts:  # No cache file was present, we'll write a new one
        print(f'Writing new cache! -> {path_cache}')
        df = pd.DataFrame(list_of_new_ocr_dicts)
        update_cache = True
    else:  # If there is no loaded cache, OR if there is a loaded cache and there is no new info to append to it.
        if loaded_cache is None:
            df = pd.DataFrame(list_of_new_ocr_dicts)
        else:
            df = loaded_cache
        print('Nothing to add to Cache.')
        update_cache = False

    if update_cache:
        with open(path_cache, 'wb') as f:
            pickle.dump(df, f)
        print('Cache written successfully')

    return df

def gen_ocr_diagnostic(stats_extract, ocr_raw_dict, fname):
    debug_save_loc = Path('debug_imgs')

    if not debug_save_loc.exists():
        debug_save_loc.mkdir()

    fig, axs = plt.subplots(1, len(stats_extract.items()), figsize=(20, 5))
    for i, (stat, info) in enumerate(stats_extract.items()):
        label = f''' Stat name: {stat}\n
        Val: {ocr_raw_dict[stat]})
        loc:{stats_extract[stat]['roi']} 
        [y_start, y_end, x_start, x_end]
        img: {stats_extract[stat]['img_id']}
        '''
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



def plot_lines(df_summary, to_plot, text_summary):

    ax_cfg = [['blue', 'green'], ['-','--']] # for plots with shared axes
    num_plots = len(to_plot) + 1  # plus one to place some text
    match_types = df_summary['match_type'].unique()
    for mtype in match_types:
        fig, axs = plt.subplots(num_plots, figsize=(6, 12))
        fig.suptitle(f'Session Summary: {mtype}', fontsize=20, fontweight='bold')

        for i, stat in enumerate(to_plot):
            i+=1 # move all plots one down
            df = df_summary[df_summary['match_type'] == mtype]
            is_multi_axis = True if type(stat) == list else False

            x_range = range(1, df.shape[0] + 1)
            if not is_multi_axis:
                sns.lineplot(data=df, x=x_range, y=stat, label=stat, ax=axs[i])
                axs[i].set_ylabel(stat.replace('per_sec', ' Per Hour '))
            else:
                for j, substat in enumerate(stat):  # stat is actually a list of stats to be plotted on one axis
                    if substat in ['my_kills']:  # share axis for "my kills / team kills" plot
                        axs_sub = axs[i]
                    else:
                        axs_sub = axs[i] if j == 0 else axs[i].twinx()
                    sns.lineplot(data=df, x=x_range, y=substat, label=stat[j], ax=axs_sub, color=ax_cfg[0][j], ls=ax_cfg[1][j])

            axs[i].grid(axis='y', linestyle='--', color='gray', alpha=.5)
            # Add labels and title
            stat_str = stat.replace('per_sec', ' Per Hour ') if not is_multi_axis else f'{stat[0]} | {stat[1].replace('per_sec', ' Per Hour')}'
            axs[i].set_title(f'{stat_str} by game')
            axs[i].set_xlabel('Match Number')
            # axs[i].legend()

            axs[i].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

            # Customize grid and ticks
            # plt.grid(True, linestyle='--', alpha=0.7)

            # Show plot

        # Add some text outside the main plot area
        text_subplot = axs[0]
        # Hide tick marks and labels
        # text_subplot.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        # text_subplot.grid(False)
        # text_subplot.set_xlabel("Your xlabel", fontsize=12)
        # text_subplot.set_ylabel("Your ylabel", fontsize=12)
        # text_subplot.set_title("Your title", fontsize=14)

        # Add your text
        text_subplot.text(0.0, 0.5, text_summary, fontsize=10, ha='left', va='center')
        text_subplot.axis('off')

        plt.tight_layout()
        plt.show()

    return
def plot_hists(df_summary, to_hist_plot):
    plot_config = {}

    colors = ['blue', 'red']
    # all_data = [y['data_by_match'] for x, y in df_summary.items()]  # list with length of two
    # df_all_data = pd.DataFrame([item for sublist in all_data for item in sublist])  # combines both match type data
    match_types = df_summary['match_type'].unique()
    for stat in to_hist_plot:
        fig, axs = plt.subplots(1, 1, figsize=(12, 6))


        for i_mtype, mtype in enumerate(match_types):
            # data_by_match = summary_dict[mtype]
            df = df_summary[df_summary['match_type'] == mtype]
            n_bins = 15
            bin_range = np.linspace(min(df_summary[stat]), max(df_summary[stat]), n_bins)

            total_time = seconds_to_hours_minutes_seconds(df['match_timer_secs'].sum())
            total_game = df.shape[0]
            label =f'{mtype}\n{total_game} Unique Games\n{total_time} In Game'
            sns.histplot(data=df, x=stat, kde=True, bins=bin_range, label=label, color=colors[i_mtype])


        # Add labels and title
        stat_str = stat.replace('per_sec', ' Per Hour ')
        # plt.title()
        fig.suptitle(f'Distribution of {stat_str} (game-to-game)', fontsize=20, fontweight='bold')
        plt.xlabel(stat_str)
        plt.ylabel('Frequency')
        plt.legend()

        # Customize grid and ticks
        # plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)

        # Show plot
        plt.show()
    return
def plot_summary_data(df_summary, tonight=False):

    if tonight:
        current_datetime = datetime.now()
        twelve_hours_ago = current_datetime - timedelta(hours=12)
        df_summary = df_summary[df_summary['match_datetime'] >= twelve_hours_ago]
        assert not df_summary.empty,  "Tonight keyword is enabled but no match data exists from the last 12 hours."
        summary_d, summary_t = df_to_summary(df_summary)
    else:
        summary_d, summary_t = df_to_summary(df_summary)
    to_hist_plot= ['xp_per_min', 'my_kills_per_hour', 'my_kills', 'team_kills_per_hour', 'cash_per_min', 'cash',
       'match_timer_mins', 'team_kills']
    if not tonight:
        plot_hists(df_summary, to_hist_plot)

    to_line_plot = [['team_kills', 'my_kills'], ['xp', 'xp_per_min'], ['cash', 'cash_per_min'], 'match_timer_mins']
    plot_lines(df_summary, to_line_plot, summary_t)

    # plot correlations

    # TODO corrolation of match time to dollar/min, xp / min,

    return
def filter_key(d, key_to_ignore):
    return {k: v for k, v in d.items() if k != key_to_ignore}


def seconds_to_hours_minutes_seconds(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60

    if hours > 0:
        return f"{hours} hour{'s' if hours > 1 else ''}, {minutes} minute{'s' if minutes > 1 else ''}, {remaining_seconds} second{'s' if remaining_seconds > 1 else ''}"
    elif minutes > 0:
        return f"{minutes} minute{'s' if minutes > 1 else ''}, {remaining_seconds} second{'s' if remaining_seconds > 1 else ''}"
    else:
        return f"{remaining_seconds} second{'s' if remaining_seconds > 1 else ''}"

def df_to_summary(df):

    sd = {}
    to_sum = ['match_timer_secs', 'cash', 'xp', 'team_kills', 'my_kills', 'assists']
    # to_avg_over_time = ['cash_per_min', 'xp_per_min', 'team_kills_per_hour']
    dt_f, dt_l = df['match_datetime'].min(), df['match_datetime'].max()
    for s in to_sum:
        key_s, key_a = s + '_total', s + '_per_sec_avg'
        sum_a = df[s].sum()
        sd[key_s] = sum_a
        sd[key_a] = sum_a / sd['match_timer_secs_total']
    # time_s_total = sd['match_timer_secs_total']
    # for a in to_sum:
    #     key = a + '_avg'
    #     total = df[s].sum() /

    #     sd[key] = df[a].mean()
    # sd['datetime_range'] = [dt_f, dt_l]

    summary_text = f'''
    Date/Time Range:     {dt_f.strftime('%m/%d/%y %I:%M %p')} - {dt_l.strftime('%m/%d/%y %I:%M %p')}
    Total Games:         {df.shape[0]}
    Total In-Game Time:  {seconds_to_hours_minutes_seconds(sd['match_timer_secs_total'])}
    Total Hunt Dollars: ${format(sd['cash_total'], ",")}
    Total XP Earned:     {format(sd['xp_total'], ",")}
    Total Team Kills:    {sd['team_kills_total']}
    Dolchy's Kills:      {sd['my_kills_total']}
    Avg $/Hour:          {round(sd['cash_per_sec_avg'] * 60 * 60)}
    Avg XP/Hour:         {round(sd['xp_per_sec_avg'] * 60 * 60)}
    Avg Team Kills/Hour: {round(sd['team_kills_per_sec_avg'] * 60 * 60, 2)}
    '''

    return sd, summary_text
def main(dir_with_pairs, debug=False, tonight=False):


    df_game_data = get_data_from_image_pairs(dir_with_pairs, debug_rois=debug, tonight=False)

    # summary = get_summary_from_data(game_data_list)
    #
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(filter_key(summary, 'data_by_match'))
    #
    plot_summary_data(df_game_data, tonight=tonight)

    return #summary

if __name__ == '__main__':
    dir_with_pairs = Path(r'C:\Users\giles\Pictures\Screenshots')
    main(dir_with_pairs, debug=False, tonight=False)