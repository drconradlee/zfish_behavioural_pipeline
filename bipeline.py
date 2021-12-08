# Import packages
import numpy as np
import os
import argparse
import multiprocessing
from ffmpeg_seg_comp import run_segmentation
from detect_and_export import detect_and_export
import shutil
import beh_video_processing as bvp
import argparse
import multiprocessing
import glob
import os
import cv2
from pprint import pprint
import pandas as pd
import deeplabcut

# Full Behavioural Pipeline for Well-segmentation and Deep-Lab-Cut
# Written by Dr Conrad Chun Yin Lee 2022
# For Usage in Scott Laboratory 

if __name__ == '__main__':
    AP = argparse.ArgumentParser()
    AP.add_argument("-d",
                    "--folder_path",
                    required=True,
                    help="Folder path where the raw videos are located")    
    AP.add_argument("-n",
                    "--target_num_wells",
                    required=True,
                    help='Expected Number of wells',
                    default=16)  
    AP.add_argument("-f",
                    "--sample_frame",
                    required=False,
                    help="Select sample frame",
                    default=1000)
    AP.add_argument("-min",
                    "--min_well_size",
                    required=False,
                    help="Minimium well radius size in pixel",
                    default=100)
    AP.add_argument("-max",
                    "--max_well_size",
                    required=False,
                    help="Maximum well radius size in pixel",
                    default=150)
    AP.add_argument("-t",
                    "--row_threshold",
                    required=False,
                    help='Well to well row tolerance',
                    default = 50)         
    AP.add_argument("-p",
                    "--padding",
                    required=False,
                    help="Pixel padding value",
                    default=5)  
    
    # DEEP-LAB CUT INPUT        

    AP.add_argument("-c",
                    "--config_fpath",
                    required=True,
                    help="Folder path where config.yaml file is located located")
    AP.add_argument("-s",
                    "--shuffle",
                    required=True,
                    type=int,
                    help="shuffle index of the training dataset. Default is 0 (ResNet50 in our case)",
                    default=0)
    AP.add_argument("-vf",
                    "--video_format",
                    help="video format of the videos. Defaults to \'.avi\'.",
                    required=False,
                    default='.avi')

 
    ARGS = vars(AP.parse_args())
    directory = ARGS['folder_path']
    min_pad = int(ARGS['padding'])
    min_well = int(ARGS['min_well_size'])
    max_well = int(ARGS['max_well_size'])
    examine_frame = int(ARGS['sample_frame'])
    target_num_well = int(ARGS['target_num_wells'])
    row_thr = int(ARGS['row_threshold'])

    # Run through the folder and get all the avi names
    file_list = [];
    for filename in os.listdir(directory):
        if filename.endswith(".avi"): 
            file_list.append(os.path.join(directory, filename))
            continue
        else:
            continue
    
    print(' ')
    print('Total AVI Files: ' + str(np.shape(file_list)[0]))
    print('###############################################')
    print(' ')
    print('Detecting Wells...')
    print('-----------------------------------------------')
    num_wells,all_coord = detect_and_export(file_list,examine_frame,min_well,max_well,min_pad,target_num_well,row_thr)
    print(' ')
    print('Writing AVI Files...')
    print('-----------------------------------------------')
    print(' ')

    # Segment Wells and export as avi
    target_dir = os.path.join(os.path.dirname(file_list[0]), 'well_videos')
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
     
    all_processes = [multiprocessing.Process(target=run_segmentation, args=(file_list,file_num,all_coord,num_wells)) for file_num in np.arange(np.shape(all_coord)[0])]
    for p in all_processes:
        p.start()

    for p in all_processes:
        p.join()

    print(' ')
    print('-----------------------------------------------')
    print('Well Segmentation is Complete!')
    print('-----------------------------------------------')
    print(' ')

    # START DEEP LAP CUT
    print('###############################################')
    print('STARTING DEEP-LAB-CUT')
    print('-----------------------------------------------')

    ARGS['videos_path'] = target_dir
    N_WELLS = target_num_well  
    pool = multiprocessing.Pool()

    print("searching for videos using pattern {}".format(os.path.join(ARGS['videos_path'], '*' + ARGS['video_format'])))
    pattern = os.path.join(ARGS['videos_path'], '*' + ARGS['video_format'])
    videos_list = glob.glob(pattern)

    if len(videos_list) == 0:
        raise ValueError("Could not find any {} videos within the folder {}".format(ARGS['video_format'], ARGS['videos_path']))
    else:
        pprint('Videos to process:')
        pprint(videos_list)

    deeplabcut.analyze_videos(ARGS['config_fpath'], videos_list, videotype=ARGS['video_format'], shuffle=ARGS['shuffle'], gputouse=0)