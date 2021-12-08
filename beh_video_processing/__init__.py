from math import e
import os
import time
import copy
import json
import numpy as np
from numpy import random
from tqdm import tqdm
import cv2
import napari
import pandas as pd
import warnings
import skimage
from skimage import io, draw, util
from skimage.feature import match_template
from skimage.transform import rescale
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt


def get_bkg_images(video, frame_fraction=0.01):
    """get_bkg_images
        computes a mean background image of the video
        using a fraction of the frames chosen randomly
        inputs: video_files
                    list of str with a file paths
                frame_fraction
                    the fraction of frames that the function uses
                    to compute the mean image
        outputs: png files of the mean images saved in the same
                    folder where the video files are located with
                    suffix *_bkg
    """
    print('Get background image of ', video)
    # load video capture
    capture = cv2.VideoCapture(video)

    # get number of frames
    capture_properties = {
        'n_frames': int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': capture.get(cv2.CAP_PROP_FPS),
    }
    # show video data
    print(capture_properties)

    # set the number of frames to be used in the mean background
    # image calculation
    n_random_frames = int(frame_fraction*capture_properties['n_frames'])
    # get the random frames to estiamte background image
    random_frames = random.randint(1, capture_properties['n_frames']+1, (n_random_frames, 1))
    # numpy ndarray to store frames
    frames_all = np.zeros([n_random_frames,
                           capture_properties['height'],
                           capture_properties['width']])

    count = 0
    # store the random frames in frames_all as greyscale images
    print('Capturing frames to compute mean image...')
    for frame_id in tqdm(random_frames):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id-1)
        ret, frame = capture.read()

        if ret:
            frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames_all[count, :, :] = frame_grey
            count += 1
        else:
            print("Error opening video stream or file")
    capture.release()

    # compute mean background image and store it as np.uint8
    background = np.mean(frames_all, axis=0)
    background = background.astype(np.uint8)

    # save background image in the same folder where the videos
    # are located (True output means save was successful)
    cv2.imwrite(video[:-4] + '_bkg.png', background)

    del frames_all
    del random_frames
    del capture_properties

    return background


def preprocess_videos(video, params=None, ftype='.avi'):
    """preprocess_videos
        preprocesses videos using a set of parameters and saves them
            in the ame folder where the input files are located.
        inputs: video
                    str with a file path
                params
                    dictionary with keys being the preprocessing types,
                    whose values are also dictionaries with the parameters
                    required to run it <<details>>

                    ['clahe']['clipLimit'] (int) 5
                    ['clahe']['tileGridSize'] (tuple) (20,20)
                ftype
                    video file type as string (e.g. '.mp4')
        outputs: video files saved in the same
                    folder where the input video files are
                    located with suffix defining preprocessing
                    method
    """

    # THIS FUNCTION IS DEPRECATED may update in the future
    if not params:
        raise ValueError("No preprocessing parameters define, "
                         "provide a dictionary with at least one "
                         "preprocessing choice")

    start = time.time()
    dpath = os.path.split(video)[0]
    fname = os.path.split(video)[1]

    # check if bkg image exists, if not, obtain one
    if os.path.exists(os.path.join(dpath, fname[0:-4] + '_bkg.png')) and 'bkg_sub' in params.keys():
        bkg_rgb = cv2.imread(fname[0:-4] + '_bkg.png')
    elif not os.path.exists(os.path.join(dpath, fname[0:-4] + '_bkg.png')) and 'bkg_sub' in params.keys():
        bkg_rgb = get_bkg_images(video)

    if 'clahe' in params.keys():
        # created clahe image of the background
        clahe = cv2.createCLAHE(clipLimit=params['clahe']['clipLimit'],
                                tileGridSize=params['clahe']['tileGridSize'])
        bkg_clahe_gray = cv2.cvtColor(bkg_rgb, cv2.COLOR_BGR2GRAY)
        bkg_clahe_gray = clahe.apply(bkg_clahe_gray)

    if 'clahe' and 'bkg_sub' in params.keys():
        # smooth bkg clahe image
        bkg_clahe_gray = cv2.GaussianBlur(bkg_clahe_gray, params['bkg_sub']['kernel'],
                                          params['bkg_sub']['sigma'])
        bkg_clahe_rgb = cv2.cvtColor(bkg_clahe_gray, cv2.COLOR_GRAY2BGR)

    # smooth bkg images
    if 'bkg_sub' in params.keys():
        bkg_rgb = cv2.GaussianBlur(bkg_rgb, params['bkg_sub']['kernel'],
                                   params['bkg_sub']['sigma'])

    # reopen and subtract background
    cap = cv2.VideoCapture(video)

    capture_properties = {'n_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                          'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                          'fps': cap.get(cv2.CAP_PROP_FPS)}
    capture_properties['fourcc'] = 'FFV1'

    # show video data
    print('Preprocessing ', video)
    print(capture_properties)

    # set video writer configuration
    # debugging: print filename without format print(video.split('\\')[-1][:-4])
    if 'bkg_sub' in params.keys():
        out = cv2.VideoWriter(dpath + '\\' +
                              video.split('\\')[-1][:-4] +
                              '_bkgsub' + ftype,
                              cv2.VideoWriter_fourcc(*capture_properties['fourcc']),
                              capture_properties['fps'],
                              (capture_properties['width'], capture_properties['height']))

    if 'clahe' in params.keys():
        out2 = cv2.VideoWriter(dpath + '\\' +
                               video.split('\\')[-1][:-4] +
                               '_clahe' + ftype,
                               cv2.VideoWriter_fourcc(*capture_properties['fourcc']),
                               capture_properties['fps'],
                               (capture_properties['width'], capture_properties['height']))

    if 'clahe' and 'bkg_sub' in params.keys():
        out3 = cv2.VideoWriter(dpath + '\\' +
                               video.split('\\')[-1][:-4] +
                               '_clahe+bkgsub' + ftype,
                               cv2.VideoWriter_fourcc(*capture_properties['fourcc']),
                               capture_properties['fps'],
                               (capture_properties['width'], capture_properties['height']))

    if 'equalizeHist' in params.keys():
        out4 = cv2.VideoWriter(dpath + '\\' +
                               video.split('\\')[-1][:-4] +
                               '_histeq' + ftype,
                               cv2.VideoWriter_fourcc(*capture_properties['fourcc']),
                               capture_properties['fps'],
                               (capture_properties['width'], capture_properties['height']))

    if cap.isOpened():
        # Check if file opened successfully
        ret, frame = cap.read()
        # if opened successfuly, crop first frame
        if ret:
            frame_old = copy.copy(frame)
            if 'bkg_sub' in params.keys():
                frame_bkg = cv2.subtract(frame_old, bkg_rgb)
                out.write(frame_bkg)
            if 'clahe' in params.keys():
                frame_old = cv2.cvtColor(frame_old, cv2.COLOR_BGR2GRAY)
                frame = clahe.apply(frame_old)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                out2.write(frame)
            if 'clahe' and 'bkg_sub' in params.keys():
                frame = cv2.subtract(frame, bkg_clahe_rgb)
                out3.write(frame)
            # frame_bkg = cv2.cvtColor(frame_bkg, cv2.COLOR_BGR2GRAY)
            # frame = clahe.apply(frame_bkg)
            # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # frame = cv2.subtract(frame, bkg_clahe_rgb)
            # out4.write(frame)
            if 'equalizeHist' in params.keys():
                frame_old = cv2.cvtColor(frame_old, cv2.COLOR_BGR2GRAY)
                frame = cv2.equalizeHist(frame_old)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                out4.write(frame)
    else:
        print("Error opening video stream or file")

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            frame_old = copy.copy(frame)
            if 'bkg_sub' in params.keys():
                frame_bkg = cv2.subtract(frame_old, bkg_rgb)
                out.write(frame_bkg)
            if 'clahe' in params.keys():
                frame_old = cv2.cvtColor(frame_old, cv2.COLOR_BGR2GRAY)
                frame = clahe.apply(frame_old)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                out2.write(frame)
            if 'clahe' and 'bkg_sub' in params.keys():
                frame = cv2.subtract(frame, bkg_clahe_rgb)
                out3.write(frame)
            # frame_bkg = cv2.cvtColor(frame_bkg, cv2.COLOR_BGR2GRAY)
            # frame = clahe.apply(frame_bkg)
            # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # frame = cv2.subtract(frame, bkg_clahe_rgb)
            # out4.write(frame)
            if 'equalizeHist' in params.keys():
                frame_old = cv2.cvtColor(frame_old, cv2.COLOR_BGR2GRAY)
                frame = cv2.equalizeHist(frame_old)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                out4.write(frame)
        else:
            break

    # When everything is done, release the video capture object
    # and the video writer object
    cap.release()
    if 'bkg_sub' in params.keys():
        out.release()
    if 'clahe' in params.keys():
        out2.release()
    if 'clahe' and 'bkg_sub' in params.keys():
        out3.release()
    if 'equalizeHist' in params.keys():
        out4.release()

    end = time.time()

    print('Elapsed time during video preprocessing: ',
          str(int((end-start)/60)), ' minutes and ', str(int((end-start) % 60)), ' seconds')


def crop_video(coordinates, method='manual', annotation_type=None, ftype='.avi', **kwargs):
    """crop_video
        crop wells from video using a few different options,
        mainly using output from annotate_video()

        inputs: coordinates
                    dictionary with keys being strings of the filepaths
                    of the videos and values are a list of x,y coordinates
                method
                    method of cropping
        outputs: cropped video files saved in new folder named '.\\well_videos'
    """
    # THIS FUNCTION IS DEPRECATED, may update in the future

    start = time.time()
    dpath = os.path.split(coordinates.keys()[0])[0]
    if not os.path.exists(dpath + '\\well_videos'):
        print("well_videos directory does not exist within {}".format(dpath))
        os.mkdir(dpath + '\\well_videos')

    if method == 'manual' and annotation_type == 'point':
        diameter_px = int(kwargs['diameter_px'])
        w, h = [diameter_px, diameter_px]

        for video in coordinates.keys():
            for roi in range(len(coordinates[video])):
                # open video capture each time we want to crop the video
                cap = cv2.VideoCapture(video)

                capture_properties = {'n_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                                      'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                      'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                      'fps': cap.get(cv2.CAP_PROP_FPS),
                                      'fourcc': cap.get(cv2.CAP_PROP_FOURCC)}

                # show video data
                print(' Cropping ', video.split('\\')[-1], ' well ', str(roi+1))
                # set video writer configuration
                out = cv2.VideoWriter(dpath + '\\well_videos\\' +
                                      video.split('\\')[-1][:-4] +
                                      '_well_' + str(roi+1) + ftype,
                                      cv2.VideoWriter_fourcc(*capture_properties['fourcc']),
                                      capture_properties['fps'], (w, h))

                if cap.isOpened():  # Check if file opened successfully
                    ret, frame = cap.read()
                    # if opened successfuly, crop first frame
                    if ret:
                        x = np.around(coordinates[video]['x'][roi])
                        y = np.around(coordinates[video]['y'][roi])
                        x, y = [int(x), int(y)]
                        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame = frame_crop(frame, [y, x], w, h).astype(np.uint8)
                        out.write(frame)
                else:
                    print("Error opening video stream or file")

                # Read until video is completed
                while cap.isOpened():
                    # Capture frame-by-frame
                    ret, frame = cap.read()
                    if ret:
                        x, y = np.around(coordinates[video][roi])
                        x, y = [int(x), int(y)]
                        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame = frame_crop(frame, [y, x], w, h).astype(np.uint8)
                        out.write(frame)
                    else:
                        break

                # When everything is done, release the video capture object
                # and the video writer object
                cap.release()
                out.release()
    if method == 'manual' and annotation_type == 'rectangle':
        for video in coordinates.keys():

            for roi in range(len(coordinates[video]['x'])):
                # open video capture each time we want to crop the video
                cap = cv2.VideoCapture(video)

                x = np.around(coordinates[video]['x'][roi])
                y = np.around(coordinates[video]['y'][roi])
                w = np.around(coordinates[video]['width'][roi])
                h = np.around(coordinates[video]['height'][roi])
                x, y, w, h = [int(x), int(y), int(w), int(h)]

                capture_properties = {'n_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                                      'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                      'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                      'fps': cap.get(cv2.CAP_PROP_FPS)}
                                     # 'fourcc' : cap.get(cv2.CAP_PROP_FOURCC)}
                # decode fourcc int to string
                # capture_properties['fourcc'] = decode_fourcc(capture_properties['fourcc'])
                capture_properties['fourcc'] = 'FFV1'
                # show video data
                print(' Cropping ', video.split('\\')[-1], ' well ', str(roi+1))

                # set video writer configuratiom
                out = cv2.VideoWriter(dpath + '\\well_videos\\' +
                                      video.split('\\')[-1][:-4] +
                                      '_well_' + str(roi+1) + ftype,
                                      cv2.VideoWriter_fourcc(*capture_properties['fourcc']),
                                      capture_properties['fps'], (w, h))

                if cap.isOpened():  # Check if file opened successfully
                    ret, frame = cap.read()
                    # if opened successfuly, crop first frame
                    if ret:
                        frame = frame_crop(frame,
                                           [y, x],
                                           w, h,
                                           pt_pos='top_left').astype(np.uint8)
                        out.write(frame)
                else:
                    print("Error opening video stream or file")

                # Read until video is completed
                while cap.isOpened():
                    # Capture frame-by-frame
                    ret, frame = cap.read()
                    if ret:
                        frame = frame_crop(frame,
                                           [y, x],
                                           w, h,
                                           pt_pos='top_left').astype(np.uint8)
                        out.write(frame)
                    else:
                        break

                # When everything is done, release the video capture object
                # and the video writer object
                cap.release()
                out.release()

    elif method == 'floodfill':
        pass
    elif method == 'template_matching':
        pass
    end = time.time()
    print('Elapsed time during video cropping step: ',
          str(int((end-start) / 60)), ' minutes and ',
          str(int((end-start) % 60)), ' seconds')


def verify_img_size(frame, w, h):
    """verify_img_size is used to confirm (True/False)
    whether the cropped frame has the expected size.
    This is necessary because the automated template
    matching may find may provide top-left points
    that are beyond image dimensions, so this needs to
    be fixed.

    Args:
        frame (np.array): 2D numpy array (image frame)
        w (int): expected image width
        h (int): expected image height

    Returns:
        (bool) : True if image has expected shape, False if not
    """
    if len(frame.shape) == 2:
        y, x = frame.shape
    else:
        y, x, _ = frame.shape
    if x != w and y != h:
        return False
    elif x == w and y == h:
        return True


def zero_pad_frame(frame, w, h):
    """Zero pad an image frame in case it does not
    have the shape expected when cv2.VideoWriter
    was created.

    Args:
        frame (np.array): 2D numpy array (image frame)
        w (int): expected image width
        h (int): expected image height

    Returns:
        (np.array().astype(np.uint8)): Zero-padded image frame
    """
    dims = frame.shape
    if len(dims) == 2:
        zero_frame = np.zeros((h, w))
        zero_frame[:dims[0], :dims[1]] = frame
    elif len(dims) == 3:
        zero_frame = np.zeros((h, w, dims[2]))
        zero_frame[:dims[0], :dims[1], :dims[2]] = frame
    return zero_frame.astype(np.uint8)


def crop_and_process_video(crop_coords, annotation_type=None, params=None, n_wells=16, ftype='.avi', **kwargs):
    """ crop and process wells from video using a few different options,
        mainly using output from annotate_video()

    Args:
        crop_coords (dict): dictionary with keys being strings of the filepaths
                            of the videos and values are a list of x,y coordinates
        annotation_type ([type], optional): [description]. Defaults to None.
        params ([type], optional): [description]. Defaults to None.
        n_wells (int, optional): [description]. Defaults to 16.
        ftype (str, optional): [description]. Defaults to '.avi'.
    """
    start = time.time()

    # verify if parameters are included in params,
    # if none exist add 'No' as key flag for params
    if not params:
        warnings.warn("No preprocessing parameters given, without a dictionary with at least one preprocessing option, only the cropping step will be run")
        params = {'No': None}
    elif params:
        # add 'No' to params as flag to process the raw videos
        params.update({'No': None})

    if annotation_type == 'auto':
        for video in crop_coords.keys():
            # get folder path from video filepath and create well_videos folder if needed
            dpath = os.path.split(video)[0]
            fname = os.path.split(video)[1]
            if not os.path.exists(os.path.join(dpath, 'well_videos')):
                print("well_videos directory does not exist within {}, so one will be created".format(dpath))
                os.mkdir(os.path.join(dpath, 'well_videos'))
            # check if background image exists, if it doesn't, run get_bkg_images to get one
            if os.path.exists(os.path.join(dpath, fname[:-4] + '_bkg.png')):
                bkg_img = io.imread(video[0:-4] + '_bkg.png')
            elif not os.path.exists(os.path.join(dpath, fname[:-4] + '_bkg.png')):
                bkg_img = get_bkg_images(video)

            # build well template
            if 'well_template' in kwargs.keys() and any(tt == kwargs['well_template'] for tt in ['circle', 'circle_with_halo', 'circle_dark','circle_with_halo_dark']):

                img_shape = bkg_img.shape
                tmplt = create_well_template(img_shape, template_type=kwargs['well_template'])
                max_dim = max(tmplt.shape)
                max_scale = 1.0

            else:
                # when 'well_template' is a file path to a template image
                tmplt = io.imread(kwargs['well_template'], as_gray=True)
                # ensure the given template will work by setting max scale
                # to test max dimension of template and minimum
                # dimension of video
                max_dim = max(tmplt.shape)
                max_scale = min(bkg_img.shape) / max_dim
                pass

            # multi-scale template matching
            start_scale = 0.10  # well must be at least 5% the smallest dimension
            optimal_scale = 0.01
            optimal_max = 0.0

            print('running multiscale template matching algorithm to find wells')
            # run template matching and find well size
            for scale in tqdm(np.linspace(start_scale, max_scale, 100)):

                rescaled_tmplt = rescale(tmplt, scale)
                result = match_template(bkg_img, rescaled_tmplt)

                if np.max(result) > optimal_max:
                    optimal_scale = scale
                    optimal_max = np.max(result)
                    optimal_result = result

            print('optimal scale is ', optimal_scale)
            print('this scale corresponds to well dimensions ', rescale(tmplt, optimal_scale).shape)

            # extract maxima for posterior cropping
            coordinates = peak_local_max(optimal_result, num_peaks=int(n_wells))

            # get a plot with template matching results
            plt.imshow(optimal_result, cmap='bwr')
            plt.scatter(coordinates[:, 1], coordinates[:, 0], c='black', s=1)
            plt.savefig(os.path.join(dpath, fname[:-4]+'_scoremap_templatematching.svg'))

            plt.close()

            # hardcode to find rows of standard well plates and other standards.
            well_rows = {'9': 3, '12': 4, '16': 4, '24': 6, '96': 12}
            order = []

            # sort rectangles in the y coordinate 
            coordinates = coordinates[coordinates[:, 0].argsort()]

            if n_wells in well_rows.keys():
                print("The code considers that this video will have wells in rows \
                of {}. For now, if this is not the case, please see \
                crop_and_process_video() in __init__.py to change it".format(well_rows[n_wells]))

                # sort now considering the number of wells per row
                k = well_rows[n_wells]
                for i in range(k):
                    order.extend(coordinates[i*k:(i*k) + k, 1].argsort() + i*k)
                coordinates = coordinates[order]
            else:
                pass

            # get size of the wells based on size of optimal template
            h, _ = rescale(tmplt, optimal_scale).shape
            coordinates = np.floor(coordinates - 0.05*h)
            coordinates = np.hstack((coordinates, (np.ones((len(coordinates), 2)) * 1.10 * h))).astype('int32')
            crop_coords[video] = {'x': coordinates[:, 1], 'y': coordinates[:, 0], 'height': coordinates[:, 2], 'width': coordinates[:, 3]}

            # save in pandas df
            df = pd.DataFrame(data=crop_coords[video])
            df.to_csv(os.path.join(dpath,fname[:-4]+'_rectanglesauto.csv'), sep=',', index=False)


        # change annotation type parameter so it goes to the same code as
        # 'annotation_type' == rectangle for cropping well videos
        annotation_type = 'rectangle'

    if annotation_type == 'point':
        diameter_px = int(kwargs['diameter_px'])
        w, h = [diameter_px, diameter_px]

        for video in crop_coords.keys():
            # get folder path from video filepath and create well_videos folder if needed
            dpath = os.path.split(video)[0]
            fname = os.path.split(video)[1]
            if not os.path.exists(os.path.join(dpath, 'well_videos')):
                print("well_videos directory does not exist within {}, so one will be created".format(dpath))
                os.mkdir(os.path.join(dpath, 'well_videos'))

            # check if bkg image exists, if not, obtain one
            if 'bkg_sub' in params.keys():
                if os.path.exists(os.path.join(dpath, fname[:-4] + '_bkg.png')) and 'bkg_sub' in params.keys():
                    bkg_rgb = cv2.imread(fname[:-4] + '_bkg.png')
                elif not os.path.exists(os.path.join(dpath, fname[:-4] + '_bkg.png')) and 'bkg_sub' in params.keys():
                    bkg_rgb = get_bkg_images(video)
                bkg_rgb = cv2.GaussianBlur(bkg_rgb, params['bkg_sub']['kernel'],
                                           params['bkg_sub']['sigma'])

            if 'clahe' in params.keys():
                # created clahe image of the background
                clahe = cv2.createCLAHE(clipLimit=params['clahe']['clipLimit'],
                                        tileGridSize=params['clahe']['tileGridSize'])

            if all(i in params.keys() for i in ['clahe', 'bkg_sub']):
                # smooth bkg clahe image
                bkg_clahe_gray = cv2.cvtColor(bkg_rgb, cv2.COLOR_BGR2GRAY)
                bkg_clahe_gray = clahe.apply(bkg_clahe_gray)
                bkg_clahe_gray = cv2.GaussianBlur(bkg_clahe_gray, params['bkg_sub']['kernel'],
                                                  params['bkg_sub']['sigma'])
                bkg_clahe_rgb = cv2.cvtColor(bkg_clahe_gray, cv2.COLOR_GRAY2BGR)

            for roi in range(len(crop_coords[video])):
                x = np.around(crop_coords[video]['x'][roi])
                y = np.around(crop_coords[video]['y'][roi])
                # open video capture each time we want to crop the video
                cap = cv2.VideoCapture(video)

                capture_properties = {'n_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                                      'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                      'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                      'fps': cap.get(cv2.CAP_PROP_FPS)}
                                      # 'fourcc' : cap.get(cv2.CAP_PROP_FOURCC)}
                capture_properties['fourcc'] = 'FFV1'

                # show video data
                print(' Cropping ', fname, ' well ', str(roi+1))

                # set video writer configuration
                # debugging: print filename without format print(video.split('\\')[-1][:-4])
                out_path = os.path.join(dpath,
                                        'well_videos',
                                        fname[:-4] + '_well_' + str(roi+1))
                if 'bkg_sub' in params.keys():
                    out1 = cv2.VideoWriter(out_path + '_bkgsub' + ftype,
                                           cv2.VideoWriter_fourcc(*capture_properties['fourcc']),
                                           capture_properties['fps'],
                                           (w, h))

                if 'clahe' in params.keys():
                    out2 = cv2.VideoWriter(out_path + '_clahe' + ftype,
                                           cv2.VideoWriter_fourcc(*capture_properties['fourcc']),
                                           capture_properties['fps'],
                                           (w, h))

                if all(i in params.keys() for i in ['clahe', 'bkg_sub']):
                    out3 = cv2.VideoWriter(out_path + '_clahe+bkgsub' + ftype,
                                           cv2.VideoWriter_fourcc(*capture_properties['fourcc']),
                                           capture_properties['fps'],
                                           (w, h))

                if 'equalizeHist' in params.keys():
                    out4 = cv2.VideoWriter(out_path + '_histeq' + ftype,
                                           cv2.VideoWriter_fourcc(*capture_properties['fourcc']),
                                           capture_properties['fps'],
                                           (w, h))

                if all(i in params.keys() for i in ['equalizeHist', 'bkg_sub']):
                    out5 = cv2.VideoWriter(out_path + '_bkg_sub+histeq' + ftype,
                                           cv2.VideoWriter_fourcc(*capture_properties['fourcc']),
                                           capture_properties['fps'],
                                           (w, h))
                if 'No' in params.keys():
                    # set video writer configuration
                    # debugging: print filename without format print(video.split('\\')[-1][:-4])
                    out = cv2.VideoWriter(out_path + ftype,
                                          cv2.VideoWriter_fourcc(*capture_properties['fourcc']),
                                          capture_properties['fps'], (w, h))

                if cap.isOpened():  # Check if file opened successfully
                    ret, frame = cap.read()
                    # if opened successfuly, crop first frame
                    if ret:
                        frame_old = copy.copy(frame)
                        if 'bkg_sub' in params.keys():
                            frame_bkgsub = 255 - np.abs(frame_old.astype(np.int32) - bkg_rgb.astype(np.int32)).astype(np.uint8)
                            frame_bkgsub = frame_crop(frame_bkgsub,
                                                      [y, x],
                                                      w, h,
                                                      pt_pos='top_left').astype(np.uint8)
                            if verify_img_size(frame_bkgsub, w, h):
                                out1.write(frame_bkgsub)
                            else:
                                frame_bkgsub = zero_pad_frame(frame_bkgsub, w, h)
                                out1.write(frame_bkgsub)

                        if 'clahe' in params.keys():
                            frame = cv2.cvtColor(frame_old, cv2.COLOR_BGR2GRAY)
                            frame_clahe = clahe.apply(frame)
                            frame_clahe = cv2.cvtColor(frame_clahe, cv2.COLOR_GRAY2BGR)
                            frame_clahe = frame_crop(frame_clahe,
                                                     [y, x],
                                                     w, h,
                                                     pt_pos='top_left').astype(np.uint8)
                            if verify_img_size(frame_clahe, w, h):
                                out2.write(frame_clahe)
                            else:
                                frame_clahe = zero_pad_frame(frame_clahe, w, h)
                                out2.write(frame_clahe)

                        if all(i in params.keys() for i in ['clahe', 'bkg_sub']):
                            frame = cv2.subtract(frame_old, bkg_clahe_rgb)
                            frame = frame_crop(frame,
                                               [y, x],
                                               w, h,
                                               pt_pos='top_left').astype(np.uint8)
                            if verify_img_size(frame, w, h):
                                out3.write(frame)
                            else:
                                frame = zero_pad_frame(frame, w, h)
                                out3.write(frame)

                        if 'equalizeHist' in params.keys():
                            frame = frame_crop(frame_old,
                                               [y, x],
                                               w, h,
                                               pt_pos='top_left').astype(np.uint8)
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            frame = cv2.equalizeHist(frame)
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                            if verify_img_size(frame, w, h):
                                out4.write(frame)
                            else:
                                frame = zero_pad_frame(frame, w, h)
                                out4.write(frame)
                        if all(i in params.keys() for i in ['equalizeHist', 'bkg_sub']):
                            frame = frame_crop(frame_old,
                                               [y, x],
                                               w, h,
                                               pt_pos='top_left').astype(np.uint8)
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            frame = cv2.equalizeHist(frame)
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                            bkg_cropped = frame_crop(bkg_rgb,
                                                     [y, x],
                                                     w, h,
                                                     pt_pos='top_left').astype(np.uint8)

                            frame = cv2.subtract(frame, bkg_cropped)
                        if 'No' in params.keys():
                            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            frame = frame_crop(frame_old,
                                               [y, x],
                                               w, h,
                                               pt_pos='top_left').astype(np.uint8)
                            if verify_img_size(frame, w, h):
                                out.write(frame)
                            else:
                                frame = zero_pad_frame(frame, w, h)
                                out.write(frame)
                else:
                    print("Error opening video stream or file")

                # Read until video is completed
                while cap.isOpened():
                    # Capture frame-by-frame
                    ret, frame = cap.read()
                    # if opened successfuly, crop first frame
                    if ret:
                        frame_old = copy.copy(frame)
                        if 'bkg_sub' in params.keys():
                            frame_bkgsub = 255 - np.abs(frame_old.astype(np.int32) - bkg_rgb.astype(np.int32)).astype(np.uint8)
                            frame_bkgsub = frame_crop(frame_bkgsub,
                                                      [y, x],
                                                      w, h,
                                                      pt_pos='top_left').astype(np.uint8)
                            if verify_img_size(frame_bkgsub, w, h):
                                out1.write(frame_bkgsub)
                            else:
                                frame_bkgsub = zero_pad_frame(frame_bkgsub, w, h)
                                out1.write(frame_bkgsub)

                        if 'clahe' in params.keys():
                            frame = cv2.cvtColor(frame_old, cv2.COLOR_BGR2GRAY)
                            frame_clahe = clahe.apply(frame)
                            frame_clahe = cv2.cvtColor(frame_clahe, cv2.COLOR_GRAY2BGR)
                            frame_clahe = frame_crop(frame_clahe,
                                                     [y, x],
                                                     w, h,
                                                     pt_pos='top_left').astype(np.uint8)
                            if verify_img_size(frame_clahe, w, h):
                                out2.write(frame_clahe)
                            else:
                                frame_clahe = zero_pad_frame(frame_clahe, w, h)
                                out2.write(frame_clahe)

                        if all(i in params.keys() for i in ['clahe', 'bkg_sub']):
                            frame = cv2.subtract(frame_old, bkg_clahe_rgb)
                            frame = frame_crop(frame,
                                               [y, x],
                                               w, h,
                                               pt_pos='top_left').astype(np.uint8)
                            if verify_img_size(frame, w, h):
                                out3.write(frame)
                            else:
                                frame = zero_pad_frame(frame, w, h)
                                out3.write(frame)

                        if 'equalizeHist' in params.keys():
                            frame = frame_crop(frame_old,
                                               [y, x],
                                               w, h,
                                               pt_pos='top_left').astype(np.uint8)
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            frame = cv2.equalizeHist(frame)
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                            if verify_img_size(frame, w, h):
                                out4.write(frame)
                            else:
                                frame = zero_pad_frame(frame, w, h)
                                out4.write(frame)
                        if all(i in params.keys() for i in ['equalizeHist', 'bkg_sub']):
                            frame = frame_crop(frame_old,
                                               [y, x],
                                               w, h,
                                               pt_pos='top_left').astype(np.uint8)
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            frame = cv2.equalizeHist(frame)
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                            bkg_cropped = frame_crop(bkg_rgb,
                                                     [y, x],
                                                     w, h,
                                                     pt_pos='top_left').astype(np.uint8)

                            frame = cv2.subtract(frame, bkg_cropped)

                        if 'No' in params.keys():
                            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            frame = frame_crop(frame_old,
                                               [y, x],
                                               w, h,
                                               pt_pos='top_left').astype(np.uint8)
                            if verify_img_size(frame, w, h):
                                out.write(frame)
                            else:
                                frame = zero_pad_frame(frame, w, h)
                                out.write(frame)
                    else:
                        break

                # When everything is done, release the video capture object
                # and the video writer object
                cap.release()
                if 'bkg_sub' in params.keys():
                    out1.release()
                if 'clahe' in params.keys():
                    out2.release()
                if all(i in params.keys() for i in ['clahe', 'bkg_sub']):
                    out3.release()
                if 'equalizeHist' in params.keys():
                    out4.release()
                if all(i in params.keys() for i in ['equalizeHist', 'bkg_sub']):
                    out5.release()
                if 'No' in params.keys():
                    out.release()

    if annotation_type == 'rectangle':
        for video in crop_coords.keys():
            dpath = os.path.split(video)[0]
            fname = os.path.split(video)[1]
            # check if bkg image exists, if not, obtain one
            if 'bkg_sub' in params.keys():
                if os.path.exists(os.path.join(dpath, fname[:-4] + '_bkg.png')) and 'bkg_sub' in params.keys():
                    bkg_rgb = cv2.imread(os.path.join(dpath, fname[:-4] + '_bkg.png'))
                elif not os.path.exists(os.path.join(dpath, fname[:-4] + '_bkg.png')) and 'bkg_sub' in params.keys():
                    bkg_rgb = get_bkg_images(video)
                bkg_rgb = cv2.GaussianBlur(bkg_rgb, params['bkg_sub']['kernel'],
                                           params['bkg_sub']['sigma'])

            if 'clahe' in params.keys():
                # created clahe image of the background
                clahe = cv2.createCLAHE(clipLimit=params['clahe']['clipLimit'],
                                        tileGridSize=params['clahe']['tileGridSize'])

            if all(i in params.keys() for i in ['clahe', 'bkg_sub']):
                # smooth bkg clahe image
                bkg_clahe_gray = cv2.cvtColor(bkg_rgb, cv2.COLOR_BGR2GRAY)
                bkg_clahe_gray = clahe.apply(bkg_clahe_gray)
                bkg_clahe_gray = cv2.GaussianBlur(bkg_clahe_gray, params['bkg_sub']['kernel'],
                                                  params['bkg_sub']['sigma'])
                bkg_clahe_rgb = cv2.cvtColor(bkg_clahe_gray, cv2.COLOR_GRAY2BGR)
                    
            for roi in range(len(crop_coords[video]['x'])):
                x = np.around(crop_coords[video]['x'][roi])
                y = np.around(crop_coords[video]['y'][roi])
                w = np.around(crop_coords[video]['width'][roi])
                h = np.around(crop_coords[video]['height'][roi])
                x, y, w, h = [int(x), int(y), abs(int(w)), abs(int(h))]
                
                # open video capture each time we want to crop the video
                cap = cv2.VideoCapture(video)

                capture_properties = {'n_frames' : int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                                      'width' : int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                      'height' : int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                      'fps' : cap.get(cv2.CAP_PROP_FPS)}  #,
                                      #'fourcc' : cap.get(cv2.CAP_PROP_FOURCC)}
                capture_properties['fourcc'] = 'FFV1'
                
                # show video data
                print(' Cropping ', video.split('\\')[-1], ' well ', str(roi+1))
                
                # set video writer configuration
                # debugging: print filename without format print(video.split('\\')[-1][:-4])
                out_path = os.path.join(dpath,
                                        'well_videos',
                                        fname[:-4] + '_well_' + str(roi+1))
                if 'bkg_sub' in params.keys():
                    out1 = cv2.VideoWriter(out_path+ '_bkgsub' + ftype,
                                           cv2.VideoWriter_fourcc(*capture_properties['fourcc']),
                                           capture_properties['fps'],
                                           (w,h))

                if 'clahe' in params.keys():
                    out2 = cv2.VideoWriter(out_path + '_clahe' + ftype,
                                           cv2.VideoWriter_fourcc(*capture_properties['fourcc']),
                                           capture_properties['fps'],
                                           (w,h))

                if all(i in params.keys() for i in ['clahe', 'bkg_sub']):
                    out3 = cv2.VideoWriter(out_path + '_clahe+bkgsub' + ftype,
                                           cv2.VideoWriter_fourcc(*capture_properties['fourcc']),
                                           capture_properties['fps'],
                                           (w,h))

                if 'equalizeHist' in params.keys():
                    out4 = cv2.VideoWriter(out_path + '_histeq' + ftype,
                                           cv2.VideoWriter_fourcc(*capture_properties['fourcc']),
                                           capture_properties['fps'],
                                           (w,h))
                if all(i in params.keys() for i in ['equalizeHist', 'bkg_sub']):
                    out5 = cv2.VideoWriter(out_path + '_bkg_sub+histeq' + ftype,
                                           cv2.VideoWriter_fourcc(*capture_properties['fourcc']),
                                           capture_properties['fps'],
                                           (w, h))
                if 'No' in params.keys():
                    # set video writer configuration
                    # debugging: print filename without format print(video.split('\\')[-1][:-4])
                    out = cv2.VideoWriter(out_path + ftype,
                                          cv2.VideoWriter_fourcc(*capture_properties['fourcc']),
                                          capture_properties['fps'],
                                          (w, h))

                if cap.isOpened(): # Check if file opened successfully
                    ret, frame = cap.read()
                    # if opened successfuly, crop first frame
                    if ret:
                        frame_old = copy.copy(frame)
                        if 'bkg_sub' in params.keys():
                            frame_bkgsub = 255 - np.abs(frame_old.astype(np.int32) - bkg_rgb.astype(np.int32)).astype(np.uint8)
                            frame_bkgsub = frame_crop(frame_bkgsub,
                                           [y, x],
                                           w, h,
                                           pt_pos='top_left').astype(np.uint8)
                            if verify_img_size(frame_bkgsub,w,h):
                                out1.write(frame_bkgsub)
                            else:
                                frame_bkgsub = zero_pad_frame(frame_bkgsub,w,h)
                                out1.write(frame_bkgsub)
                                
                        if 'clahe' in params.keys():
                            frame = cv2.cvtColor(frame_old, cv2.COLOR_BGR2GRAY)
                            frame_clahe = clahe.apply(frame)
                            frame_clahe = cv2.cvtColor(frame_clahe, cv2.COLOR_GRAY2BGR)
                            frame_clahe = frame_crop(frame_clahe,
                                           [y, x],
                                           w, h,
                                           pt_pos='top_left').astype(np.uint8)
                            if verify_img_size(frame_clahe,w,h):
                                out2.write(frame_clahe)
                            else:
                                frame_clahe = zero_pad_frame(frame_clahe,w,h)
                                out2.write(frame_clahe)
                            
                        if all(i in params.keys() for i in ['clahe', 'bkg_sub']):
                            frame = cv2.subtract(frame_old, bkg_clahe_rgb)
                            frame = frame_crop(frame,
                                               [y, x],
                                               w, h,
                                               pt_pos='top_left').astype(np.uint8)
                            if verify_img_size(frame,w,h):
                                out3.write(frame)
                            else:
                                frame = zero_pad_frame(frame,w,h)
                                out3.write(frame)
                                
                        if 'equalizeHist' in params.keys():
                            frame = frame_crop(frame_old,
                                               [y, x],
                                               w, h,
                                               pt_pos='top_left').astype(np.uint8)
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            frame = cv2.equalizeHist(frame)
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                            if verify_img_size(frame,w,h):
                                out4.write(frame)
                            else:
                                frame = zero_pad_frame(frame,w,h)
                                out4.write(frame)

                        if all(i in params.keys() for i in ['equalizeHist', 'bkg_sub']):
                            frame = frame_crop(frame_old,
                                               [y, x],
                                               w, h,
                                               pt_pos='top_left').astype(np.uint8)
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            frame = cv2.equalizeHist(frame)
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                            bkg_cropped = frame_crop(bkg_rgb,
                                                     [y, x],
                                                     w, h,
                                                     pt_pos='top_left').astype(np.uint8)
                            
                            frame = cv2.subtract(frame, bkg_cropped)
            
                        if 'No' in params.keys():
                            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            frame = frame_crop(frame_old,
                                               [y, x],
                                               w, h,
                                               pt_pos='top_left').astype(np.uint8)
                            if verify_img_size(frame,w,h):
                                out.write(frame)
                            else:
                                frame = zero_pad_frame(frame,w,h)
                                out.write(frame)
                else:
                    print("Error opening video stream or file")

                # Read until video is completed
                while cap.isOpened():
                    ret, frame = cap.read()
                    # if opened successfuly, crop first frame
                    if ret: 
                        frame_old = copy.copy(frame)
                        if 'bkg_sub' in params.keys():
                            frame_bkgsub = 255 - np.abs(frame_old.astype(np.int32) - bkg_rgb.astype(np.int32)).astype(np.uint8)                         
                            frame_bkgsub = frame_crop(frame_bkgsub,
                                           [y, x],
                                           w, h,
                                           pt_pos='top_left').astype(np.uint8)
                            if verify_img_size(frame_bkgsub,w,h):
                                out1.write(frame_bkgsub)
                            else:
                                frame_bkgsub = zero_pad_frame(frame_bkgsub,w,h)
                                out1.write(frame_bkgsub)
                        
                        if 'clahe' in params.keys():
                            frame = cv2.cvtColor(frame_old, cv2.COLOR_BGR2GRAY)
                            frame_clahe = clahe.apply(frame)
                            frame_clahe = cv2.cvtColor(frame_clahe, cv2.COLOR_GRAY2BGR)
                            frame_clahe = frame_crop(frame_clahe,
                                           [y, x],
                                           w, h,
                                           pt_pos='top_left').astype(np.uint8)
                            if verify_img_size(frame_clahe,w,h):
                                out2.write(frame_clahe)
                            else:
                                frame_clahe = zero_pad_frame(frame_clahe,w,h)
                                out2.write(frame_clahe)
                            
                        if all(i in params.keys() for i in ['clahe', 'bkg_sub']):
                            frame = cv2.subtract(frame_old, bkg_clahe_rgb)
                            frame = frame_crop(frame,
                                               [y, x],
                                               w, h,
                                               pt_pos='top_left').astype(np.uint8)
                            if verify_img_size(frame,w,h):
                                out3.write(frame)
                            else:
                                frame = zero_pad_frame(frame,w,h)
                                out3.write(frame)
                                
                        if 'equalizeHist' in params.keys():
                            frame = frame_crop(frame_old,
                                               [y, x],
                                               w, h,
                                               pt_pos='top_left').astype(np.uint8)
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            frame = cv2.equalizeHist(frame)
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                            if verify_img_size(frame,w,h):
                                out4.write(frame)
                            else:
                                frame = zero_pad_frame(frame,w,h)
                                out4.write(frame)
                        
                        if all(i in params.keys() for i in ['equalizeHist', 'bkg_sub']):
                            frame = frame_crop(frame_old,
                                               [y, x],
                                               w, h,
                                               pt_pos='top_left').astype(np.uint8)
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            frame = cv2.equalizeHist(frame)
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                            bkg_cropped = frame_crop(bkg_rgb,
                                                     [y, x],
                                                     w, h,
                                                     pt_pos='top_left').astype(np.uint8)
                            
                            frame = cv2.subtract(frame, bkg_cropped)

                            if verify_img_size(frame, w, h):
                                out5.write(frame)
                            else:
                                frame = zero_pad_frame(frame, w, h)
                                out5.write(frame)

                        if 'No' in params.keys():
                            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            frame = frame_crop(frame_old,
                                               [y, x],
                                               w, h,
                                               pt_pos='top_left').astype(np.uint8)
                            if verify_img_size(frame,w,h):
                                out.write(frame)
                            else:
                                frame = zero_pad_frame(frame,w,h)
                                out.write(frame)
                    else:
                        break

                # When everything is done, release the video capture object
                # and the video writer object
                cap.release()
                if 'bkg_sub' in params.keys():
                    out1.release()
                if 'clahe' in params.keys():
                    out2.release()
                if all(i in params.keys() for i in ['clahe', 'bkg_sub']):
                    out3.release()
                if 'equalizeHist' in params.keys():
                    out4.release()
                if all(i in params.keys() for i in ['equalizeHist', 'bkg_sub']):
                    out5.release()
                if 'No' in params.keys():
                    out.release()
                    
    end = time.time()
    
    print('Elapsed time during video preprocessing: ', \
          str(int((end-start)/60)), ' minutes and ', str(int((end-start)%60)), ' seconds')


def frame_crop(image, point_coord, w, h, pt_pos='center'):
    """crops an image frame based on two different point references.
       'center' is based on a point at the center of the rectangle
       'top_lef' is based on a point at the top left 
       vertex of the rectangle

    Args:
        image (np.array): 2D or 3D np.array (greyscale or RGB image)
        point_coord (np.array): 1D array with dimension (2,) defining x,y coordinates
        w (float, int): Width of the desired cropped frame
        h (float, int): Height of the desired cropped frame
        pt_pos (str, optional): Reference of the point position ('center' or 'top_left'). Defaults to 'center'.

    Returns:
        (np.array): Cropped section of the 2D or 3D np.array (greyscale or RGB image)
    """    


    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
    if not isinstance(w, int) & isinstance(h, int):
        w = abs(int(w))
        h = abs(int(h))

    if pt_pos == 'center':
        w_start, w_end = [point_coord[1]-int(w/2), point_coord[1]+int(w/2)]
        h_start, h_end = [point_coord[0]-int(h/2), point_coord[0]+int(h/2)]

        if h_start < 0.0:
            h_start = 0
        elif h_end > height:
            h_end = height
        elif w_start < 0.0:
            w_start = 0
        elif w_end > width:
            w_end = width

    elif pt_pos == 'top_left':
        w_start, w_end = [point_coord[1], point_coord[1] + w]
        h_start, h_end = [point_coord[0], point_coord[0] + h]

        if h_start < 0.0:
            h_start = 0
        elif h_end > height:
            h_end = height
        elif w_start < 0.0:
            w_start = 0
        elif w_end > width:
            w_end = width

    if len(image.shape) == 3:
        return image[h_start:h_end, w_start:w_end, :]
    else:
        return image[h_start:h_end, w_start:w_end]


def create_well_template(img_shape, template_type):
    """creates a circle well template to be used in a template matching step. 

    Args:
        img_shape (tuple): The shape of a 2D image. In our case,
            we use the shape of the 2D image of a frame of the video we 
            want to process.  
        template_type (str): the type of well template. 
            Two options available: 'circle' or 'circle_with_halo'.

    Returns:
        (np.array): an 8-bit 2D image of the well template. 
            Only bright well and dark edges atm. 
    """    
    min_dim = np.min(img_shape)
    
    if template_type == 'circle_with_halo':
        tmplt = np.zeros((min_dim,min_dim))
        circle_pixels = draw.circle(int(min_dim/2)-1,int(min_dim/2)-1, int((min_dim/2)*0.99))
        tmplt[circle_pixels[1],circle_pixels[0]] = 125
        circle_pixels = draw.circle(int(min_dim/2)-1,int(min_dim/2)-1, int((min_dim/2)*0.94))
        tmplt[circle_pixels[1],circle_pixels[0]] = 255
        tmplt = tmplt.astype(np.uint8)

    elif template_type == 'circle_with_halo_dark':
        tmplt = np.ones((min_dim,min_dim))*255
        circle_pixels = draw.circle(int(min_dim/2)-1,int(min_dim/2)-1, int((min_dim/2)*0.99))
        tmplt[circle_pixels[1],circle_pixels[0]] = 125
        circle_pixels = draw.circle(int(min_dim/2)-1,int(min_dim/2)-1, int((min_dim/2)*0.94))
        tmplt[circle_pixels[1],circle_pixels[0]] = 0
        tmplt = tmplt.astype(np.uint8)
        
    elif template_type == 'circle':
        tmplt = np.zeros((min_dim,min_dim))
        circle_pixels = draw.circle(int(min_dim/2)-1,int(min_dim/2)-1, int((min_dim/2)*0.99))
        tmplt[circle_pixels[1],circle_pixels[0]] = 255
    
    elif template_type == 'circle_dark':
        tmplt = np.ones((min_dim,min_dim))*255
        circle_pixels = draw.circle(int(min_dim/2)-1,int(min_dim/2)-1, int((min_dim/2)*0.99))
        tmplt[circle_pixels[1],circle_pixels[0]] = 0
    
    return tmplt.astype(np.uint8)


def run_napari_annotations(image, annotation_type='rectangle'):
    """open a napari viewer of an image for annotation of rectangles or points 

    Args:
        image (np.array): a 2D image of a behavioural video with wells 
        annotation_type (str, optional): [description]. Defaults to 'rectangle'.

    Returns:
        (dict): dictionary with keys 'data' and 'shape_type' with the 
            information of the annotated points or rectangles.
    """    
    if annotation_type == 'point':
        with napari.gui_qt():
            viewer = napari.view_image(image)
            annotations = viewer.add_points()

        return annotations.data
    if annotation_type == 'rectangle':
        with napari.gui_qt():
            viewer = napari.view_image(image)
            annotations = viewer.add_shapes()
            
        return  {'data' : annotations.data, 'shape_type' : annotations.shape_type}


def decode_fourcc(cc):
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])


def load_from_json(fpath):
    """ load parameters for video processing from json file.

    Args:
        fpath (str): file path to json file with preprocessing parameters.

    Returns:
        (dict): dictionary with the parsed parameters from the json file.
    """    
    with open(fpath,'r') as read_file:
        params = json.load(read_file)
    
    for key,value in params.items():
        if key == 'equalizeHist':
            params[key] = True
        else:
            for par, val in value.items():
                if isinstance(val, list):
                    params[key][par] = tuple(val)
    return params

