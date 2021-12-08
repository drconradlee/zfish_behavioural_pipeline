import numpy as np
import cv2
import os
import sys
from align_coordinates import order_coordinates
from sort_wells import sort_wells

def detect_and_export(file_list,examine_frame,min_well,max_well,min_pad,target_num_well,row_thr):
    num_wells = []
    all_coord =[]
    for video in np.arange(np.shape(file_list)[0]):
        cap = cv2.VideoCapture(file_list[video])
        cap.set(1,examine_frame)
        ret, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_num_well = []
        while current_num_well != target_num_well:
            circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                                    param1=50,param2=30,minRadius=min_well,maxRadius=max_well)
            circles = np.uint16(np.around(circles))
            current_num_well = np.shape(circles[0])[0]
            if current_num_well == target_num_well:
                break
            elif current_num_well != target_num_well:
                max_well = (int(np.round(np.mean(circles[0][:,2])))+10)
                min_well = (int(np.round(np.mean(circles[0][:,2])))-10)
        num_wells.append(np.shape(circles[0,:,0]))

        print('File #' + str(video+1) + '-       Number of wells detected: ' + str(num_wells[video][0]))
        sorted_circles = sort_wells(circles,row_thr)
        
        for idx,i in enumerate(sorted_circles):
            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
            cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
            cv2.putText(frame,'Well: ' +str(idx+1), (i[0]-40,i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0))
        img_out_name = os.path.splitext(file_list[video])[0] + '_well_detection.jpg'
        cv2.imwrite(img_out_name, frame)
        if num_wells[video][0] > target_num_well:
            sys.exit("Too Many Wells detected! Check Pixel Range for well-detection!")
        if num_wells[video][0] < target_num_well:
            sys.exit("No Enough Wells detected! Check Pixel Range for well-detection!")

        coord_lines = np.zeros((num_wells[video][0],4))
        for well in range(num_wells[video][0]):
            coord_lines[well][0] = (max(sorted_circles[:,2])+min_pad)*2 # Width (the biggest radius found + padding)x2
            coord_lines[well][1] = (max(sorted_circles[:,2])+min_pad)*2 # Height (the biggest radius found + padding)x2
            coord_lines[well][2] = sorted_circles[well][0]-(max(sorted_circles[:,2])+min_pad) # Top left X
            coord_lines[well][3] = sorted_circles[well][1]-(max(sorted_circles[:,2])+min_pad) # Top left Y  
        all_coord.append(coord_lines)

    return num_wells,all_coord