from functools import reduce
import numpy as np

def sort_wells(circles,row_thr):
    x = circles[0][:,0]
    y = circles[0][:,1]
    idx = np.arange(np.shape(x)[0])
    well_order = []
    for i in np.arange(np.shape(x)[0]):
        current_y = y[idx]
        current_x = x[idx]
        if len(current_y)<1:
            break
        current_row_choice = np.logical_and(np.greater(current_y,(min(current_y)-row_thr)), np.less(current_y,(min(current_y)+row_thr)))
        current_row_wells = idx[current_row_choice]
        well_order.append(list(current_row_wells[np.argsort(current_x[current_row_choice])]))
        idx = np.delete(idx,current_row_choice)
        
    well_list = reduce(lambda x,y: x+y, well_order)
    sorted_circles = circles[0][well_list]

    return sorted_circles