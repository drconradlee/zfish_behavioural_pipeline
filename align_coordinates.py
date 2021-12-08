import numpy as np

def order_coordinates(num_wells,circles):
    well_order = np.arange(num_wells).reshape(int(np.sqrt(num_wells)),int(np.sqrt(num_wells)))
    row_idx = np.argsort(circles[0][:,0])
    sorted_row_coord = circles[0][row_idx]

    col_idx = []
    all_sorted_coord = np.zeros((num_wells,4))
    for i, value in enumerate(np.arange(0,np.sqrt(num_wells))):  # Loops thru each row (0-3)
        col_idx_temp = np.argsort(sorted_row_coord[well_order[i],1]) + i*np.sqrt(num_wells) # At that row, grab the x-values, and sorts them as an idx and adds row value 
        col_idx.append(col_idx_temp)
    col_idx = np.reshape(col_idx,(1,num_wells))[0]
    sorted_circles = sorted_row_coord[col_idx.astype(int)]

    return sorted_circles