import context
from afmprocessing.imagedata.jpktxtfile import read_afm_jpk_txt_file, create_height_image, save_AFM_height_data
from afmprocessing.stats.visualization import show_selected_objects_intersect_with_a_line, select_intersected_objects_with_a_line
from afmprocessing.stats.visualization import plot_image_with_column_profiles, show_height_distribution_of_objects
from afmprocessing.stats.measurement import measure_objects

import os
from datetime import datetime
import numpy as np 
import matplotlib.pyplot as plt
from skimage import io, measure

if __name__ == "__main__":
    # Step 1: Load AFM data after levelling
    # Load TXT file
    file_folder = 'inputs//test-simple-21062024'
    file_path = os.path.join(file_folder, 'simple-before-levelled.txt')

    metadata, height_data_levelled = read_afm_jpk_txt_file(file_path)

    image_data_levelled = create_height_image(height_data_levelled)

    # plt.figure(figsize=(8, 7))
    # plt.imshow(image_data_levelled, cmap='gray')
    # plt.title('The AFM image')
    # plt.axis('off')
    # plt.show(block=False)


    # Step 2: Load segmented image 
    file_folder = 'inputs//test-simple-21062024'
    tif_file_path = os.path.join(file_folder, 'simple-after-segmentation.tif')
    
    segmented_mask = io.imread(tif_file_path, as_gray=True)
    
    # Label the objects in mask to make sure all objects have same labels through the test
    labeled_mask = measure.label(segmented_mask) 

    # Step 3: Show the (mean) height of objects across a vertical line
    line_num = 173
    selected_objects = select_intersected_objects_with_a_line(labeled_mask, line_num) # select objects across the vertical line

    # selected_objects = np.unique(labeled_mask) # in case of selecting all objects found in the image
    #selected_objects = np.array([1,4,5,6,9]) # in case of selecting customed objects

    measurements = measure_objects(height_data_levelled, labeled_mask, selected_objects) # get basic measurements for selected objects

    print(f"Objects intersected by the line at x={line_num}: {selected_objects}")
    print(f"Total number of objects: {np.max(labeled_mask)}")

    # Save the levelled data into an output folder
    # Get the file name without extension
    input_file_name = os.path.splitext(os.path.basename(file_path))[0]
    tags = "test_measure_height"

    output_folder = f'outputs//{input_file_name}-{tags}'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder created: {output_folder}")
    else:
        print(f"Folder already exists: {output_folder}")

    show_selected_objects_intersect_with_a_line(image_data_levelled, line_num, selected_objects, labeled_mask, measurements)
    
    # Save measurement in file
    output_file_path = os.path.join(output_folder, f'{input_file_name}-distribution_of_heights.tif')
    show_height_distribution_of_objects(image_data_levelled, labeled_mask, selected_objects, measurements, output_file_path, colormap='tab20')

    plt.show()  # This will keep all figures open

