import context
from afmprocessing.imagedata.jpktxtfile import read_afm_jpk_txt_file, create_height_image
from afmprocessing.stats.visualization import plot_image_with_row_profiles

import os
from datetime import datetime
import numpy as np 
import matplotlib.pyplot as plt
from skimage import io

if __name__ == "__main__":
    # Load AFM jpk text file and create AFM image from raw data
    file_folder = 'inputs//test-simple-21062024'
    file_path = os.path.join(file_folder, 'simple-before.txt')

    metadata, height_data = read_afm_jpk_txt_file(file_path)

    image_data = np.flipud(create_height_image(height_data))

    

    # Create output folder from the name of the input AFM data file
    input_file_name = os.path.splitext(os.path.basename(file_path))[0]
    tags = "test_showImageWithRowProfile"

    output_folder = f'outputs//{input_file_name}-{tags}'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder created: {output_folder}")
    else:
        print(f"Folder already exists: {output_folder}")

    # Save figure in the output folder
    output_file_path = os.path.join(output_folder, f'figure-{input_file_name}-With_Profile.tif')
    
    
    # Show the profile of list of rows and save it as tif file in the output folder
    plot_image_with_row_profiles(image_data, np.flipud(height_data), [100, 101,102, 127,128,129, 114,115,116, 150], output_file_path)
 
