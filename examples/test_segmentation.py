import context
from afmprocessing.imagedata.segmentation import try_all_methods
from afmprocessing.imagedata.segmentation import extract_Objects_With_Global_Thresholding

import os
from datetime import datetime
import numpy as np 
import matplotlib.pyplot as plt
from skimage import io


if __name__ == "__main__":
    # step 1 - Load AFM TIF image produced by the built-in application of the AFM instrument
    file_folder = 'inputs//test-simple-21062024'
    file_path = os.path.join(file_folder, 'simple-after.tif')
    
    tif_image_data = io.imread(file_path, as_gray=True)

    # try all implemented methods
    fig, ax = try_all_methods(tif_image_data, figsize=(20, 10), num_cols= 5)

    plt.show(block=False)

    # Choose the method which is the best as your observation for final processing. For example, i choose "mean" method
    segmented_mask = extract_Objects_With_Global_Thresholding(tif_image_data, 'mean')

    # visualize
    plt.figure(figsize=(8, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(tif_image_data, cmap='gray')
    plt.title('The AFM image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_mask, cmap='gray')
    plt.title('The segemented mask')
    plt.axis('off')

    plt.show()


    # Save the image as TIF file into an output folder
    # Get the file name without extension
    input_file_name = os.path.splitext(os.path.basename(file_path))[0]
    tags = "test_segmentation"

    output_folder = f'outputs//{input_file_name}-{tags}'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder created: {output_folder}")
    else:
        print(f"Folder already exists: {output_folder}")

    # Save as TIF image in the output folder
    output_file_path = os.path.join(output_folder, f'{input_file_name}-segmentation.tif')
    io.imsave(output_file_path, segmented_mask)

    print(f"Image saved: {output_file_path}")