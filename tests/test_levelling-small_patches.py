import context
from afmprocessing.imagedata.jpktxtfile import read_afm_jpk_txt_file, create_height_image, save_AFM_height_data
from afmprocessing.stats.visualization import show_selected_objects_intersect_with_a_line, select_intersected_objects_with_a_line
from afmprocessing.stats.visualization import plot_image_with_column_profiles, show_height_distribution_of_objects
from afmprocessing.stats.measurement import measure_objects
from afmprocessing.imagedata.segmentation import extract_Objects_With_Global_Thresholding
from afmprocessing.artefacts.scanline   import do_median_of_differences
from afmprocessing.levelling.polynomialsurface import sliding_window_fit_polynomial_surface

import os
from datetime import datetime
import numpy as np 
import matplotlib.pyplot as plt
from skimage import io, measure

if __name__ == "__main__":
    ############################################################################################
    # Step 1: Load raw AFM data 
    # Load TXT file
    file_folder = 'inputs//test-small_patches-21062024'
    file_path = os.path.join(file_folder, 'small_patches-before.txt')

    metadata, height_data = read_afm_jpk_txt_file(file_path)

    image_data = np.flipud(create_height_image(height_data))

    # plt.figure(figsize=(8, 7))
    # plt.imshow(image_data, cmap='gray')
    # plt.title('The AFM image')
    # plt.axis('off')
    # plt.show(block=False)
    ############################################################################################
    # Step 2: Load image and do segmentation
    file_folder = 'inputs//test-small_patches-21062024'
    tif_file_path = os.path.join(file_folder, 'small_patches-after.tif')
    
    tif_image_data = io.imread(tif_file_path, as_gray=True)
    
    segmented_mask = extract_Objects_With_Global_Thresholding(tif_image_data, 'mean')
    
    # # visualize
    # plt.figure(figsize=(8, 7))
    # plt.subplot(1, 2, 1)
    # plt.imshow(tif_image_data, cmap='gray')
    # plt.title('The AFM image')
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.imshow(segmented_mask, cmap='gray')
    # plt.title('The segemented mask')
    # plt.axis('off')

    # plt.show(block=False)
    ############################################################################################
    # Step 3: Removel scan line artefacts
    height_data_correction = do_median_of_differences(np.flipud(height_data)) # scan line artefact correction
    
    image_data_correction = create_height_image(height_data_correction) # show the image before and after the artefact correction

    # plt.figure(figsize=(8, 7))
    # plt.subplot(1, 2, 1)
    # plt.imshow(image_data, cmap='gray')
    # plt.title('The AFM image')
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.imshow(image_data_correction, cmap='gray')
    # plt.title('The AFM image after removing scan line artefact')
    # plt.axis('off')

    # plt.show(block=False)
    ############################################################################################
    # Step 4: Levelling the AFM height data
    mask = segmented_mask.astype(bool)  # Ensure it's a boolean mask
    
    height_data_fitted = sliding_window_fit_polynomial_surface(height_data_correction, mask, window_size=8, poly_order=3) # Fit the polynomial surface
    
    
    height_data_levelled = height_data_correction - height_data_fitted # Calculate the flattened data

    height_data_levelled = np.nan_to_num(height_data_levelled, nan=np.nanmin(height_data_levelled))


    height_data_levelled = height_data_levelled - np.min(height_data_levelled) # shift the minimum to zero

    image_data_levelled = create_height_image(height_data_levelled)
    # Visualize the results
    # fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # axs[0, 0].imshow(image_data_correction, cmap='gray')
    # axs[0, 0].set_title('The AFM image after correction of scan line artefact')

    # axs[0, 1].imshow(segmented_mask, cmap='gray')
    # axs[0, 1].set_title('Foreground')

    # axs[1, 0].imshow(create_height_image(height_data_fitted), cmap='gray')
    # axs[1, 0].set_title('Polynomial Fitted Surface')

    # axs[1, 1].imshow(image_data_levelled, cmap='gray')
    # axs[1, 1].set_title('AFM image after levelling')

    # for ax in axs.ravel():
    #     ax.axis('off')

    # plt.tight_layout()
    # plt.show(block=False)
    ############################################################################################
    # Save the levelled data into an output folder
    # Get the file name without extension
    input_file_name = os.path.splitext(os.path.basename(file_path))[0]
    tags = "test_levelled"

    output_folder = f'outputs//{input_file_name}-{tags}'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder created: {output_folder}")
    else:
        print(f"Folder already exists: {output_folder}")

    # Save the data file as TXT file in the output folder
    output_file_path = os.path.join(output_folder, f'{input_file_name}-levelled.txt')
    save_AFM_height_data(height_data_levelled, output_file_path)
    
    plt.show()
