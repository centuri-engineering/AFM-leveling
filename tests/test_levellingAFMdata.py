import context
from afmprocessing.imagedata.jpktxtfile import read_afm_jpk_txt_file, create_height_image, save_AFM_height_data
from afmprocessing.artefacts.scanline   import do_median_of_differences
from afmprocessing.imagedata.segmentation import extract_Objects_With_Global_Thresholding
from afmprocessing.levelling.polynomialsurface import fit_polynomial_surface
from afmprocessing.stats.visualization import plot_image_with_row_profiles, plot_image_with_column_profiles


import os
from datetime import datetime
import numpy as np 
import matplotlib.pyplot as plt
from skimage import io

if __name__ == "__main__":
    # Step 1: Load AFM data and do scan line artefact correction 
    # Load AFM jpk text file 
    file_folder = 'inputs//test-simple-21062024'
    file_path = os.path.join(file_folder, 'simple-before.txt')

    metadata, height_data = read_afm_jpk_txt_file(file_path)

    image_data = np.flipud(create_height_image(height_data))
  
    height_data_correction = do_median_of_differences(np.flipud(height_data)) # scan line artefact correction
    
    image_data_correction = create_height_image(height_data_correction) # show the image before and after the artefact correction

    plt.figure(figsize=(8, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(image_data, cmap='gray')
    plt.title('The AFM image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image_data_correction, cmap='gray')
    plt.title('The AFM image after removing scan line artefact')
    plt.axis('off')

    plt.show(block=False)

    # Step 2: Extract foreground
    ## For more details, take a look the file 'test_segmentation.py'
    # Load AFM TIF image produced by the built-in application of the AFM instrument
    file_folder = 'inputs//test-simple-21062024'
    tif_file_path = os.path.join(file_folder, 'simple-after.tif')
    
    tif_image_data = io.imread(tif_file_path, as_gray=True)
    
    segmented_mask = extract_Objects_With_Global_Thresholding(tif_image_data, 'mean') # Do segmentation    
    

    # Step 3: Levelling with exclusion of the foreground
    mask = segmented_mask.astype(bool)  # Ensure it's a boolean mask
    
    data_fitted, coeffs = fit_polynomial_surface(height_data_correction, mask, poly_order=3) # Fit the polynomial surface
    
    data_levelled = height_data_correction - data_fitted # Calculate the flattened data

    # Visualize the results
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    axs[0, 0].imshow(image_data_correction, cmap='gray')
    axs[0, 0].set_title('The AFM image after correction of scan line artefact')

    axs[0, 1].imshow(segmented_mask, cmap='gray')
    axs[0, 1].set_title('Foreground')

    axs[1, 0].imshow(create_height_image(data_fitted), cmap='gray')
    axs[1, 0].set_title('Polynomial Fitted Surface')

    axs[1, 1].imshow(create_height_image(data_levelled), cmap='gray')
    axs[1, 1].set_title('AFM image after levelling')

    for ax in axs.ravel():
        ax.axis('off')

    plt.tight_layout()

    
    # show height profile in veritcal and horizontal after levelling
    line_num = 128
    plot_image_with_column_profiles(create_height_image(data_levelled), data_levelled, [line_num]) # show the column profiles
    plot_image_with_row_profiles(create_height_image(data_levelled), data_levelled, [line_num]) # show the column profiles
    plt.show()


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
    save_AFM_height_data(data_levelled, output_file_path)

    print(f"The levelled data is saved: {output_file_path}")

    