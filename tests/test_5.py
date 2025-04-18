import context

####################
# Import modules here 
import os
import numpy as np
import matplotlib.pyplot as plt

from skimage import io, util

from afmprocessing.imagedata.jpktxtfile import read_afm_jpk_txt_file, create_height_image, save_AFM_height_data
from afmprocessing.imagedata.segmentation import extract_Objects_With_Global_Thresholding
from afmprocessing.imagedata.refinement import zero_min_value, remove_scars
from afmprocessing.artefacts.scanline import do_median_of_differences, do_trimmed_mean_of_differences
from afmprocessing.leveling.leveling import fit_surface_using_random_forest_regression
from afmprocessing.stats.visualization import plot_height_distribution




if __name__ == "__main__":
    
    ########### Put your codes here ###################
    
    #########################################
    ### Step 1.1 - Load AFM jpk text file ###
    file_folder = 'inputs//test_5'
    file_path = os.path.join(file_folder, 'case_2.txt')
    metadata, height_data = read_afm_jpk_txt_file(file_path)
    image_data = np.flipud(create_height_image(height_data)) # create an image from raw height data for visualization purpose
    
    ##########################################
    ### Step 1.2 - Load AFM image TIF file ###
    tif_file_path = os.path.join(file_folder, 'case_2.tif')
    
    tif_image_data = io.imread(tif_file_path, as_gray=True)

    
    
    #########################################################################################
    ### Step 2 - Do data pre-processing such as denoising, and scan line artefact removal ###
    height_data_correction = do_median_of_differences(np.flipud(height_data), method='linear')
    
    #### Plot to verify the scan line artefact correction
    # f, axarr = plt.subplots(nrows=1, ncols=2, figsize=(8, 8), tight_layout=True)
    # axarr = axarr.reshape(1, 2)  # Reshape to 2D array
    # axarr[0,0].imshow(image_data, cmap='gray')
    # axarr[0,0].set_title("Original Image")
    # axarr[0,1].imshow(create_height_image(height_data_correction), cmap='gray')
    # axarr[0,1].set_title("Image after scan line artifact removal")
    # for ax in axarr.ravel():
    #     ax.set_axis_off()
    # plt.show()
    
    
    ##########################################################
    ### Step 3 - Segmentation of foreground and background ###
    segmented_mask, background_mask = extract_Objects_With_Global_Thresholding(tif_image_data, method='mean', show_figure=False)
    
    
    ##############################
    ### Step 4 - Data leveling ###
    height_data_correction_fitted_surface = fit_surface_using_random_forest_regression(height_data_correction, mask=background_mask) # use background mask to compute the fitted surface
    
    height_data_correction_leveled = height_data_correction - height_data_correction_fitted_surface

    # #############################################################################################
    # ### Step 5 (Optional) Do data post-processing such as removing scars, and remove outliers ###
    # height_data_correction_leveled = do_trimmed_mean_of_differences(height_data_correction_leveled,  trim_fraction=0.18)
    height_data_correction_0min = zero_min_value(height_data_correction_leveled)
    
    height_data_correction_0min = remove_scars(height_data_correction_0min, threshold=0.5, show_figure=True)

    # Plot the results
    f, axarr = plt.subplots(nrows=1, ncols=2, figsize=(10, 10), tight_layout=True)
    axarr = axarr.reshape(1, 2)  # Reshape to 2D array
    axarr[0,0].imshow(create_height_image(height_data_correction_fitted_surface), cmap='gray')
    axarr[0,0].set_title("Fitted surface")
    axarr[0,1].imshow(create_height_image(height_data_correction_0min), cmap='gray')
    axarr[0,1].set_title("Leveled AFM data")
    plt.show()

    # # Plot height distribution of leveled data
    fig = plot_height_distribution(height_data_correction_0min)
    # ############################################################################################
    # ### Step 6 (Optional) Save all results and figures in a subfolder of the folder "output" ###
    input_file_name = os.path.splitext(os.path.basename(file_path))[0]
    tags_folder = "test-standard_workflow"
    tags_name = "20241108"

    output_folder = f'outputs//{os.path.split(file_folder)[-1]}-{input_file_name}-{tags_folder}'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder created: {output_folder}")
    else:
        print(f"Folder already exists: {output_folder}")

    # Save the data file as TXT file in the output folder
    output_file_path = os.path.join(output_folder, f'{input_file_name}-{tags_name}-leveled.txt')
    save_AFM_height_data(height_data_correction_0min, output_file_path)
    
    # Save segmentation results
    output_file_path = os.path.join(output_folder, f'{input_file_name}-{tags_name}-foreground.tif')
    io.imsave(output_file_path, util.img_as_ubyte(segmented_mask))
    output_file_path = os.path.join(output_folder, f'{input_file_name}-{tags_name}-background.tif')
    io.imsave(output_file_path, util.img_as_ubyte(background_mask))
    
    # Save statistical analysis of height distribution
    output_file_path = os.path.join(output_folder, f'{input_file_name}-{tags_name}-height_distribution.jpg')
    fig.savefig(output_file_path,
            dpi=300,
            bbox_inches='tight',     # removes extra white space
            pad_inches=0.1,          # adds small padding
            format='jpg',
            transparent=True)        # transparent background