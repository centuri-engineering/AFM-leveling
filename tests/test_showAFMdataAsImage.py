import context
from afmprocessing.imagedata.jpktxtfile import read_afm_jpk_txt_file, create_height_image

import os
from datetime import datetime
import numpy as np 
import matplotlib.pyplot as plt
from skimage import io

if __name__ == "__main__":
    # step 1 - Load AFM jpk text file and show the image
    file_folder = 'inputs//test-simple-21062024'
    file_path = os.path.join(file_folder, 'simple-before.txt')

    metadata, height_data = read_afm_jpk_txt_file(file_path)

    image_data = np.flipud(create_height_image(height_data))

    # Display the image
    plt.imshow(image_data, cmap='gray')
    plt.title('The RAW AFM image')
    plt.axis('off')
    plt.show()

    # Save the image as TIF file into an output folder
    # Get the file name without extension
    input_file_name = os.path.splitext(os.path.basename(file_path))[0]
    tags = "test_showImage"

    output_folder = f'outputs//{input_file_name}-{tags}'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder created: {output_folder}")
    else:
        print(f"Folder already exists: {output_folder}")

    # Save as TIF image in the output folder
    output_file_path = os.path.join(output_folder, f'{input_file_name}.tif')
    io.imsave(output_file_path, image_data)

    print(f"Image saved: {output_file_path}")
