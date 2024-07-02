import math
import numpy as np
from scipy import ndimage
from skimage import  filters, morphology
from collections import OrderedDict
from matplotlib import pyplot as plt

def extract_Objects_With_Local_Thresholding(image_data, method = 'niblack', window_size=1023, k=0.3, post_processing=True):
	
	binary_mask = np.zeros_like(image_data)

	if method == 'niblack':
		binary_mask = image_data > filters.threshold_niblack(image_data, window_size=window_size, k=k)
	elif method == 'sauvola':
		binary_mask = image_data > filters.threshold_sauvola(image_data, window_size=window_size, k=k)
	else:
		raise ValueError("Unknown method. Choose 'niblack' or 'sauvola'.")
	
	# apply post-processing
	if post_processing is True:
		binary_mask = apply_post_processing(binary_mask, 15)

	return binary_mask


def extract_Objects_With_Global_Thresholding(image_data, method = 'mean', post_processing=True):
	
	binary_mask = np.zeros_like(image_data)

	if method == 'mean':
		binary_mask = image_data > filters.threshold_mean(image_data)
	elif method == 'li':
		binary_mask = image_data > filters.threshold_li(image_data)
	elif method == 'minimum':
		binary_mask = image_data > filters.threshold_minimum(image_data)
	elif method == 'otsu':
		binary_mask = image_data > filters.threshold_otsu(image_data)
	elif method == 'triangle':
		binary_mask = image_data > filters.threshold_triangle(image_data)
	elif method == 'yen':
		binary_mask = image_data > filters.threshold_yen(image_data)
	elif method == 'isodata':
		binary_mask = image_data > filters.threshold_isodata(image_data)
	else:
		raise ValueError("Unknown method. Choose 'niblack' or 'sauvola'.")
	
	# apply post-processing
	if post_processing is True:
		binary_mask = apply_post_processing(binary_mask, 15)

	return binary_mask

def apply_post_processing(binary_mask, min_object_size=15):
	# Fill holes
	binary_mask_fh = ndimage.binary_fill_holes(binary_mask > 0) 

	# Remove salt noise in binary or grayscale images (i.e., bright spots (white pixels) randomly scattered throughout the image)
	se = morphology.disk(1)
	binary_mask_fh_ope = morphology.opening(binary_mask_fh, se)

	# Remove small objects with size <= min_object_size
	binary_mask_final = morphology.remove_small_objects(binary_mask_fh_ope, min_size=min_object_size)

	return binary_mask_final

def try_all_methods(image_data, figsize=(8, 5), num_cols=2):
	methods = OrderedDict(
        {
			'niblack': extract_Objects_With_Local_Thresholding(image_data),
			'sauvola': extract_Objects_With_Local_Thresholding(image_data),
            'Isodata': extract_Objects_With_Global_Thresholding(image_data),
            'Li': extract_Objects_With_Global_Thresholding(image_data),
            'Mean': extract_Objects_With_Global_Thresholding(image_data),
            'Minimum': extract_Objects_With_Global_Thresholding(image_data),
            'Otsu': extract_Objects_With_Global_Thresholding(image_data),
            'Triangle': extract_Objects_With_Global_Thresholding(image_data),
            'Yen': extract_Objects_With_Global_Thresholding(image_data),
        }
    )

	num_rows = math.ceil((len(methods) + 1.0) / num_cols)

	fig, ax = plt.subplots(
        num_rows, num_cols, figsize=figsize, sharex=True, sharey=True
    )

	ax = ax.reshape(-1)

	ax[0].imshow(image_data, cmap='gray')
	ax[0].set_title('The input image')

	i = 1
	for name, func in methods.items():
		# Use precomputed histogram for supporting functions
		ax[i].set_title(name)
		try:
			ax[i].imshow(func, cmap='gray')
		except Exception as e:
			ax[i].text(
				0.5,
				0.5,
				f"{type(e).__name__}",
				ha="center",
				va="center",
				transform=ax[i].transAxes,
			)
		i += 1

	for a in ax:
		a.axis('off')

	fig.tight_layout()

	return fig, ax