import math
import numpy as np

from scipy import ndimage
from scipy.ndimage import gaussian_filter

from sklearn.mixture import GaussianMixture

from skimage import  filters, morphology

from collections import OrderedDict

from matplotlib import pyplot as plt


def extract_Objects_With_Local_Thresholding(image_data, method = 'niblack', window_size=127, k=0.3, post_processing=False):
	
	foreground_mask = np.zeros_like(image_data)

	if method == 'niblack':
		foreground_mask = image_data > filters.threshold_niblack(image_data, window_size=window_size, k=k)
	elif method == 'sauvola':
		foreground_mask = image_data > filters.threshold_sauvola(image_data, window_size=window_size, k=k)
	else:
		raise ValueError("Unknown method. Choose 'niblack' or 'sauvola'.")
	
	# apply post-processing
	if post_processing is True:
		binary_mask = apply_post_processing(foreground_mask, 15)

	return foreground_mask


def extract_Objects_With_Global_Thresholding(image_data, method = 'mean', post_processing=False):
	
	foreground_mask = np.zeros_like(image_data)

	if method == 'mean':
		foreground_mask = image_data > filters.threshold_mean(image_data)
	elif method == 'li':
		foreground_mask = image_data > filters.threshold_li(image_data)
	elif method == 'minimum':
		foreground_mask = image_data > filters.threshold_minimum(image_data)
	elif method == 'otsu':
		foreground_mask = image_data > filters.threshold_otsu(image_data)
	elif method == 'triangle':
		foreground_mask = image_data > filters.threshold_triangle(image_data)
	elif method == 'yen':
		foreground_mask = image_data > filters.threshold_yen(image_data)
	elif method == 'isodata':
		foreground_mask = image_data > filters.threshold_isodata(image_data)
	else:
		raise ValueError("Unknown method. Choose 'niblack' or 'sauvola'.")
	
	# apply post-processing
	if post_processing is True:
		foreground_mask = apply_post_processing(foreground_mask, 15)

	return foreground_mask

def apply_post_processing(binary_mask, min_object_size=15):
	# Fill holes
	binary_mask_fh = ndimage.binary_fill_holes(binary_mask > 0) 

	# Remove salt noise in binary or grayscale images (i.e., bright spots (white pixels) randomly scattered throughout the image)
	se = morphology.disk(1)
	binary_mask_fh_ope = morphology.opening(binary_mask_fh, se)

	# Remove small objects with size <= min_object_size
	binary_mask_final = morphology.remove_small_objects(binary_mask_fh_ope, min_size=min_object_size)

	return binary_mask_final

def try_all_global_methods(image_data, post_processing=False, figsize=(8, 5), num_cols=2):
	methods = OrderedDict(
		{
			'Isodata': extract_Objects_With_Global_Thresholding(image_data, method="isodata", post_processing=post_processing),
			'Li': extract_Objects_With_Global_Thresholding(image_data, method="li", post_processing=post_processing),
			'Mean': extract_Objects_With_Global_Thresholding(image_data, method="mean", post_processing=post_processing),
			'Minimum': extract_Objects_With_Global_Thresholding(image_data, method="minimum", post_processing=post_processing),
			'Otsu': extract_Objects_With_Global_Thresholding(image_data, method="otsu", post_processing=post_processing),
			'Triangle': extract_Objects_With_Global_Thresholding(image_data, method="triangle", post_processing=post_processing),
			'Yen': extract_Objects_With_Global_Thresholding(image_data, method="yen", post_processing=post_processing),
		}
	)

	num_rows = math.ceil((len(methods) + 1.0) / num_cols)

	# Create figure 
	fig = plt.figure(figsize=figsize)
	gs = plt.GridSpec(num_rows, num_cols, figure=fig)
	axes = []
	
	# Plot original image
	ax = fig.add_subplot(gs[0, 0])
	im = ax.imshow(image_data, cmap='gray')
	ax.set_title('Original Image', pad=10, fontsize=12, fontweight='bold')
	ax.axis('off')

	axes.append(ax)
	
	# Plot results for each method
	for i, (name, result) in enumerate(methods.items(), 1):
		row = i // num_cols
		col = i % num_cols
		ax = fig.add_subplot(gs[row, col])
		
		# Plot result
		im = ax.imshow(result, cmap='gray')
			
		# Set title with method name
		ax.set_title(name, pad=10, fontsize=12, fontweight='bold')
		ax.axis('off')
		axes.append(ax)
	
	# Remove empty subplots
	for i in range(len(methods) + 1, num_rows * num_cols):
		ax = fig.add_subplot(gs[i // num_cols, i % num_cols])
		ax.remove()
	
	# Add overall title
	fig.suptitle('Comparison of Global Thresholding Methods', 
				 fontsize=14, fontweight='bold')

	# Adjust layout
	plt.tight_layout()
	
	return fig, axes

def try_all_local_methods(image_data, post_processing=False, figsize=(8, 5), num_cols=2, **kwargs):
	window_size = kwargs.get('window_size', 127)
	k = kwargs.get('k', 0.3)

	methods = OrderedDict(
		{
			'Niblack': extract_Objects_With_Local_Thresholding(image_data, method = 'niblack', window_size=window_size, k=k, post_processing=post_processing),
			'Sauvola': extract_Objects_With_Local_Thresholding(image_data, method = 'sauvola', window_size=window_size, k=k, post_processing=post_processing)
		}
	)

	num_rows = math.ceil((len(methods) + 1.0) / num_cols)

	# Create figure 
	fig = plt.figure(figsize=figsize)
	gs = plt.GridSpec(num_rows, num_cols, figure=fig)
	axes = []
	
	# Plot original image
	ax = fig.add_subplot(gs[0, 0])
	im = ax.imshow(image_data, cmap='gray')
	ax.set_title('Original Image', pad=10, fontsize=12, fontweight='bold')
	ax.axis('off')

	axes.append(ax)
	
	# Plot results for each method
	for i, (name, result) in enumerate(methods.items(), 1):
		row = i // num_cols
		col = i % num_cols
		ax = fig.add_subplot(gs[row, col])
		
		# Plot result
		im = ax.imshow(result, cmap='gray')
			
		# Set title with method name
		ax.set_title(name, pad=10, fontsize=12, fontweight='bold')
		ax.axis('off')
		axes.append(ax)
	
	# Remove empty subplots
	for i in range(len(methods) + 1, num_rows * num_cols):
		ax = fig.add_subplot(gs[i // num_cols, i % num_cols])
		ax.remove()
	
	# Add overall title
	fig.suptitle('Comparison of Local Thresholding Methods', 
				 fontsize=14, fontweight='bold')

	# Adjust layout
	plt.tight_layout()
	
	return fig, axes
	
def separate_background_gmm(image_data, n_components=3, background_label=None, post_processing=False, show_figure=False):
	"""
	Separate background using Gaussian Mixture Model (GMM).
	Good when height distributions are approximately Gaussian.
	
	Parameters:
	-----------
	datafield : numpy.ndarray
		2D array of AFM height data
	n_components : int
		Number of components (usually 2: background + foreground)
	background_label : int
		Label of background, usually 0
	Returns:
	--------
	background_mask : numpy.ndarray
		Binary mask (True for background points)
	foreground_mask : numpy.ndarray
		Binary mask (True for foreground points)
	"""
	# Reshape AFM height data for GMM
	X = image_data.reshape(-1, 1)
	
	# Fit GMM
	gmm = GaussianMixture(n_components=n_components, random_state=42)
	labels = gmm.fit_predict(X)
	
	# Find background component (usually the one with most points)
	unique_labels, label_counts = np.unique(labels, return_counts=True)

	if background_label is None:
		background_label = unique_labels[np.argmax(label_counts)]
	
	# Create boolean masks
	background_mask = (labels.reshape(image_data.shape) == background_label)
	
	foreground_mask = ~background_mask.astype(bool)  
	# apply post-processing
	if post_processing is True:
		foreground_mask = apply_post_processing(foreground_mask, 15)
	
	if show_figure:
		fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), tight_layout=True)
		axarr = axarr.reshape(1, 3)  # Reshape to 2D array
		axarr[0,0].imshow(image_data, cmap='gray')
		axarr[0,0].set_title("Image From JPK Software")
		axarr[0,1].imshow(foreground_mask, cmap='gray')
		axarr[0,1].set_title("Foreground")
		axarr[0,2].imshow(background_mask, cmap='gray')
		axarr[0,2].set_title("Background")
		for ax in axarr.ravel():
			ax.set_axis_off()

		# Add overall title
		fig.suptitle('Segmentation (number of components = ' + str(n_components) + ', and label of background = '+ str(background_label) +')', 
				fontsize=14, fontweight='bold')
		# Regular tight layout
		plt.tight_layout()
		plt.show()
        
	return background_mask.astype(bool), foreground_mask