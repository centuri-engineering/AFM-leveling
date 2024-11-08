import numpy as np
from scipy import ndimage, signal
from skimage import restoration
from matplotlib import pyplot as plt


def calculate_snr(datafield, method='rms'):
	"""
	Calculate the Signal-to-Noise Ratio (SNR) of AFM height data.
	
	Parameters:
	datafield (numpy.ndarray): 2D array of height values
	method (str): Method to use for SNR calculation. 'rms', 'peak', or 'histogram'
	
	Returns:
	float: Calculated SNR value
	"""	
	if method == 'rms':
		# RMS method
		signal = np.sqrt(np.mean(np.square(datafield)))
		noise = np.std(datafield)
		snr = 20 * np.log10(signal / noise)
	
	elif method == 'peak':
		# Peak-to-peak method
		signal = np.max(datafield) - np.min(datafield)
		noise = np.std(datafield)
		snr = 20 * np.log10(signal / noise)
	
	elif method == 'histogram':
		# Histogram method
		hist, bin_edges = np.histogram(datafield, bins='auto')
		peak_height = np.max(hist)
		background = np.mean(hist)
		snr = 10 * np.log10(peak_height / background)
	
	else:
		raise ValueError("Method must be 'rms', 'peak', or 'histogram'")
	
	return snr

def estimate_noise_params(datafield, snr):
	"""
	Estimate noise parameters from the SNR.
	
	Parameters:
	data (numpy.ndarray): 2D array of height values
	snr (float): Signal-to-Noise Ratio in dB

	Returns:
	dict: Estimated noise parameters (sigma, mean, std)
	"""
	
	# Convert SNR from dB to linear scale
	snr_linear = 10**(snr / 20)
	
	# Estimate signal power (assuming RMS method was used for SNR calculation)
	signal_power = np.mean(np.square(datafield))
	
	# Estimate noise power
	noise_power = signal_power / (snr_linear**2)
	
	# Estimate noise parameters
	noise_sigma = np.sqrt(noise_power)
	noise_mean = np.mean(datafield)
	noise_std = np.std(datafield)
	
	return {
		'sigma': noise_sigma,
		'mean': noise_mean,
		'std': noise_std
	}


def denoise_wiener(datafield, noise_params, **kwargs):
	"""
	Apply Wiener filter based on SNR.
	"""
	window_size = kwargs.get('window_size', 3)

	# Estimate noise power from SNR
	noise_power = noise_params['sigma']**2
	
	# Apply Wiener filter
	denoised_datafield = signal.wiener(datafield, mysize = window_size, noise=noise_power)
	
	return denoised_datafield


def denoise_gaussian(datafield, noise_params, **kwargs):
	"""
	Apply Gaussian filter based on SNR.
	"""
	sigma_factor = kwargs.get('sigma_factor', 1.0)*noise_params['sigma']
	
	# Apply Gaussian filter
	denoised_datafield = ndimage.gaussian_filter(datafield, sigma=sigma_factor)
	
	return denoised_datafield


def denoise_nlm(datafield, noise_params, **kwargs):
	"""
	Apply Non-local means denoising based on SNR.
	"""
	patch_size = kwargs.get('patch_size', 5)

	patch_distance = kwargs.get('patch_distance', 6)

	h = kwargs.get('h_factor', 1.0) * noise_params['sigma']
	
	denoised_datafield = restoration.denoise_nl_means(datafield,
									h=h,
									sigma=noise_params['sigma'],
									patch_size=patch_size,
									patch_distance=patch_distance,
									fast_mode=kwargs.get('fast_mode', True))

	return denoised_datafield


def denoise_wavelet(datafield, noise_params, **kwargs):
	"""
	Apply Wavelet denoising based on SNR.
	"""
	sigma_est = kwargs.get('sigma_factor', 1.0) * noise_params['sigma']
	wavelet = kwargs.get('wavelet', 'db4')
	wavelet_levels = kwargs.get('level', 3)
	method = kwargs.get('wavelet_method', 'BayesShrink')

	denoised_datafield = restoration.denoise_wavelet(datafield,
									sigma=sigma_est,
									mode='soft',
									wavelet=wavelet,
									wavelet_levels=wavelet_levels,
									method=method)
	return denoised_datafield


def reduce_noise_based_on_snr(datafield, noise_params=None, method='weiner', show_figure=False, **kwargs):
	"""
	Reduce noise in AFM data.
	
	Parameters:
	datafield (numpy.ndarray): 2D array of height values
	method (str): Noise reduction method 
	**kwargs: Additional arguments for the chosen method
	
	Returns:
	numpy.ndarray: Noise-reduced data
	"""
	
	if noise_params is None:
		snr = calculate_snr(datafield)
		noise_params = estimate_noise_params(datafield, snr)
	
	if method == 'wiener':
		denoised_datafield = denoise_wiener(datafield, noise_params, **kwargs)
	elif method == 'gaussian':
		denoised_datafield = denoise_gaussian(datafield, noise_params, **kwargs)
	elif method == 'nlm':	
		denoised_datafield = denoise_nlm(datafield, noise_params, **kwargs)
	elif method == 'wavelet':
		denoised_datafield = denoise_wavelet(datafield, noise_params, **kwargs)
	else:
		raise ValueError("Unknown denoising method")
	
	if show_figure:
		fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(15, 6), tight_layout=True)
		axarr = axarr.reshape(1, 2)  # Reshape to 2D array
		axarr[0,0].imshow(datafield, cmap='gray')
		axarr[0,0].set_title("Original Data")
		# Original statistics
		orig_snr = calculate_snr(datafield, method='rms')
		stats_text = (f'SNR: {orig_snr:.2f} dB')
		axarr[0,0].text(0.02, 0.98, stats_text,
				transform=axarr[0,0].transAxes,
				verticalalignment='top',
				fontsize=8,
				bbox=dict(facecolor='white', alpha=0.8))
	
		axarr[0,1].imshow(denoised_datafield, cmap='gray')
		axarr[0,1].set_title("Denoised Data")
		
		# Denoised data statistics
		snr_metrics = calculate_snr_after_noise_reduction(datafield, denoised_datafield)
		if snr_metrics['validity']:
			stats_text = (f'SNR: {snr_metrics["snr_denoised"]:.2f} dB\n'
						f'Improvement: {snr_metrics["improvement"]:.2f} dB\n'
						f'Noise Reduction: {snr_metrics["noise_reduction"]:.2f} dB')
		else:
			stats_text = (f'SNR calculation invalid\n'
						f'No Improvement nor Noise Reduction')
		axarr[0,1].text(0.02, 0.98, stats_text,
			transform=axarr[0,1].transAxes,
			verticalalignment='top',
			fontsize=8,
			bbox=dict(facecolor='white', alpha=0.8))	
		
		for ax in axarr.ravel():
			ax.set_axis_off()
		# Add overall title
		fig.suptitle(str(method) + ' denoising', 
				fontsize=14, fontweight='bold')
		# Regular tight layout
		plt.tight_layout()
		plt.show()

	return denoised_datafield


def calculate_snr_after_noise_reduction(original_data, denoised_data, method='rms'):
	"""
	Calculate SNR for both original and denoised data.
	
	Parameters:
	-----------
	original_data : numpy.ndarray
		Original AFM data
	denoised_data : numpy.ndarray
		Denoised AFM data
	method : str
		Method for SNR calculation
		
	Returns:
	--------
	dict : SNR values and improvement metrics
	"""
	# Small value to prevent division by zero
	epsilon = np.finfo(float).eps
	
	# Calculate noise as the difference between original and denoised
	noise = original_data - denoised_data

	noise_std = np.std(noise)

	# Check if there's any actual denoising
	if noise_std < epsilon:
		return {
			'snr_original': 0,
			'snr_denoised': 0,
			'improvement': 0,  # No improvement if data unchanged
			'noise_reduction': 0,  # No noise reduction if data unchanged
			'original_std': np.std(original_data),
			'noise_std': 0,
			'validity': False,
			'message': 'No denoising detected - original and denoised data are identical'
		}
	
	if method == 'rms':
		# RMS implementation
		signal = np.sqrt(np.mean(np.square(denoised_data)))
		noise_power = noise_std
		snr_denoised = 20 * np.log10(signal / noise_power)
		
		signal_orig = np.sqrt(np.mean(np.square(original_data)))
		noise_orig = np.std(original_data)
		snr_original = 20 * np.log10(signal_orig / noise_orig)
		
		# Calculate improvement and noise reduction
		improvement = snr_denoised - snr_original
		noise_reduction = 20 * np.log10(noise_orig / noise_std)
		
	elif method == 'peak':
		# Peak implementation
		signal = np.max(denoised_data) - np.min(denoised_data)
		noise_power = noise_std
		snr_denoised = 20 * np.log10(signal / noise_power)
		
		signal_orig = np.max(original_data) - np.min(original_data)
		noise_orig = np.std(original_data)
		snr_original = 20 * np.log10(signal_orig / noise_orig)
		
		# Calculate improvement and noise reduction
		improvement = snr_denoised - snr_original
		noise_reduction = 20 * np.log10(noise_orig / noise_std)
		
	elif method == 'histogram':
		# Histogram implementation
		hist_orig, _ = np.histogram(original_data, bins='auto')
		hist_denoised, _ = np.histogram(denoised_data, bins='auto')
		
		peak_orig = np.max(hist_orig)
		background_orig = np.mean(hist_orig)
		snr_original = 10 * np.log10(peak_orig / background_orig)
		
		peak_denoised = np.max(hist_denoised)
		background_denoised = np.mean(hist_denoised)
		snr_denoised = 10 * np.log10(peak_denoised / background_denoised)
		
		# Calculate improvement
		improvement = snr_denoised - snr_original
		# For histogram method, noise reduction is calculated differently
		noise_reduction = 10 * np.log10(background_orig / background_denoised)
	
	return {
		'snr_original': snr_original,
		'snr_denoised': snr_denoised,
		'improvement': improvement,
		'noise_reduction': noise_reduction,
		'original_std': np.std(original_data),
		'noise_std': noise_std,
		'validity': True,
		'message': 'Valid calculation'
	}


def compare_denoising_methods_based_on_snr(datafield, noise_params):
	"""
	Compare different denoising methods with proper SNR calculation.
	"""
	# Create figure with GridSpec
	fig = plt.figure(figsize=(15, 10))
	gs = plt.GridSpec(2, 3, figure=fig)
	axes = []
	
	# Original data
	ax = fig.add_subplot(gs[0, 0])
	im0 = ax.imshow(datafield)
	ax.set_title('Original Data', pad=10, fontsize=12, fontweight='bold')
	ax.axis('off')
	axes.append(ax)
	# Original statistics
	orig_snr = calculate_snr(datafield, method='rms')
	stats_text = (f'SNR: {orig_snr:.2f} dB')
	ax.text(0.02, 0.98, stats_text,
			transform=ax.transAxes,
			verticalalignment='top',
			fontsize=8,
			bbox=dict(facecolor='white', alpha=0.8))
	
	# Wiener filter
	ax = fig.add_subplot(gs[0, 1])
	denoised_wiener = reduce_noise_based_on_snr(datafield, noise_params, 
									  method='wiener', window_size=5)
	im1 = ax.imshow(denoised_wiener)
	ax.set_title('Wiener Filter', pad=10, fontsize=12, fontweight='bold')
	ax.axis('off')
	axes.append(ax)
	# Wiener statistics
	snr_metrics = calculate_snr_after_noise_reduction(datafield, denoised_wiener)
	if snr_metrics['validity']:
		stats_text = (f'SNR: {snr_metrics["snr_denoised"]:.2f} dB\n'
					 f'Improvement: {snr_metrics["improvement"]:.2f} dB\n'
					 f'Noise Reduction: {snr_metrics["noise_reduction"]:.2f} dB')
	else:
		stats_text = (f'SNR calculation invalid\n'
					 f'No Improvement nor Noise Reduction')

	ax.text(0.02, 0.98, stats_text,
			transform=ax.transAxes,
			verticalalignment='top',
			fontsize=8,
			bbox=dict(facecolor='white', alpha=0.8))
	

	# Gaussian filter
	ax = fig.add_subplot(gs[0, 2])
	denoised_gaussian = reduce_noise_based_on_snr(datafield, noise_params, 
										method='gaussian', sigma_factor=1.0)
	im1 = ax.imshow(denoised_gaussian)
	ax.set_title('Gaussian Filter', pad=10, fontsize=12, fontweight='bold')
	ax.axis('off')
	axes.append(ax)
	# Gaussian filter statistics
	snr_metrics = calculate_snr_after_noise_reduction(datafield, denoised_gaussian)
	if snr_metrics['validity']:
		stats_text = (f'SNR: {snr_metrics["snr_denoised"]:.2f} dB\n'
					 f'Improvement: {snr_metrics["improvement"]:.2f} dB\n'
					 f'Noise Reduction: {snr_metrics["noise_reduction"]:.2f} dB')
	else:
		stats_text = (f'SNR calculation invalid\n'
					 f'No Improvement nor Noise Reduction')

	ax.text(0.02, 0.98, stats_text,
			transform=ax.transAxes,
			verticalalignment='top',
			fontsize=8,
			bbox=dict(facecolor='white', alpha=0.8))
	
	# Wavelet denoising
	ax = fig.add_subplot(gs[1, 0])
	denoised_wavelet = reduce_noise_based_on_snr(datafield, noise_params, 
									   method='wavelet', wavelet='db4', level=3)
	im1 = ax.imshow(denoised_wavelet)
	ax.set_title('Wavelet denoising', pad=10, fontsize=12, fontweight='bold')
	ax.axis('off')
	axes.append(ax)
	# Wavelet statistics
	snr_metrics = calculate_snr_after_noise_reduction(datafield, denoised_wavelet)
	if snr_metrics['validity']:
		stats_text = (f'SNR: {snr_metrics["snr_denoised"]:.2f} dB\n'
					 f'Improvement: {snr_metrics["improvement"]:.2f} dB\n'
					 f'Noise Reduction: {snr_metrics["noise_reduction"]:.2f} dB')
	else:
		stats_text = (f'SNR calculation invalid\n'
					 f'No Improvement nor Noise Reduction')

	ax.text(0.02, 0.98, stats_text,
			transform=ax.transAxes,
			verticalalignment='top',
			fontsize=8,
			bbox=dict(facecolor='white', alpha=0.8))
	
	# Non-local means
	ax = fig.add_subplot(gs[1, 1])
	denoised_nlm = reduce_noise_based_on_snr(datafield, noise_params, 
								   method='nlm', patch_size=5, h_factor=1.0)
	im1 = ax.imshow(denoised_nlm)
	ax.set_title('Non-local means', pad=10, fontsize=12, fontweight='bold')
	ax.axis('off')
	axes.append(ax)
	# Non-local means statistics
	snr_metrics = calculate_snr_after_noise_reduction(datafield, denoised_nlm)
	if snr_metrics['validity']:
		stats_text = (f'SNR: {snr_metrics["snr_denoised"]:.2f} dB\n'
					 f'Improvement: {snr_metrics["improvement"]:.2f} dB\n'
					 f'Noise Reduction: {snr_metrics["noise_reduction"]:.2f} dB')
	else:
		stats_text = (f'SNR calculation invalid\n'
					 f'No Improvement nor Noise Reduction')

	ax.text(0.02, 0.98, stats_text,
			transform=ax.transAxes,
			verticalalignment='top',
			fontsize=8,
			bbox=dict(facecolor='white', alpha=0.8))
	
	# Add comparison plot
	ax = fig.add_subplot(gs[1, 2])
	# Compare SNR improvements
	methods = ['Wiener', 'Gaussian', 'Wavelet', 'NLM']
	improvements = [
		calculate_snr_after_noise_reduction(datafield, denoised_wiener)['improvement'],
		calculate_snr_after_noise_reduction(datafield, denoised_gaussian)['improvement'],
		calculate_snr_after_noise_reduction(datafield, denoised_wavelet)['improvement'],
		calculate_snr_after_noise_reduction(datafield, denoised_nlm)['improvement']
	]
	
	ax.bar(methods, improvements)
	ax.set_title('SNR Improvement by Method', pad=10, fontsize=12, fontweight='bold')
	ax.set_ylabel('SNR Improvement (dB)')
	plt.setp(ax.get_xticklabels(), rotation=45)
	
	# Add overall title
	fig.suptitle('Comparison of SNR-based Denoising Techniques', 
				 fontsize=14, fontweight='bold')
	
	# Adjust layout
	plt.tight_layout()
	
	plt.show()


def denoise_median(datafield, window_size = 3, show_figure=False):
	"""
	Apply Median filtering 
	"""
	denoised_datafield = ndimage.median_filter(datafield, size=window_size)

	if show_figure:
		fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(15, 6), tight_layout=True)
		axarr = axarr.reshape(1, 2)  # Reshape to 2D array
		axarr[0,0].imshow(datafield, cmap='gray')
		axarr[0,0].set_title("Original Data")
		# Original statistics
		orig_snr = calculate_snr(datafield, method='rms')
		stats_text = (f'SNR: {orig_snr:.2f} dB')
		axarr[0,0].text(0.02, 0.98, stats_text,
				transform=axarr[0,0].transAxes,
				verticalalignment='top',
				fontsize=8,
				bbox=dict(facecolor='white', alpha=0.8))
	
		axarr[0,1].imshow(denoised_datafield, cmap='gray')
		axarr[0,1].set_title("Denoised Data")
		
		# Denoised data statistics
		snr_metrics = calculate_snr_after_noise_reduction(datafield, denoised_datafield)
		if snr_metrics['validity']:
			stats_text = (f'SNR: {snr_metrics["snr_denoised"]:.2f} dB\n'
						f'Improvement: {snr_metrics["improvement"]:.2f} dB\n'
						f'Noise Reduction: {snr_metrics["noise_reduction"]:.2f} dB')
		else:
			stats_text = (f'SNR calculation invalid\n'
						f'No Improvement nor Noise Reduction')
		axarr[0,1].text(0.02, 0.98, stats_text,
			transform=axarr[0,1].transAxes,
			verticalalignment='top',
			fontsize=8,
			bbox=dict(facecolor='white', alpha=0.8))	
		
		for ax in axarr.ravel():
			ax.set_axis_off()
		# Add overall title
		fig.suptitle('Denoising using Median Filtering \n (window size = ' + str(window_size) + ')', 
				fontsize=14, fontweight='bold')
		# Regular tight layout
		plt.tight_layout()
		plt.show()
	
	return denoised_datafield


def denoise_tv_minimization(datafield, weight=0.5, show_figure=False):
	"""
	Apply total variation minimization.
	
	Parameters:
	- datafield: AFM height data
	- weight: Regularization parameter; higher values reduce more variation.
	"""
	denoised_datafield = restoration.denoise_tv_chambolle(datafield, weight=weight)

	if show_figure:
		fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(15, 6), tight_layout=True)
		axarr = axarr.reshape(1, 2)  # Reshape to 2D array
		axarr[0,0].imshow(datafield, cmap='gray')
		axarr[0,0].set_title("Original Data")
		# Original statistics
		orig_snr = calculate_snr(datafield, method='rms')
		stats_text = (f'SNR: {orig_snr:.2f} dB')
		axarr[0,0].text(0.02, 0.98, stats_text,
				transform=axarr[0,0].transAxes,
				verticalalignment='top',
				fontsize=8,
				bbox=dict(facecolor='white', alpha=0.8))
	
		axarr[0,1].imshow(denoised_datafield, cmap='gray')
		axarr[0,1].set_title("Denoised Data")
		
		# Denoised data statistics
		snr_metrics = calculate_snr_after_noise_reduction(datafield, denoised_datafield)
		if snr_metrics['validity']:
			stats_text = (f'SNR: {snr_metrics["snr_denoised"]:.2f} dB\n'
						f'Improvement: {snr_metrics["improvement"]:.2f} dB\n'
						f'Noise Reduction: {snr_metrics["noise_reduction"]:.2f} dB')
		else:
			stats_text = (f'SNR calculation invalid\n'
						f'No Improvement nor Noise Reduction')
		axarr[0,1].text(0.02, 0.98, stats_text,
			transform=axarr[0,1].transAxes,
			verticalalignment='top',
			fontsize=8,
			bbox=dict(facecolor='white', alpha=0.8))	
		
		for ax in axarr.ravel():
			ax.set_axis_off()
		# Add overall title
		fig.suptitle('Denoising total variation minimization \n (weight = ' + str(weight) + ')', 
				fontsize=14, fontweight='bold')
		# Regular tight layout
		plt.tight_layout()
		plt.show()
	
	return denoised_datafield