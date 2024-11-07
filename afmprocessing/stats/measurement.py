import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from skimage import measure


def measure_objects(height_data, labeled_mask, selected_objects=None):
    props = measure.regionprops(labeled_mask, intensity_image=height_data)
    measurements = []
    
    all_objects = labeled_mask
    
    if selected_objects is not None:
        all_objects = selected_objects
        
    for obj in selected_objects:
        prop = props[obj - 1]  # regionprops labels start from 0
        
        mean_height = prop.intensity_mean
        area = prop.area
        centroid = prop.centroid
        distribution_height = prop.image_intensity.flatten()
        distribution_height =  distribution_height[distribution_height != 0]

        measurements.append({
            'label': obj,
            'mean_height': mean_height,
            'area': area,
            'centroid': centroid,
            'dist_height': distribution_height
        })
    return measurements  
    
    
def analyze_height_distribution(data, n_bins='auto', normalize=True):
    """
    Analyze height distribution similar to Gwyddion's Statistical Functions.
    
    Parameters:
    -----------
    data : numpy.ndarray
        2D array of AFM height data
    n_bins : int or str
        Number of bins for histogram or 'auto' for automatic binning
    normalize : bool
        If True, return probability density (area=1)
        If False, return raw counts
        
    Returns:
    --------
    hist_data : dict
        Dictionary containing:
        - 'bin_centers': Centers of histogram bins
        - 'hist_values': Height distribution values
        - 'statistics': Basic statistical measures
    """
    # Flatten the data
    heights = data.ravel()
    
    # Calculate histogram
    hist, bin_edges = np.histogram(heights, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    if normalize:
        # Convert to probability density
        hist = hist / (len(heights) * (bin_edges[1] - bin_edges[0]))
    
    # Calculate basic statistics
    statistics = {
        'mean': np.mean(heights),
        'median': np.median(heights),
        'rms': np.sqrt(np.mean(np.square(heights))),
        'skew': stats.skew(heights),
        'kurtosis': stats.kurtosis(heights),
        'Ra': np.mean(np.abs(heights - np.mean(heights))),  # Roughness average
        'Rq': np.sqrt(np.mean(np.square(heights - np.mean(heights)))),  # RMS roughness
    }
    
    return {
        'bin_centers': bin_centers,
        'hist_values': hist,
        'statistics': statistics
    }