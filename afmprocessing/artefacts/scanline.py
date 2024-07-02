import numpy as np
from scipy import polyfit, polyval, signal

def do_median_of_differences(datafield, method='linear', **kwargs):
    yres, xres = datafield.shape
    medians = np.zeros(yres)
    
    for i in range(yres - 1):
        row = datafield[i, :]
        next_row = datafield[i + 1, :]
        diff = next_row - row
        
        if diff.size > 0:
            median_diff = np.median(diff)
        else:
            median_diff = 0
        
        medians[i + 1] = median_diff + medians[i]
    
    medians = slope_level_rowShifts(medians, method)
    result = apply_row_shifts(datafield, medians)
    
    return result

def slope_level_rowShifts(medians, method='linear', **kwargs):
    x = np.arange(len(medians))
    
    if method == 'linear':
        # Linear detrending
        p = np.polyfit(x, medians, 1)
        trend = np.polyval(p, x)
    
    elif method == 'polynomial':
        # Polynomial detrending
        poly_order = kwargs.get('poly_order', 3)
        p = np.polyfit(x, medians, poly_order)
        trend = np.polyval(p, x)
    
    elif method == 'savgol':
        # Savitzky-Golay filter
        window_size = kwargs.get('window_size', 32)
        poly_order = kwargs.get('poly_order', 3)
        trend = signal.savgol_filter(medians, window_size, poly_order)
    
    elif method == 'moving_average':
        # Moving average
        window_size = kwargs.get('window_size', 32)
        trend = np.convolve(medians, np.ones(window_size)/window_size, mode='same')
        
        # Handle edge effects
        half_window = window_size // 2
        trend[:half_window] = trend[half_window]
        trend[-half_window:] = trend[-half_window-1]
    
    else:
        raise ValueError("Unknown method. Choose 'linear', 'polynomial', 'savgol', or 'moving_average'.")
    
    detrended_medians = medians - trend
    return detrended_medians

def apply_row_shifts(datafield, medians):
    # Apply the calculated shifts to each row
    result = datafield.copy()
    for i in range(datafield.shape[0]):
        result[i, :] = result[i, :] - medians[i]
    
    return result
    
    