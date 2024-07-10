import numpy as np

def fit_polynomial_surface(height_data, mask=None, poly_order=3):
    # Create a grid of x and y coordinates
    y, x = np.mgrid[0:height_data.shape[0], 0:height_data.shape[1]]
    
    # Flatten the arrays
    x = x.flatten()
    y = y.flatten()
    z = height_data.flatten()
    
    valid = z
    # Create a mask to exclude the main objects if any
    if mask is not None:
        valid = ~mask.flatten()
    
    # Function to calculate the polynomial terms
    def cal_polynomial_terms(x, y, poly_order):
        terms = []
        for i in range(poly_order + 1):
            for j in range(poly_order + 1 - i):
                terms.append(x**i * y**j)
        return np.vstack(terms).T
    
    # Calculate the polynomial terms
    A = cal_polynomial_terms(x[valid], y[valid], poly_order)
    
    # Perform the least squares fit
    coeffs, _, _, _ = np.linalg.lstsq(A, z[valid], rcond=None)
    
    # Function to evaluate the fitted polynomial
    def eval_polynomial_surface(x, y, coeffs):
        return np.dot(cal_polynomial_terms(x, y, poly_order), coeffs)
    
    # Create the fitted surface
    fitted_surface = eval_polynomial_surface(x, y, coeffs).reshape(height_data.shape)
    
    return fitted_surface, coeffs
    
    
 def sliding_window_fit_polynomial_surface(image, mask, window_size=8, poly_order=3):
    def local_fitting(window, mask_window):
        y, x = np.mgrid[:window_size, :window_size]
        y = y.flatten()
        x = x.flatten()
        z = window.flatten()
        mask_wd = mask_window.flatten()
        
        # Only consider points where the mask is True
        valid_points = mask_wd > 0
        y = y[valid_points]
        x = x[valid_points]
        z = z[valid_points]
        
        if len(z) < (poly_order + 1) * (poly_order + 2) // 2:
            return np.nan  # Not enough points to fit
        
        # Construct the design matrix
        A = np.column_stack([np.ones(len(z)), x, y, x**2, x*y, y**2])
        if poly_order > 2:
            A = np.column_stack([A, x**3, x**2*y, x*y**2, y**3])
        
        # Fit the polynomial
        coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        
        # Return the fitted value at the center of the window
        center = window_size // 2
        center_A = np.array([1, center, center, center**2, center**2, center**2])
        if poly_order > 2:
            center_A = np.append(center_A, [center**3, center**3, center**3, center**3])
        return np.dot(center_A, coeffs)

    # Pad the image and mask to handle edges
    pad_width = window_size // 2
    padded_image = np.pad(image, pad_width, mode='reflect')
    padded_mask = np.pad(mask, pad_width, mode='constant', constant_values=0)
    
    # Initialize the output data
    fitted_image = np.full_like(image, np.nan, dtype=float)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if mask[i, j] > 0:  # Only process pixels in the segmented area
                window = padded_image[i:i+window_size, j:j+window_size]
                mask_window = padded_mask[i:i+window_size, j:j+window_size]
                fitted_image[i, j] = local_fitting(window, mask_window)
    
    return fitted_image    