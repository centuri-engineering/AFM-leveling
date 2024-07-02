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
    
    
    