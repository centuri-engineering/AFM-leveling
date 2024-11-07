import numpy as np
from scipy.optimize import lsq_linear
from scipy.interpolate import RectBivariateSpline, LinearNDInterpolator

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from afmprocessing.artefacts.scanline import do_median_of_differences

from numpy.polynomial.chebyshev import chebvander2d, chebval2d
from numpy.polynomial.legendre import legvander2d, legval2d

def fit_polynomial_surface(datafield, x_order=1, y_order=1, mask=None):
    """
    Surface fitting using "traditional" method.
    """
    y, x = np.mgrid[0:datafield.shape[0], 0:datafield.shape[1]]
    
    X_full = x.ravel()
    Y_full = y.ravel()

    if mask is not None:
        X = x[mask]
        Y = y[mask]
        Z = datafield[mask]
    else:
        X = x.ravel()
        Y = y.ravel()
        Z = datafield.ravel()

    # Calculate the polynomial terms
    poly_terms = cal_polynomial_terms(X, Y, x_order=x_order, y_order=y_order)
    
    # Perform the linear fitting 
    result = lsq_linear(poly_terms, Z) # Using scipy.optimize.lsq_linear
    coeffs = result.x 
    # coeffs = np.linalg.lstsq(poly_terms, Z, rcond=None)[0] # Using numpy.linalg.lstsq

    # Evaluate the fitted polynomial
    result = cal_polynomial_terms(X_full, Y_full, x_order=x_order, y_order=y_order) @ coeffs
    
    # Create the fitted surface
    Z_fitted = result.reshape(datafield.shape)
    
    return Z_fitted

def cal_polynomial_terms(x, y, x_order, y_order):
    """
    Calculate polynomial terms
    
    Parameters:
    x, y: numpy arrays of shape (n_samples,)
    x_order: maximum power for x terms
    y_order: maximum power for y terms
    
    Returns:
    terms: numpy array of shape (n_samples, (x_order+1)*(y_order+1))
    """
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    x_powers = np.arange(x_order + 1)
    y_powers = np.arange(y_order + 1)
    
    x_terms = x ** x_powers  
    y_terms = y ** y_powers 
    
    x_terms_expanded = x_terms[:, :, np.newaxis]  
    y_terms_expanded = y_terms[:, np.newaxis, :]  
    
    terms = x_terms_expanded * y_terms_expanded  # Shape: (n_samples, x_order+1, y_order+1)
    
    return terms.reshape(len(x), -1)


def fit_polynomial_surface_using_regression_model(datafield, poly_degree=3, mask=None, model_type='linear', **model_params):
    """
    Surface fitting by fitting regression models on polynomial features of the input data.

    Parameters:
    -----------
    datafield : numpy.ndarray
        2D array containing the AFM height data
    poly_degree : int
        Degree of the polynomial surface (default=2)
    mask : numpy.ndarray, optional
        Boolean array of the same shape as `AFM_im` indicating background-like pixels.
        `True` values indicate pixels used for model fitting.
    model_type : str
        Type of regression model to use:
        'linear' - Standard LinearRegression
        'ridge' - Ridge regression with L2 regularization
        'lasso' - Lasso regression with L1 regularization
        'elastic' - ElasticNet combining L1 and L2 regularization
        'huber' - HuberRegressor for robustness to outliers
        'svr' - Support Vector Regression
        'rf' - Random Forest Regression
    """
    y, x = np.mgrid[0:datafield.shape[0], 0:datafield.shape[1]]

    if mask is not None:
        X = np.column_stack((x[mask], y[mask]))
        Z = datafield[mask] 
    else: 
        X = np.column_stack((x.ravel(), y.ravel()))
        Z = datafield.ravel()

    # Compute polynomial features from coordinates
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Standardize polynomial features
    scaler = StandardScaler()
    X_poly_standardized = scaler.fit_transform(X_poly)

    # Initialize the regression model
    models = {
        'linear': LinearRegression(),
        'ridge': Ridge(alpha=1.2),
        'lasso': Lasso(alpha=1.2),
        'elastic': ElasticNet(alpha=1.0, l1_ratio=0.75),
        'huber': HuberRegressor(epsilon=1.35),
        'svr': LinearSVR(),
        'rf': RandomForestRegressor(n_estimators=100)
    }
    model = models[model_type]
    if model_params:
        model.set_params(**model_params)

    # Fit the model to the standardized polynomial features 
    model.fit(X_poly_standardized, Z)

    # Create the fitted surface
    X_poly_full = poly.transform(np.column_stack((x.ravel(), y.ravel())))
    X_poly_full_standardized = scaler.transform(X_poly_full)

    # Evaluate the fitted polynomial features
    X_poly_full = poly.transform(np.column_stack((x.ravel(), y.ravel())))
    X_poly_full_standardized = scaler.transform(X_poly_full)

    # Create the fitted surface
    Z_fitted = model.predict(X_poly_full_standardized).reshape(datafield.shape)
    if model_type == 'svr':
        return do_median_of_differences(Z_fitted)

    return Z_fitted


def fit_polynomial_surface_using_basis_function(datafield, x_order=1, y_order=1, mask=None, function_type='legendre'):
    """
    Surface fitting using tensor product of basis functions such as legendre and chebyshev.
    """
    y, x = np.mgrid[0:datafield.shape[0], 0:datafield.shape[1]]
    
    X_full = x.ravel()
    Y_full = y.ravel()

    if mask is not None:
        X = x[mask]
        Y = y[mask]
        Z = datafield[mask]
    else:
        X = x.ravel()
        Y = y.ravel()
        Z = datafield.ravel()

    if function_type == 'legendre':
        # Calculate the Chebyshev polynomial terms
        poly_terms = chebvander2d(X, Y, [x_order, y_order])
    else:
        # Calculate the Chebyshev polynomial terms
        poly_terms = legvander2d(X, Y, [x_order, y_order])
    
    # Perform the linear fitting 
    # result = lsq_linear(poly_terms, Z) # Using scipy.optimize.lsq_linear
    # coeffs = result.x 
    coeffs = np.linalg.lstsq(poly_terms.reshape(-1, poly_terms.shape[-1]), Z, rcond=None)[0] # Using numpy.linalg.lstsq
    
    # Evaluate the fitted polynomial
    if function_type == 'legendre':
        result = legval2d(X_full, Y_full, coeffs.reshape(x_order + 1, y_order + 1))
    else:
        result = chebval2d(X_full, Y_full, coeffs.reshape(x_order + 1, y_order + 1))
    
    # Create the fitted surface
    Z_fitted = result.reshape(datafield.shape)
    
    return Z_fitted


def fit_surface_using_random_forest_regression(datafield, mask=None, **model_params):
    """
    Fit using Random Forest Regression with normalized data
    
    Parameters
    ----------
    Z0 : 2D numpy array
        Input surface data
    mask : 2D numpy array, optional
        Mask where True values will be used for fitting
    n_estimators : int, default=100
        Number of trees in the forest
    max_depth : int, optional
        Maximum depth of the trees
        
    Returns
    -------
    pipeline : sklearn Pipeline
        Fitted model pipeline including scaler
    Z2 : 2D numpy array
        Fitted surface
    """
    y, x = np.mgrid[0:datafield.shape[0], 0:datafield.shape[1]]

    if mask is not None:
        # Interpolate invalid points using valid data
        valid_coords = np.argwhere(mask)  # Get coordinates of valid points
        invalid_coords = np.argwhere(~mask)  # Get coordinates of invalid points
        interpolator = LinearNDInterpolator(valid_coords, datafield[mask])
        
        # Fill in interpolated values
        Z = datafield.copy()
        Z[~mask] = interpolator(invalid_coords)
    else:
        Z = datafield
    
    scaler = StandardScaler()
    Z_scaled = scaler.fit_transform(Z.reshape(-1, 1)).ravel()

    # Prepare features and target
    features = np.column_stack([x.ravel(), y.ravel()])
    target = Z_scaled.ravel()
    
    # initialize the random forest regression model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    if model_params:
        rf.set_params(model_params)
    # Fit the model to height data    
    rf.fit(features, target)

    # Predict on the data
    grid_features = np.column_stack([x.ravel(), y.ravel()])
    Z2 = rf.predict(grid_features)

    # unscale the predicted surface
    z_pred = scaler.inverse_transform(Z2.reshape(-1, 1)).ravel()
    
    # Reshape predicted surface
    Z2_rescaled = z_pred.reshape(datafield.shape)

    return Z2_rescaled


def fit_bspline_surface(datafield, x_order=3, y_order=3, mask=None):
   # Create coordinate arrays
    y_coords = np.arange(datafield.shape[0])
    x_coords = np.arange(datafield.shape[1])
    
    # Create meshgrid for output points
    Y, X = np.mgrid[0:datafield.shape[0], 0:datafield.shape[1]]
    
    if mask is not None:
        # Interpolate invalid points using valid data
        valid_coords = np.argwhere(mask)  # Get coordinates of valid points
        invalid_coords = np.argwhere(~mask)  # Get coordinates of invalid points
        interpolator = LinearNDInterpolator(valid_coords, datafield[mask])
        
        # Fill in interpolated values
        grid_z = datafield.copy()
        grid_z[~mask] = interpolator(invalid_coords)
    else:
        grid_z = datafield

    # Fit the B-spline surface using the full coordinate arrays
    spline = RectBivariateSpline(
        x=y_coords,      # Full y coordinates
        y=x_coords,      # Full x coordinates
        z=grid_z,        # Full grid of z values
        kx=x_order,
        ky=y_order
    )
    
    # Evaluate the spline on the full grid
    Z_fitted = spline.ev(Y.ravel(), X.ravel())
    
    return Z_fitted.reshape(datafield.shape)
        

def gradient_based_tilt_correction(datafield, mask=None):
    """
    Perform gradient-based tilt correction on AFM height data.
    """
    # Compute gradients
    grad_y, grad_x = np.gradient(datafield)
    
    # Compute mean gradients of valid pixels
    if mask is not None:
        mean_grad_x = np.mean(grad_x[mask])
        mean_grad_y = np.mean(grad_y[mask])
    else:
        mean_grad_x = np.mean(grad_x)
        mean_grad_y = np.mean(grad_y)
    
    # Create coordinate grids
    y, x = np.mgrid[0:datafield.shape[0], 0:datafield.shape[1]]
    
    # Compute the tilt plane
    tilt_plane = mean_grad_x * x + mean_grad_y * y
    
    return tilt_plane