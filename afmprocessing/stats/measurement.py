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