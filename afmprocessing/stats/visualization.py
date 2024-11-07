import numpy as np
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage import measure
from measurement import analyze_height_distribution

def generate_distinct_colors(num_colors, colormap_name='viridis'):
    """Generate distinct colors using a colormap."""
    cmap = cm.get_cmap(colormap_name)
    if num_colors == 1:
        return [cmap(1)]
    return [cmap(i / (num_colors-1)) for i in range(num_colors)]  
    
def plot_image_with_row_profiles(image_data, height_data, row_indexes, save_path=None, dpi=300):
    # Generate distinct colors
    colors = generate_distinct_colors(len(row_indexes))

    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot the image
    im = ax1.imshow(image_data, cmap='gray')
    ax1.set_title('The AFM image with the Horizontal Profile line')
    
    for idx, color in zip(row_indexes, colors):
        profile = height_data[idx, :]
        x = np.arange(len(profile))
        
        ax1.axhline(y=idx, color=color, linestyle='--')
        
        ax2.plot(x, profile, color=color, linewidth=1.5, label=f'Row {idx}')
        ax2.set_title(f'Horizontal Profile')
        ax2.set_xlabel('Pixel Position')
        ax2.set_ylabel('Height')
        ax2.grid(True)
    
    ax2.legend()
    
    # Adjust layout and display the plot
    plt.tight_layout()

    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save the figure
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

        print(f"Image saved: {save_path}")


    plt.show()    

def plot_image_with_column_profiles(image_data, height_data, column_indexes, save_path=None, dpi=300):
    # Generate distinct colors
    colors = generate_distinct_colors(len(column_indexes))

    # Create the figure with two subplots

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot the image
    im = ax1.imshow(image_data, cmap='gray')
    ax1.set_title('The AFM image with the Vertical Profile line')
    
    for idx, color in zip(column_indexes, colors):
        profile = height_data[:, idx]
        x = np.arange(len(profile))
        
        ax1.axvline(x=idx, color=color, linestyle='--')
        
        ax2.plot(x, profile, color=color, linewidth=1.5, label=f'Row {idx}')
        ax2.set_title(f'Vertical Profile')
        ax2.set_xlabel('Pixel Position')
        ax2.set_ylabel('Height')
        ax2.grid(True)
    
    ax2.legend()
    
    # Adjust layout and display the plot
    plt.tight_layout()

    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save the figure
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

        print(f"Figure saved: {save_path}")


    plt.show(block=False)

def select_intersected_objects_with_a_line(labeled_mask, line_idx, vertical_line=True):
    # Create vertical line mask
    line_mask = np.zeros_like(labeled_mask, dtype=bool)
    if vertical_line is True:
        line_mask[:, line_idx] = True
    else: # Create horizontal line mask
        line_mask[line_idx, :] = True
    
    # Find intersection
    intersection = np.logical_and(labeled_mask, line_mask)
    
    # Find unique labels in the intersection
    intersected_objects = np.unique(labeled_mask[intersection])
    
    # Remove background label (0)
    intersected_objects = intersected_objects[intersected_objects != 0]
    
    return intersected_objects    
    
def show_selected_objects_intersect_with_a_line(image_data, line_idx, selected_objects, labeled_mask, measurements=None, vertical_line=True):    
    # Create highlighted mask for selected objects
    highlight_mask = np.isin(labeled_mask, selected_objects)
    
    # Create the figure and subplots

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original image with vertical line or horizontal line
    ax1.imshow(image_data, cmap='gray')
    if vertical_line is True:
        ax1.axvline(x=line_idx, color='r', linestyle='--') # vertical line
    else: 
        ax1.axhline(y=line_idx, color='r', linestyle='--') # horizontal line   
    ax1.set_title('AFM Image with Vertical Line')
    
    # Plot highlighted objects with measurements
    ax2.imshow(image_data, cmap='gray')
    ax2.imshow(np.ma.masked_where(~highlight_mask, highlight_mask), 
               cmap='spring', alpha=0.3)
    if vertical_line is True:
        ax2.axvline(x=line_idx, color='r', linestyle='--') # vertical line
    else: 
        ax2.axhline(y=line_idx, color='r', linestyle='--') # horizontal line
    ax2.set_title('AFM Image with Selected Objects')
    
    if measurements is not None:
        for obj in measurements:
            y, x = obj['centroid']
            ax2.text(x, y, f"{obj['mean_height']*1e9:.2f} nm", 
                    color='cornflowerblue', ha='left', va='center', fontsize=18)
    
    # Add labels and adjust layout
    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show(block=False)    

def show_height_distribution_of_objects(image_data, labeled_mask, selected_objects, measurements, save_path=None, dpi=300, colormap='tab20'):
    
    colors = generate_distinct_colors(len(selected_objects), colormap)
    heights_ratio = np.ones(len(selected_objects)+1)
    heights_ratio[0]=2

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(len(selected_objects)+1, 2, height_ratios=heights_ratio)

    # Main image subplot (takes up the entire first row)
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.imshow(image_data, cmap='gray')
    height_values = []
    mean_height_values = []
    for label, color in zip(selected_objects, colors):
        for measure in measurements:
            if measure['label'] == label:
                prop = measure
                mask = labeled_mask == label
                ax_main.imshow(np.ma.masked_where(~mask, mask), alpha=0.5, cmap=plt.cm.colors.ListedColormap([color]))
                height_values.append(prop['dist_height'])
                mean_height_values.append(prop['mean_height'])
                break

    ax_main.set_title('The AFM Image with Selected Objects')
    ax_main.axis('off')
    
    for i, (heights, mean_height, color, label) in enumerate(zip(height_values, mean_height_values, colors, selected_objects)):
        ax_hist = fig.add_subplot(gs[i+1, :])
        ax_hist.axvline(x=mean_height, color = 'r', linestyle='--')
        ax_hist.hist(heights, bins=100, color=color, alpha=0.5, edgecolor='black')
        ax_hist.set_xlim((1e-9, 7e-9))
        ax_hist.grid(True)
        if i == 0:
            ax_hist.set_title(f'Distribution of height values of the objects')
        if i == len(height_values) - 1:
            ax_hist.set_xlabel('Pixel Heights')
        if i == np.round(len(height_values)/2):
            ax_hist.set_ylabel('Frequency')

    plt.tight_layout()

    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save the figure
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

        print(f"Figure saved: {save_path}")


    plt.show(block=False) 


def plot_height_distribution(data, n_bins='auto', normalize=False):
    """
    Plot height distribution with statistics similar to Gwyddion.
    """
    # Get distribution data
    dist_data = analyze_height_distribution(data, n_bins, normalize)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot height distribution
    ax1.plot(dist_data['bin_centers'], dist_data['hist_values'], 'b-', lw=2)
    ax1.set_xlabel('Height')
    # ax1.set_xlim([2.75e-7, 3.25e-7])
    if normalize:
        ax1.set_ylabel('Probability Density')
    else:
        ax1.set_ylabel('Count')
    ax1.set_title('Height Distribution')
    
    # Plot original data
    im = ax2.imshow(data)
    ax2.set_title('AFM Data')
    plt.colorbar(im, ax=ax2)
    
    # Add statistics text box
    stats_text = '\n'.join([
        f"Mean: {dist_data['statistics']['mean']:.2f}",
        f"Median: {dist_data['statistics']['median']:.2f}",
        f"RMS: {dist_data['statistics']['rms']:.2f}",
        f"Skewness: {dist_data['statistics']['skew']:.2f}",
        f"Kurtosis: {dist_data['statistics']['kurtosis']:.2f}",
        f"Ra: {dist_data['statistics']['Ra']:.2f}",
        f"Rq: {dist_data['statistics']['Rq']:.2f}"
    ])
    ax1.text(1.05, 0.95, stats_text,
             transform=ax1.transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    plt.show()