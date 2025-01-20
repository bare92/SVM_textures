#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:53:28 2024

@author: rbarella
"""



from skimage.morphology import opening, closing, disk
import rasterio
import matplotlib.pyplot as plt

from scipy.ndimage import label
import numpy as np


from rasterio.transform import from_origin

def save_raster(output_path, data, reference_raster):
    """
    Save a binary map as a raster using rasterio.
    
    Args:
        output_path (str): Path to save the output raster.
        data (ndarray): The processed binary data to save (2D array).
        reference_raster (str): Path to a reference raster to copy metadata.
    """
    # Open the reference raster to copy metadata
    with rasterio.open(reference_raster) as src:
        # Get the transform, CRS, and other metadata
        meta = src.meta.copy()

    # Update metadata for the output
    meta.update({
        "dtype": "uint8",  # Data type for binary maps
        "count": 1,  # Single band
        "nodata": 0  # Specify no-data value
    })

    # Write the output
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(data.astype("uint8"), 1)


def morphological_regularization(binary_map, structuring_element_size=3):
    """
    Regularizes a binary map using morphological opening and closing.
    
    Args:
        binary_map (ndarray): Input binary map (0s and 1s).
        structuring_element_size (int): Size of the structuring element.
    
    Returns:
        ndarray: Regularized binary map.
    """
    struct_elem = disk(structuring_element_size)
    opened_map = opening(binary_map, struct_elem)
    regularized_map = closing(opened_map, struct_elem)
    return regularized_map

def connected_component_regularization(binary_map, size_threshold=100):
    """
    Regularizes a binary map by removing small connected components.

    Args:
        binary_map (ndarray): Input binary map (0s and 1s).
        size_threshold (int): Minimum size for connected components to keep.

    Returns:
        ndarray: Regularized binary map.
    """
    # Ensure the input is binary
    binary_map = (binary_map > 0).astype(int)

    # Label connected components
    labeled_array, num_features = label(binary_map)

    # Get the sizes of each component
    component_sizes = np.bincount(labeled_array.ravel())

    # Create a mask to keep components larger than the threshold
    keep_components = np.where(component_sizes > size_threshold)[0]

    # Create a new binary map where only large components are kept
    regularized_map = np.isin(labeled_array, keep_components).astype(int)

    return regularized_map


def combined_regularization(binary_map, structuring_element_size=3, size_threshold=100):
    """
    Combines morphological operations and connected component filtering.
    
    Args:
        binary_map (ndarray): Input binary map (0s and 1s).
        structuring_element_size (int): Size of the structuring element.
        size_threshold (int): Minimum size for connected components to keep.
    
    Returns:
        ndarray: Regularized binary map.
    """
    morph_map = morphological_regularization(binary_map, structuring_element_size)
    regularized_map = connected_component_regularization(morph_map, size_threshold)
    return regularized_map




image_path = '/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/HYPERSPECTRAL/Downsampled_UAV/Round_06_snow_map_56feat.tif'


with rasterio.open(image_path, 'r') as binary_src:
    
    binary_map = binary_src.read(1)




# Test each method
morph_result = morphological_regularization(binary_map, structuring_element_size=7)
cc_result = connected_component_regularization(binary_map, size_threshold=100)
combined_result = combined_regularization(binary_map, structuring_element_size=3, size_threshold=100)

# Plot results
plt.figure(figsize=(15, 8))
titles = ["Original Map", "Morphological Regularization", 
          "Connected Component Regularization", "Combined Regularization"]
results = [binary_map, morph_result, cc_result, combined_result]

for i, (title, result) in enumerate(zip(titles, results)):
    plt.subplot(2, 2, i + 1)
    plt.title(title)
    plt.imshow(result, cmap="gray")
    plt.axis("off")

plt.tight_layout()
plt.show()


output_path = image_path[:-4] + '_smoothed.tif'

save_raster(output_path, morph_result, image_path)

