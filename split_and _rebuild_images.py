#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:02:32 2024

@author: rbarella
"""



import rasterio
import numpy as np
import os
from osgeo import gdal
import subprocess
import glob

def open_image(image_path):
    """
    Open a georeferenced image and retrieve its data and metadata.

    :param image_path: Path to the georeferenced image file.
    :return: A tuple containing the image array and a dictionary of metadata information.
    """
    # Open the georeferenced image using GDAL
    image = gdal.Open(image_path)

    # Get the number of columns (width) and rows (height) of the image
    cols = image.RasterXSize
    rows = image.RasterYSize

    # Get the geotransformation parameters
    geotransform = image.GetGeoTransform()

    # Get the projection information
    proj = image.GetProjection()

    # Calculate the extents of the image
    minx = geotransform[0]
    maxy = geotransform[3]
    maxx = minx + geotransform[1] * cols
    miny = maxy + geotransform[5] * rows

    # Store the size of the raster in terms of columns and rows
    X_Y_raster_size = [cols, rows]

    # Store the extents in a list
    extent = [minx, miny, maxx, maxy]

    # Create a dictionary to hold the geotransformation, extent, size, and projection information
    information = {}
    information['geotransform'] = geotransform
    information['extent'] = extent
    information['X_Y_raster_size'] = X_Y_raster_size
    information['projection'] = proj

    # Read the image data as a NumPy array
    image_array = np.array(image.ReadAsArray(0, 0, cols, rows))

    return image_array, information

    image = gdal.Open(image_path)

    cols = image.RasterXSize
    rows = image.RasterYSize
    geotransform = image.GetGeoTransform()
    proj = image.GetProjection()
    minx = geotransform[0]
    maxy = geotransform[3]
    maxx = minx + geotransform[1] * cols
    miny = maxy + geotransform[5] * rows
    X_Y_raster_size = [cols, rows]
    extent = [minx, miny, maxx, maxy]
    information = {}
    information['geotransform'] = geotransform
    information['extent'] = extent
    information['X_Y_raster_size'] = X_Y_raster_size
    information['projection'] = proj
    image_array = np.array(image.ReadAsArray(0, 0, cols, rows))

    return image_array, information;

def calculate_image_extents(big_image_extent, rows, cols):
    """
    Calculate the extents for subimages to split a big image.

    :param big_image_extent: Tuple with the extent (xmin, ymin, xmax, ymax) of the big image.
    :param rows: Number of rows to split the big image into.
    :param cols: Number of columns to split the big image into.
    :return: List of tuples with extents (xmin, ymin, xmax, ymax) for each subimage.
    """
    xmin, ymin, xmax, ymax = big_image_extent
    subimage_width = (xmax - xmin) / cols
    subimage_height = (ymax - ymin) / rows

    extents = []
    for row in range(rows):
        for col in range(cols):
            sub_xmin = xmin + col * subimage_width
            sub_xmax = sub_xmin + subimage_width
            sub_ymin = ymin + row * subimage_height
            sub_ymax = sub_ymin + subimage_height
            extents.append((sub_xmin, sub_ymin, sub_xmax, sub_ymax))

    return extents

def split_georeferenced_image(input_file, output_dir, rows, cols):
    """
    Split a georeferenced image into subimages and save them.

    :param input_file: Path to the input georeferenced image.
    :param output_dir: Directory where the subimages will be saved.
    :param rows: Number of rows to split the image into.
    :param cols: Number of columns to split the image into.
    """
    # Open the georeferenced image
    with rasterio.open(input_file) as src:
        width = src.width // cols
        height = src.height // rows
        transform = src.transform

        for i in range(rows):
            for j in range(cols):
                window = rasterio.windows.Window(
                    col_off=j * width,
                    row_off=i * height,
                    width=width,
                    height=height
                )

                transform_window = rasterio.windows.transform(window, transform)
                
                out_profile = src.profile.copy()
                out_profile.update({
                    "height": height,
                    "width": width,
                    "transform": transform_window
                })
                
                out_file = os.path.join(output_dir, f"subimage_{i}_{j}.tif")
                
                with rasterio.open(out_file, "w", **out_profile) as dst:
                    dst.write(src.read(window=window))
                    
                print(f"Saved subimage {i},{j} to {out_file}")

def build_vrts(image_extents, image_files, output_dir, features_name, resolution):
    """
    Build VRTs for given image extents and list of images.

    :param image_extents: List of tuples with the extents (xmin, ymin, xmax, ymax) for each VRT.
    :param image_file: List of Input image file to be included in the VRTs.
    :param output_dir: Directory where the VRT files will be saved.
    """
    
    file_string = " ".join(image_files)
    
    for idx, extent in enumerate(image_extents):
        xmin, ymin, xmax, ymax = extent
        vrtname = os.path.join(output_dir, f"00_features_{idx+1}.vrt")
        
        
        
        cmd = "gdalbuildvrt -separate -r bilinear -tr " + ' '.join([str(resolution), str(resolution)]) + ' -te ' + ' '.join([str(xmin), str(ymin), str(xmax), str(ymax)]) + ' ' + vrtname + ' ' + file_string
        print(cmd)
        os.system(cmd)
        
        # Set Band Description
    
        VRT_dataset = gdal.Open(vrtname, gdal.GA_Update)
        for band_name, idx in zip(features_name, range(1, len(features_name) + 1)):
            VRT_dataset.GetRasterBand(idx).SetDescription(band_name)
    
        VRT_dataset = None
        print(f"Saved VRT {idx+1} to {vrtname}")

def merge_images(image_paths, output_path):
    """
    Merge multiple georeferenced images into a single GeoTIFF file.

    :param image_paths: List of paths to the input georeferenced images.
    :param output_path: Path to the output merged GeoTIFF file.
    """
    # Open the input images
    input_images = [gdal.Open(image) for image in image_paths]

    # Use GDAL's gdal.Warp to merge the images
    gdal.Warp(output_path, input_images, format='GTiff')

    print(f"Merged image saved to {output_path}")



# Example usage:
input_file = '/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/HYPERSPECTRAL/Downsampled_UAV/20240704_Cervinia_snow_patches_ortho_25cm_orto_32632.tif'
output_dir = os.path.join(os.path.dirname(input_file), 'Subimages_56feat')


gabor_folder_path = os.path.join(os.path.dirname(input_file), '0_pca_Gabor_features_56_filters')


theta_list = [theta / 4. * np.pi for theta in range(4)]  #Define number of thetas. Here only 2 theta values 0 and 1/4 . pi 

lambda_list = np.arange(2, 16, 4) #Range of wavelengths
gamma_list = [0.5]
number_of_filters = len(theta_list)  * len(lambda_list) * len(gamma_list)
filters_name = ['Gabor' + str(i).zfill(2) for i in np.arange(0, number_of_filters, 1) + 1]

# Generate a list of all parameter combinations
param_combinations = [(num, theta, lamda) 
                      for num, (theta, lamda) 
                      in enumerate([(theta, lamda) 
                                    for theta in theta_list 
                                   
                                    for lamda in lambda_list], start=1)]

rows, cols = 4, 4  # For example, split into 4 rows and 4 columns

big_image_extent = open_image(input_file)[1]['extent']

resolution = open_image(input_file)[1]['geotransform'][1]

image_extents = calculate_image_extents(big_image_extent, rows, cols)

for i, extent in enumerate(image_extents):
    print(f"Subimage {i+1} extent: {extent}")


# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)


# Create feature stack
name_RGB_list = ['1_R', '2_G', '3_B']


features_path_list = sorted(glob.glob(os.path.join(gabor_folder_path, '*.tif')))

features_name = name_RGB_list + filters_name

build_vrts(image_extents, features_path_list, output_dir, features_name, resolution)



###############################################################################
stop
## Now classify the images with SVM

###############################################################################

### Image merging part

#classified_img_folder = os.path.join(os.path.dirname(input_file), 'Classified_subimages')
classified_img_folder = '/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/HYPERSPECTRAL/UAV_SVM_texture/Classified_image_model4'

# Example usage:
image_paths = sorted(glob.glob(os.path.join(classified_img_folder, '*snow_map')))

output_path = os.path.join('/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/HYPERSPECTRAL/Downsampled_UAV', '_'.join(image_paths[0].split('_')[-4:]) + '.tif')

merge_images(image_paths, output_path)










