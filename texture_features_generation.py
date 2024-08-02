#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:29:41 2024

@author: rbarella
"""
# Function to process each Gabor filter configuration
def process_gabor_filter(pc, gabor_folder_path, img2, gray_image, image_info):
    import cv2
    import numpy as np
    import os
    from utilities import save_image

    num = pc[0]  # Filter number
    lamda = pc[2]  # Wavelength of the sinusoidal factor
    theta = pc[1]  # Orientation of the normal to the parallel stripes of a Gabor function
    gamma = 0.5  # Spatial aspect ratio
    gabor_label = 'Gabor' + str(num).zfill(2)  # Label for the Gabor filter

    curr_gabor_path = os.path.join(gabor_folder_path, gabor_label + '.tif')
    sigma = 0.56 * lamda  # Standard deviation of the Gaussian function used in the Gabor filter

    # Check if the Gabor filter image already exists
    if not os.path.exists(curr_gabor_path):
        print(gabor_label)

        # Create the Gabor kernel
        ksize = int(8 * sigma + 1)  # Size of the filter
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)

        # Apply the Gabor filter to the image
        fimg = cv2.filter2D(img2, cv2.CV_32F, kernel)
        filtered_img = np.reshape(fimg, np.shape(gray_image))

        # Save the filtered image
        save_image(filtered_img, curr_gabor_path, 'GTiff', 6, image_info['geotransform'], image_info['projection'])
        print(gabor_label + ' SAVED')

# Function to create a VRT (Virtual Dataset) from a list of files
def create_vrt(file_list, band_name_list, resolution=0.25, overwrite=False):
    from osgeo import gdal
    import os

    vrtname = os.path.join(os.path.dirname(file_list[0]), '00_features.vrt')

    # Check if the VRT already exists
    if os.path.exists(vrtname) and not overwrite:
        print(vrtname + ' has already been created')
    else:
        print('Elaborating %s' % vrtname)

        # Command to build the VRT
        file_string = " ".join(file_list)
        cmd = "gdalbuildvrt -separate -r bilinear -tr " + str(resolution) + ' ' + str(resolution) + ' ' + vrtname + ' ' + file_string
        print(cmd)
        os.system(cmd)

        # Set Band Description
        VRT_dataset = gdal.Open(vrtname, gdal.GA_Update)
        for band_name, idx in zip(band_name_list, range(1, len(band_name_list) + 1)):
            VRT_dataset.GetRasterBand(idx).SetDescription(band_name)
        VRT_dataset = None

# Function to generate Gabor features
def gabor_features_generator(vhr_img_path, gabor_params, num_cores):
    from utilities import open_image, save_image
    import os
    from osgeo import gdal
    import cv2
    import numpy as np
    import glob
    from joblib import Parallel, delayed
    import rasterio

    # Open the VHR (Very High Resolution) image
    image, image_info = open_image(vhr_img_path)

    # Convert the image to grayscale and remove the alpha band
    image = np.delete(image, -1, axis=0)
    gray_image = cv2.cvtColor(np.transpose(image, (1, 2, 0)), cv2.COLOR_BGR2GRAY)

    # Reshape the image to 2D
    img2 = gray_image.reshape(-1)

    # Specify the directory to save Gabor features
    gabor_folder_path = os.path.join(os.path.dirname(vhr_img_path), 'Gabor_features')
    if not os.path.exists(gabor_folder_path):
        os.makedirs(gabor_folder_path)
        print('Gabor_features folder created')

    # Initialize lists to hold Gabor filter parameters
    kernels = []
    names_list = []

    # Extract Gabor filter parameters
    theta_list = gabor_params['theta_list']  # List of theta values
    lambda_list = gabor_params['lambda_list']  # List of lambda values
    gamma_list = [0.5]
    number_of_filters = len(theta_list) * len(lambda_list) * len(gamma_list)
    filters_name = ['Gabor' + str(i).zfill(2) for i in np.arange(0, number_of_filters, 1) + 1]

    # Generate a list of all parameter combinations
    param_combinations = [(num, theta, lamda)
                          for num, (theta, lamda)
                          in enumerate([(theta, lamda)
                                        for theta in theta_list
                                        for lamda in lambda_list], start=1)]

    # Parallel computation of Gabor features
    Parallel(n_jobs=num_cores, verbose=1)(delayed(process_gabor_filter)(pc, gabor_folder_path, img2, gray_image, image_info)
                                          for pc in param_combinations)

    # Create feature stack
    name_RGB_list = ['1_R', '2_G', '3_B']

    # Save RGB bands as separate images
    for curr_band, curr_name in zip(range(len(name_RGB_list)), name_RGB_list):
        print(curr_band)
        with rasterio.open(vhr_img_path) as dataset:
            curr_image = dataset.read(curr_band + 1)
            curr_band_path = os.path.join(gabor_folder_path, os.path.basename(vhr_img_path)[:-4] + '_' + curr_name + '.tif')
            save_image(curr_image, curr_band_path, 'GTiff', 1, image_info['geotransform'], image_info['projection'])

    # Create a list of feature paths and names
    features_path_list = sorted(glob.glob(os.path.join(gabor_folder_path, '*.tif')))
    features_name = name_RGB_list + filters_name

    # Create a VRT (Virtual Dataset) from the features
    create_vrt(features_path_list, features_name)
    print('Texture features are saved in:  ' + gabor_folder_path)

    return gabor_folder_path
