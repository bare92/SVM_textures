#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:29:41 2024

@author: rbarella
"""
from utilities import save_image, open_image

def rescale_array(array):
    """
    Rescale a numpy array to the range [0, 255].

    Parameters:
    - array: Input numpy array (can be any shape)

    Returns:
    - rescaled_array: Rescaled numpy array with values in the range [0, 255]
    """
    # Get the minimum and maximum values of the array
    
    import numpy as np
    
    min_val = np.min(array)
    max_val = np.max(array)

    # Rescale the array to the range [0, 255]
    rescaled_array = 255 * (array - min_val) / (max_val - min_val)

    # Convert to unsigned 8-bit integer (uint8)
    rescaled_array = rescaled_array.astype(np.uint8)

    return rescaled_array


# Function to process each Gabor filter configuration
def process_gabor_filter(pc, gabor_folder_path, img2, gray_image, image_info, no_data_val):
    import cv2
    import numpy as np
    import os
    
    
    current_directory = os.getcwd()
    print("Current Working Directory:", current_directory)

    num = pc[0]  # Filter number
    lamda = pc[2]  # Wavelength of the sinusoidal factor
    theta = pc[1]  # Orientation of the normal to the parallel stripes of a Gabor function
    gamma = 0.5  # Spatial aspect ratio
    gabor_label = 'Gabor' + str(num).zfill(2)  # Label for the Gabor filter
    
    kernels = []
    
    curr_gabor_path = os.path.join(gabor_folder_path, gabor_label + '.tif')
    sigma = 0.56 * lamda  # Standard deviation of the Gaussian function used in the Gabor filter
    
    # Create the Gabor kernel
    ksize = int(8 * sigma + 1)  # Size of the filter
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)

    # Check if the Gabor filter image already exists
    if not os.path.exists(curr_gabor_path):
        print(gabor_label)

        

        # Apply the Gabor filter to the image
        fimg = cv2.filter2D(gray_image, cv2.CV_32F, kernel)
        filtered_img = np.reshape(fimg, np.shape(gray_image))
        filtered_img[gray_image==0] = no_data_val

        # Save the filtered image
        save_image(filtered_img, curr_gabor_path, 'GTiff', 6, image_info['geotransform'], image_info['projection'], NoDataValue=no_data_val)
        print(gabor_label + ' SAVED')
        
    return (gabor_label, kernel);

# Function to create a VRT (Virtual Dataset) from a list of files
def create_vrt(file_list, band_name_list, filename_features, no_data_val, resolution=None, overwrite=False):
    from osgeo import gdal
    import os

    vrtname = os.path.join(os.path.dirname(file_list[0]), filename_features)
    
    # Check if the VRT already exists
    if os.path.exists(vrtname) and not overwrite:
        print(vrtname + ' has already been created')
    else:
        print('Elaborating %s' % vrtname)

        # Command to build the VRT
        file_string = " ".join(file_list)
        cmd = "gdalbuildvrt -separate -r bilinear -tr " + str(resolution) + ' ' + str(resolution) +\
            ' -srcnodata ' + str(no_data_val) + ' ' + vrtname + ' ' + file_string
        print(cmd)
        os.system(cmd)

        # Set Band Description
        VRT_dataset = gdal.Open(vrtname, gdal.GA_Update)
        for band_name, idx in zip(band_name_list, range(1, len(band_name_list) + 1)):
            VRT_dataset.GetRasterBand(idx).SetDescription(band_name)
        VRT_dataset = None

# def save_subplot_png(out, output_file):
    
#     import matplotlib.pyplot as plt
    
#     num_plots = len(out)
#     cols = 3
#     rows = (num_plots // cols) + (num_plots % cols > 0)

#     fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))

#     for idx, (title, array) in enumerate(out):
#         row = idx // cols
#         col = idx % cols
#         ax = axs[row, col] if rows > 1 else axs[col]
#         im = ax.imshow(array, cmap='viridis')
#         ax.set_title(title)
#         ax.axis('off')  # Turn off axis
#         fig.colorbar(im, ax=ax, orientation='vertical')

#     # Hide any empty subplots
#     for j in range(idx + 1, rows * cols):
#         fig.delaxes(axs.flatten()[j])

#     plt.tight_layout()
#     plt.savefig(output_file, bbox_inches='tight')
#     plt.close()
    
#     print('save image gabor')

def save_subplot_png(data, output_file):
    
    import matplotlib.pyplot as plt
    import os
    print(f"Saving subplot PNG to {output_file}")
    
    # Check if the directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist. Creating it.")
        os.makedirs(output_dir)

    num_plots = len(data)
    cols = 3
    rows = (num_plots // cols) + (num_plots % cols > 0)

    fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    for idx, (title, array) in enumerate(data):
        row = idx // cols
        col = idx % cols
        ax = axs[row, col] if rows > 1 else axs[col]
        im = ax.imshow(array, cmap='viridis')
        ax.set_title(title)
        ax.axis('off')  # Turn off axis
        fig.colorbar(im, ax=ax, orientation='vertical')

    # Hide any empty subplots
    for j in range(idx + 1, rows * cols):
        fig.delaxes(axs.flatten()[j])

    plt.tight_layout()
    
    # Debugging: Print the figure's tight layout and bounds
    print("Figure tight layout set.")
    
    # Save the figure
    try:
        plt.savefig(output_file, bbox_inches='tight')
        print(f"File saved successfully: {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")

    plt.close()
    print("Figure closed.")




def majority_filter(binary_map, size=5):
    
    import numpy as np
    import scipy.ndimage as ndimage
    # Define a function to calculate the majority value in the neighborhood
    def majority(values):
        values = values.astype(int)
        counts = np.bincount(values)
        return np.argmax(counts)

    # Apply the generic filter using the majority function
    filtered_map = ndimage.generic_filter(
        binary_map, function=majority, size=(size, size)
    )
    
    return filtered_map


def fill_nodata_multichannel(vhr_img_path, no_data_value=-9999):
    
    import numpy as np
    import scipy.ndimage as ndimage
    from utilities import open_image
    
    image, image_info = open_image(vhr_img_path)
    # Initialize an output image with the same shape as the input image
    filled_image = np.copy(image)
    
    def_no_data_mask = (image[0] == no_data_value).astype('uint8')
    
    # Apply the majority filter with a 5x5 window
    filtered_nodata_map = majority_filter(def_no_data_mask, size=5).astype('bool')
    
    # Loop through each channel
    for c in range(image.shape[0]):
        # Create a mask of the no data pixels for the current channel
        no_data_mask = (image[c] == no_data_value)
        
        # Create a structure element that defines connectivity (4-connected neighborhood)
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=bool)
        
        # Label the connected components in the no data mask
        labeled_array, num_features = ndimage.label(no_data_mask, structure=structure)
        
        for label in range(1, num_features + 1):
            # Extract the current connected component
            component_mask = (labeled_array == label)
            
            # Find the bounding box of the component
            slice_y, slice_x = ndimage.find_objects(component_mask)[0]
            
            # Extract the neighborhood around the component
            neighborhood_slice_y = slice(max(slice_y.start - 1, 0), min(slice_y.stop + 1, image.shape[1]))
            neighborhood_slice_x = slice(max(slice_x.start - 1, 0), min(slice_x.stop + 1, image.shape[2]))
            
            neighborhood = filled_image[c, neighborhood_slice_y, neighborhood_slice_x]
            neighborhood_mask = (neighborhood != no_data_value)
            
            if np.any(neighborhood_mask):
                # Compute the mean of the valid neighborhood pixels
                mean_value = np.mean(neighborhood[neighborhood_mask])
                
                # Fill the component with the mean value
                filled_image[c, component_mask] = mean_value
                
    out_path = vhr_img_path[:-4] + '_filled.tif'
    
    filled_image[:, filtered_nodata_map] = no_data_value
    
    save_image(filled_image, out_path, 'GTiff', 6, image_info['geotransform'], image_info['projection'], NoDataValue=no_data_value)
    
    
    return filled_image    

# Function to generate Gabor features

def gabor_features_generator(vhr_img_path, gabor_params, num_cores, no_data_val, resolution=None, assign_names=False):
    from utilities import open_image, save_image
    import os
    from osgeo import gdal
    import cv2
    import numpy as np
    import glob
    from joblib import Parallel, delayed
    import rasterio
    
    current_directory = os.getcwd()
    print("Current Working Directory:", current_directory)

    # Move to the current working directory (useful if you previously changed directories)
    os.chdir(current_directory)

    # Open the VHR (Very High Resolution) image
    image, image_info = open_image(vhr_img_path)
    no_data_mask = image[0] == no_data_val
    #flt_img = fill_isolated_nodata_multichannel(image, no_data_value=no_data_val)
    
    if resolution == None:
        
        resolution = image_info['geotransform'][1]
        
        print('Resolution set to input image resolution' )
        
        
    

    # Convert the image to grayscale and remove the alpha band if present
    if np.shape(image)[0] == 4:
        image = np.delete(image, -1, axis=0)
        
    image[image<0] = 0
    
    if np.shape(image)[0] == 3:
    
        gray_image = cv2.cvtColor(np.transpose(image.astype('uint8'), (1, 2, 0)), cv2.COLOR_BGR2GRAY)
    
        
        
    elif len(np.shape(image)) == 2:
        gray_image = rescale_array(image)
        
    
        
   
        
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
    
   
    out = Parallel(n_jobs=num_cores, verbose=1)(delayed(process_gabor_filter)(pc, gabor_folder_path, img2, gray_image, image_info, no_data_val)
                                      for pc in param_combinations)
    
    output_file = os.path.join(gabor_folder_path, 'Gabor_kernels.png')
    
    save_subplot_png(out, output_file)
    
    #if assign_names: 
    # Create feature stack
    name_RGB_list = ['1_R', '2_G', '3_B']

    # Save RGB bands as separate images
    for curr_band, curr_name in zip(range(len(name_RGB_list)), name_RGB_list):
        print(curr_band)
        with rasterio.open(vhr_img_path) as dataset:
            curr_image = dataset.read(curr_band + 1)
            curr_band_path = os.path.join(gabor_folder_path, os.path.basename(vhr_img_path)[:-4] + '_' + curr_name + '.tif')
            save_image(curr_image, curr_band_path, 'GTiff', 6, image_info['geotransform'], image_info['projection'], NoDataValue=-9999)
    # Create a list of feature paths and names
    features_path_list = sorted(glob.glob(os.path.join(gabor_folder_path, '*.tif')))
    
    features_name = [os.path.basename(n).split('_')[-1][:-4] for n in features_path_list]
        
  
        
    #features_name = name_RGB_list + filters_name

    # Create a VRT (Virtual Dataset) from the features
    filename_features = os.path.basename(vhr_img_path)[:-4] + '_features.vrt'
    create_vrt(features_path_list, features_name, filename_features,no_data_val, resolution=resolution)
    print('Texture features are saved in:  ' + gabor_folder_path)

    return gabor_folder_path
