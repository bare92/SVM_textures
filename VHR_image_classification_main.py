#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:23:44 2024

@author: rbarella
"""

import os
import numpy as np
import glob


##########################################################################
# Fill holes
fill_holes = True
#vhr_unf_img_path_list = ['/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_June/georef_result/June_11/hintereisferner1-220611-1230-hu_map.tif']

vhr_unf_img_path_list =  glob.glob(os.path.join('/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/webcam_orthorect', 'georef*', 'georef_result', '*', '*-hu_map.tif'))


no_data_val = -9999

###########################################################################

crop_maps = True
# Definisci il percorso della directory contenente le immagini

mask_shapefile = "/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect_1/hintereisferner_extent_s2.shp"
 
# Ottieni una lista di tutti i file TIFF nella directory di input
input_files = glob.glob(os.path.join('/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/webcam_orthorect', 'georef*', 'georef_result', '*', 'h*-hu_map_filled.tif'))  
 
 


##########################################################################
# Texture features generation
generate_features = True # Flag to control whether to generate texture features
#vhr_img_path = '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_August/georef_result/August_01/hintereisferner1-220801-0830-hu_map_filled.tif'
vhr_img_path_list = glob.glob(os.path.join('/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/webcam_orthorect', 'georef*', 'georef_result', '*', 'cropped*hu_map_filled.tif'))
# Parallel processing
num_cores = 8  # Number of CPU cores to use for parallel processing

#Parameters for Gabor filter
gabor_params = {'theta_list': [theta / 4. * np.pi for theta in range(4)],  # List of theta values for Gabor filters
                'lambda_list': np.arange(2, 16, 2),
                'gamma_list': [0.5],
                'sigma_list': [1, 3],
                'ksize': 7}  # List of lambda values for Gabor filters

# gabor_params = {'theta_list': [theta / 4. * np.pi for theta in range(4)],  # List of theta values for Gabor filters
#                 'lambda_list': np.arange(2, 8, 4),
#                 'gamma_list': [0.5],
#                 'sigma_list': [],
#                 'ksize': None}  # List of lambda values for Gabor filters

# PCA
#

PCA = True
n_components = 10
no_data_val = -9999
##########################################################################
# Parameters for extracting training samples from shapefiles
performs2d = False  # Flag to control whether to perform the extraction of training samples

# # List of raster files
# fileNameList_raster = [
#     '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_August/georef_result/August_01/Gabor_features/hintereisferner1-220801-0830-hu_map_filled_features.vrt',
#     '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_August/georef_result/August_01/Gabor_features/hintereisferner1-220801-0830-hu_map_filled_features.vrt',
#     '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_August/georef_result/August_01/Gabor_features/hintereisferner1-220801-0830-hu_map_filled_features.vrt',
#     '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_June/georef_result/June_11/Gabor_features/hintereisferner1-220611-1230-hu_map_filled_features.vrt',
#     '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_June/georef_result/June_11/Gabor_features/hintereisferner1-220611-1230-hu_map_filled_features.vrt',
#     '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_June/georef_result/June_14/Gabor_features/hintereisferner1-220614-1130-hu_map_filled_features.vrt',
#     '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_August/georef_result/August_05/Gabor_features/hintereisferner1-220805-1230-hu_map_filled_features.vrt',
#     '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_August/georef_result/August_01/Gabor_features/hintereisferner1-220801-0830-hu_map_filled_features.vrt'
# ]

# List of raster files
fileNameList_raster = [
    '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_August/georef_result/August_01/Gabor_features_56_filters/hintereisferner1-220801-0830-hu_map_filled_PCA_features.tif',
    '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_August/georef_result/August_01/Gabor_features_56_filters/hintereisferner1-220801-0830-hu_map_filled_PCA_features.tif',
    '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_August/georef_result/August_01/Gabor_features_56_filters/hintereisferner1-220801-0830-hu_map_filled_PCA_features.tif',
    '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_June/georef_result/June_11/Gabor_features_56_filters/hintereisferner1-220611-1230-hu_map_filled_PCA_features.tif',
    '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_June/georef_result/June_11/Gabor_features_56_filters/hintereisferner1-220611-1230-hu_map_filled_PCA_features.tif',
    '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_June/georef_result/June_14/Gabor_features_56_filters/hintereisferner1-220614-1130-hu_map_filled_PCA_features.tif',
    '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_August/georef_result/August_05/Gabor_features_56_filters/hintereisferner1-220805-1230-hu_map_filled_PCA_features.tif',
    '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_August/georef_result/August_01/Gabor_features_56_filters/hintereisferner1-220801-0830-hu_map_filled_PCA_features.tif'
]

# fileNameList_raster = [
#     '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/OUT_Flavia/georef_result/test_run_Flavia_4/hintereisferner1-2018-07-19-1100_map.tif',
#     '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/OUT_Flavia/georef_result/test_run_Flavia_4/hintereisferner1-2018-07-19-1100_map.tif'
        
# ]

# List of shapefiles
fileNameList_shape = [
    '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_August/georef_result/Image_classification/round_01/round_01.shp',
    '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_August/georef_result/Image_classification/round_02/round_02.shp',
    '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_August/georef_result/Image_classification/round_03/round_03.shp',
    '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_August/georef_result/Image_classification/round_04/round_04.shp',
    '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_August/georef_result/Image_classification/round_05/round_05.shp',
    '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_August/georef_result/Image_classification/round_06/round_06.shp',
    '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_August/georef_result/Image_classification/round_07/round_07.shp',
    '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_August/georef_result/Image_classification/round_08/round_08.shp'
]

fieldName = 'class'  # Field name in shapefiles representing the class labels
round_ = os.path.basename(os.path.dirname(fileNameList_shape[-1]))  # Extracting the round name from the directory path
output_training_filename = os.path.join(os.path.dirname(fileNameList_shape[-1]), round_ + '_Snow_TrainingSet.p')  # Output path for the training set
noDataValue = -9999  # Value representing no data in raster files

##########################################################################
# Parameters for SVM training
performSVMtrain = False  # Flag to control whether to perform SVM training

# Parameters for SVM training
training_set_filename = output_training_filename
gamma_range = np.logspace(-2, 1, 50)  # Range of gamma values for SVM
C_range = np.logspace(-1, 2, 50)  # Range of C values for SVM
cv = 8  # Number of cross-validation folds
probFlag = False  # Flag to indicate whether to enable probability estimates
n_jobs = -1  # Number of jobs to run in parallel (-1 means using all processors)
cs = 4028  # Random seed or custom setting
target_names=['snow', 'ice', 'other']

svm_model_filename_out = os.path.join(os.path.dirname(fileNameList_shape[-1]), round_ + '_Snow_model2.p')  # Output path for the trained SVM model
grid_search_filename = os.path.join(os.path.dirname(fileNameList_shape[-1]), round_ + '_Snow_grid2.png')  # Output path for the grid search plot


##########################################################################
# Parameters for SVM prediction
performSVMpredict = False  # Flag to control whether to perform SVM prediction

# Parameters for SVM prediction
score = False  # Flag to indicate whether to compute prediction scores
#svm_model_filename = svm_model_filename_out  # Input path for the trained SVM model
svm_model_filename = '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect_1/georef_August/georef_result/Image_classification/round_08/round_08_Snow_model.p'
Nprocesses = 8  # Number of processes to use for prediction
svmCacheSize = 8000  # Cache size for the SVM model in MB
classColors = {'other': (152,118,84),
'snow': (255,255,255),
'ice': (153,255,255)}  # Colors to represent the classes in the output map

# Preparing input and output file names for prediction

#input_fileName = [fileNameList_raster[0]]
input_fileName= ['/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_August/georef_result/August_01/Gabor_features/hintereisferner1-220801-0830-hu_map_filled_features.vrt',
                 '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_August/georef_result/August_06/Gabor_features/hintereisferner1-220806-1230-hu_map_filled_features.vrt',
                 '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_June/georef_result/June_11/Gabor_features/hintereisferner1-220611-1230-hu_map_filled_features.vrt']


folder_to_search = '/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/webcam_orthorect/'

output_folder = os.path.join(folder_to_search, '00_classified_Gabor_features_0_filters')
os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

    
input_fileName = glob.glob(os.path.join(folder_to_search, 'georef*', 'georef_result', '*', 'Gabor_features*0*', 'cropped_*-hu_map_filled_features.vrt'))    

output_filename = []
for curr_filename in input_fileName:
    curr_output_filename = os.path.join(output_folder, os.path.basename(curr_filename)[:-4] + '_round_' + round_ + '_snow_map')
    output_filename.append(curr_output_filename)
##########################################################################
# Main execution logic
if fill_holes:
    print('NO DATA FILLING...')
    from texture_features_generation import fill_nodata_multichannel_optimized
    
    for vhr_unf_img_path in vhr_unf_img_path_list:
        
        filled_img = fill_nodata_multichannel_optimized(vhr_unf_img_path, no_data_value=no_data_val)

if crop_maps:
    
    print('CROP MAPS WITH SHAPEFILE...')
    # Loop per elaborare ogni file
    for input_file in input_files:
        
        output_path = os.path.join(os.path.dirname(input_file), "cropped_" + os.path.basename(input_file))
        # Comando gdalwarp per ritagliare il file
        #gdal_command = f"gdalwarp -overwrite -of GTiff -cutline {mask_shapefile} -cl hintereisferner_extent_s2 -crop_to_cutline {input_file} {output_path}"
        
        gdal_command = f"gdalwarp -overwrite -of GTiff -cutline {mask_shapefile} -cl hintereisferner_extent_s2 -crop_to_cutline {input_file} {output_path}"
        
        if not os.path.exists(output_path):
            # Esegui il comando
            os.system(gdal_command)
            print(f"Processed {input_file}")

if generate_features:    
    print('GENERATING CLASSIFICATION FEATURES...')
    from texture_features_generation import gabor_features_generator
    for vhr_img_path in vhr_img_path_list:
        gabor_folder_path = gabor_features_generator(vhr_img_path, gabor_params, num_cores, no_data_val, PCA = PCA, n_components=n_components)

if performs2d:
    print('EXTRACTING TRAINING SAMPLES...')
    from shp2data import shp2data
    shp2data(fileNameList_raster, fileNameList_shape, fieldName, output_training_filename, noDataValue, target_names=target_names)

if performSVMtrain:
    print('TRAINING SVM...')
    from svmTraining import svmTraining
    svmTraining(training_set_filename, gamma_range, C_range, cv, probFlag, n_jobs, cs, svm_model_filename_out, grid_search_filename)

if performSVMpredict:
    print('PREDICTING CLASSES...')
    from svmPredict_image import svmPredict_image
    svmPredict_image(svm_model_filename, input_fileName, Nprocesses, svmCacheSize, noDataValue, score, classColors, output_filename)

