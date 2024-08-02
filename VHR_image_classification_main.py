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
# Texture features generation
generate_features = True  # Flag to control whether to generate texture features
vhr_img_path = '/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/HYPERSPECTRAL/Downsampled_subset_UAV/20240704_Cervinia_snow_patches_ortho_25cm_orto_6707_subset.tif'

# Parallel processing
num_cores = 4  # Number of CPU cores to use for parallel processing

# Parameters for Gabor filter
gabor_params = {'theta_list': [theta / 4. * np.pi for theta in range(4)],  # List of theta values for Gabor filters
                'lambda_list': np.arange(2, 16, 4)}  # List of lambda values for Gabor filters

##########################################################################
# Parameters for extracting training samples from shapefiles
performs2d = True  # Flag to control whether to perform the extraction of training samples

# List of raster files
fileNameList_raster = [
    '/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/HYPERSPECTRAL/Downsampled_UAV/Gabor_features/00_features.vrt',
    '/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/HYPERSPECTRAL/Downsampled_UAV/Gabor_features/00_features.vrt',
    '/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/HYPERSPECTRAL/Downsampled_UAV/Gabor_features/00_features.vrt'
]

# List of shapefiles
fileNameList_shape = [
    '/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/HYPERSPECTRAL/UAV_SVM_texture/Round_01/round_01.shp',
    '/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/HYPERSPECTRAL/UAV_SVM_texture/Round_02/round_02_tex.shp',
    '/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/HYPERSPECTRAL/UAV_SVM_texture/Round_03/round_03_tex.shp'
]

fieldName = 'class'  # Field name in shapefiles representing the class labels
round_ = os.path.basename(os.path.dirname(fileNameList_shape[-1]))  # Extracting the round name from the directory path
output_training_filename = os.path.join(os.path.dirname(fileNameList_shape[-1]), round_ + '_Snow_TrainingSet.p')  # Output path for the training set
noDataValue = 0  # Value representing no data in raster files

##########################################################################
# Parameters for SVM training
performSVMtrain = False  # Flag to control whether to perform SVM training

# Parameters for SVM training
training_set_filename = output_training_filename
gamma_range = np.logspace(-2, 1, 50)  # Range of gamma values for SVM
C_range = np.logspace(-2, 2, 50)  # Range of C values for SVM
cv = 5  # Number of cross-validation folds
probFlag = False  # Flag to indicate whether to enable probability estimates
n_jobs = -1  # Number of jobs to run in parallel (-1 means using all processors)
cs = 4028  # Random seed or custom setting

svm_model_filename_out = os.path.join(os.path.dirname(fileNameList_shape[-1]), round_ + '_Snow_model.p')  # Output path for the trained SVM model
grid_search_filename = os.path.join(os.path.dirname(fileNameList_shape[-1]), round_ + '_Snow_grid.png')  # Output path for the grid search plot

##########################################################################
# Parameters for SVM prediction
performSVMpredict = True  # Flag to control whether to perform SVM prediction

# Parameters for SVM prediction
score = False  # Flag to indicate whether to compute prediction scores
svm_model_filename = svm_model_filename_out  # Input path for the trained SVM model
Nprocesses = 8  # Number of processes to use for prediction
svmCacheSize = 8000  # Cache size for the SVM model in MB
classColors = {'snow': (255, 255, 255), 'no_snow': (152, 118, 84)}  # Colors to represent the classes in the output map

# Preparing input and output file names for prediction
vrt_folder = '/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/HYPERSPECTRAL/Downsampled_UAV/Subimages'
input_fileName = sorted(glob.glob(os.path.join(vrt_folder, '*.vrt')))  # List of input VRT files

output_folder = os.path.join(os.path.dirname(vrt_folder), 'Classified_subimages')
os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

output_filename = []
for curr_filename in input_fileName:
    curr_output_filename = os.path.join(output_folder, os.path.basename(curr_filename)[:-4] + '_round_' + round_ + '_snow_map')
    output_filename.append(curr_output_filename)

##########################################################################
# Main execution logic
if generate_features:    
    print('GENERATING CLASSIFICATION FEATURES...')
    from texture_features_generation import gabor_features_generator
    gabor_folder_path = gabor_features_generator(vhr_img_path, gabor_params, num_cores)

if performs2d:
    print('EXTRACTING TRAINING SAMPLES...')
    from shp2data import shp2data
    shp2data(fileNameList_raster, fileNameList_shape, fieldName, output_training_filename, noDataValue, target_names=['snow', 'no_snow'])

if performSVMtrain:
    print('TRAINING SVM...')
    from svmTraining import svmTraining
    svmTraining(training_set_filename, gamma_range, C_range, cv, probFlag, n_jobs, cs, svm_model_filename_out, grid_search_filename)

if performSVMpredict:
    print('PREDICTING CLASSES...')
    from svmPredict_image import svmPredict_image
    svmPredict_image(svm_model_filename, input_fileName, Nprocesses, svmCacheSize, noDataValue, score, classColors, output_filename)

