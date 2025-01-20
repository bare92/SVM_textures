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
fill_holes = False
#vhr_unf_img_path_list = ['/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_June/georef_result/June_11/hintereisferner1-220611-1230-hu_map.tif']

vhr_unf_img_path_list =  glob.glob(os.path.join('/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/webcam_orthorect', 'georef*', 'georef_result', '*', '*-hu_map.tif'))


no_data_val = np.nan

###########################################################################

crop_maps = False
# Definisci il percorso della directory contenente le immagini

mask_shapefile = "/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect_1/hintereisferner_extent_s2.shp"
 
# Ottieni una lista di tutti i file TIFF nella directory di input
input_files = glob.glob(os.path.join('/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/webcam_orthorect', 'georef*', 'georef_result', '*', 'h*-hu_map_filled.tif'))  
 
 


##########################################################################
# Texture features generation
generate_features = False # Flag to control whether to generate texture features
#vhr_img_path = '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/georef_August/georef_result/August_01/hintereisferner1-220801-0830-hu_map_filled.tif'
vhr_img_path_list = glob.glob(os.path.join('/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/webcam_orthorect', 'georef*', 'georef_result', '*', 'cropped*hu_map_filled.tif'))

vhr_img_path_list = ['/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/HYPERSPECTRAL/Downsampled_UAV/20240704_Cervinia_snow_patches_ortho_25cm_orto_32632.tif']
# Parallel processing
num_cores = 8  # Number of CPU cores to use for parallel processing

#Parameters for Gabor filter
gabor_params = {'theta_list': [theta / 4. * np.pi for theta in range(4)],  # List of theta values for Gabor filters
                'lambda_list': np.arange(2, 16, 2),
                'gamma_list': [0.5],
                'sigma_list': [1, 3],
                'ksize': 15}  # List of lambda values for Gabor filters

# gabor_params = {'theta_list': [theta / 4. * np.pi for theta in range(4)],  # List of theta values for Gabor filters
#                 'lambda_list': np.arange(2, 8, 4),
#                 'gamma_list': [0.5],
#                 'sigma_list': [],
#                 'ksize': None}  # List of lambda values for Gabor filters

# PCA
#

PCA = False
n_components = 10
no_data_val = np.nan
##########################################################################
# Parameters for extracting training samples from shapefiles
performs2d = True  # Flag to control whether to perform the extraction of training samples


###### LANDSAT 7
# List of raster files
fileNameList_raster = [
    '/mnt/CEPH_PROJECTS/PROSNOW/MRI_Andes/Landsat_Maipo/Landsat-7/LE07_L1TP_merged_20130803_20200907_02_T1/LE07_L1TP_merged_20130803_20200907_02_T1_cloud.vrt',
    '/mnt/CEPH_PROJECTS/PROSNOW/MRI_Andes/Landsat_Maipo/Landsat-7/LE07_L1TP_merged_20151129_20200903_02_T1/LE07_L1TP_merged_20151129_20200903_02_T1_cloud.vrt',
    '/mnt/CEPH_PROJECTS/PROSNOW/MRI_Andes/Landsat_Maipo/Landsat-7/LE07_L1TP_merged_20170408_20200831_02_T1/LE07_L1TP_merged_20170408_20200831_02_T1_cloud.vrt',
    '/mnt/CEPH_PROJECTS/PROSNOW/MRI_Andes/Landsat_Maipo/Landsat-7/LE07_L1TP_merged_20130515_20200907_02_T1/LE07_L1TP_merged_20130515_20200907_02_T1_cloud.vrt',
    '/mnt/CEPH_PROJECTS/PROSNOW/MRI_Andes/Landsat_Maipo/Landsat-7/LE07_L1TP_merged_20150629_20200904_02_T1/LE07_L1TP_merged_20150629_20200904_02_T1_cloud.vrt'
    ]

# List of raster files
# fileNameList_raster = [
#     '/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/HYPERSPECTRAL/Downsampled_UAV/Gabor_features/00_features.vrt',
#     '/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/HYPERSPECTRAL/Downsampled_UAV/Gabor_features/00_features.vrt',
#     '/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/HYPERSPECTRAL/Downsampled_UAV/Gabor_features/00_features.vrt',
#     '/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/HYPERSPECTRAL/Downsampled_UAV/Gabor_features/00_features.vrt'
#     ]

# fileNameList_raster = [
#     '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/OUT_Flavia/georef_result/test_run_Flavia_4/hintereisferner1-2018-07-19-1100_map.tif',
#     '/mnt/CEPH_PROJECTS/ALPSNOW/Flavia/webcam_orthorect/OUT_Flavia/georef_result/test_run_Flavia_4/hintereisferner1-2018-07-19-1100_map.tif'
        
# ]

# List of shapefiles
# List of shapefiles
fileNameList_shape = [
    '/mnt/CEPH_PROJECTS/PROSNOW/MRI_Andes/Landsat_Maipo/clouds/Landsat-7/LE07_L1TP_merged_20130803_20200907_02_T1.shp',
    '/mnt/CEPH_PROJECTS/PROSNOW/MRI_Andes/Landsat_Maipo/clouds/Landsat-7/LE07_L1TP_merged_20151129_20200903_02_T1.shp',
    '/mnt/CEPH_PROJECTS/PROSNOW/MRI_Andes/Landsat_Maipo/clouds/Landsat-7/LE07_L1TP_merged_20170408_20200831_02_T1.shp',
    '/mnt/CEPH_PROJECTS/PROSNOW/MRI_Andes/Landsat_Maipo/clouds/Landsat-7/LE07_L1TP_merged_20130515_20200907_02_T1.shp',
    '/mnt/CEPH_PROJECTS/PROSNOW/MRI_Andes/Landsat_Maipo/clouds/Landsat-7/LE07_L1TP_merged_20150629_20200904_02_T1.shp'
    ]

## LANDSAT 8-9

fileNameList_raster = [
    '/mnt/CEPH_PROJECTS/PROSNOW/MRI_Andes/Landsat_Maipo/Landsat-8/LC08_L1TP_merged_20221030_20221108_02_T1/LC08_L1TP_merged_20221030_20221108_02_T1_cloud.vrt',
    '/mnt/CEPH_PROJECTS/PROSNOW/MRI_Andes/Landsat_Maipo/Landsat-8/LC08_L1TP_merged_20230118_20230131_02_T1/LC08_L1TP_merged_20230118_20230131_02_T1_cloud.vrt',
    '/mnt/CEPH_PROJECTS/PROSNOW/MRI_Andes/Landsat_Maipo/Landsat-8/LC08_L1TP_merged_20230307_20230316_02_T1/LC08_L1TP_merged_20230307_20230316_02_T1_cloud.vrt'
    ]


fileNameList_shape = [
    '/mnt/CEPH_PROJECTS/PROSNOW/MRI_Andes/Landsat_Maipo/clouds/Landsat-8_9/LC08_L1TP_merged_20221030_20221108_02_T1.shp',
    '/mnt/CEPH_PROJECTS/PROSNOW/MRI_Andes/Landsat_Maipo/clouds/Landsat-8_9/LC08_L1TP_merged_20230118_20230131_02_T1.shp',
    '/mnt/CEPH_PROJECTS/PROSNOW/MRI_Andes/Landsat_Maipo/clouds/Landsat-8_9/LC08_L1TP_merged_20230307_20230316_02_T1.shp'
    ]





fieldName = 'class'  # Field name in shapefiles representing the class labels
round_ = os.path.basename(os.path.dirname(fileNameList_shape[-1]))  # Extracting the round name from the directory path
output_training_filename = os.path.join(os.path.dirname(fileNameList_shape[-1]), round_ + '_Cloud_TrainingSet8.p')  # Output path for the training set
noDataValue = np.nan  # Value representing no data in raster files

##########################################################################
# Parameters for SVM training
performSVMtrain = True  # Flag to control whether to perform SVM training

# Parameters for SVM training
training_set_filename = output_training_filename

cv = 5  # Number of cross-validation folds
probFlag = False  # Flag to indicate whether to enable probability estimates
n_jobs = -1  # Number of jobs to run in parallel (-1 means using all processors)
cs = 4028  # Random seed or custom setting
target_names=['cloud', 'cloud_free']

xgboost_model_filename_out = os.path.join(os.path.dirname(fileNameList_shape[-1]), round_ + '_cloud_model_xgboost8.p')  # Output path for the trained SVM model
#grid_search_filename = os.path.join(os.path.dirname(fileNameList_shape[-1]), round_ + '_Snow_grid_xgboost.png')  # Output path for the grid search plot


##########################################################################
# Parameters for SVM prediction
performSVMpredict = True  # Flag to control whether to perform SVM prediction

# Parameters for SVM prediction
score = False  # Flag to indicate whether to compute prediction scores
svm_model_filename = xgboost_model_filename_out  # Input path for the trained SVM model
#svm_model_filename = '/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/HYPERSPECTRAL/UAV_SVM_texture/Round_04/Round_04_Snow_model_xgboost16feat.p'
Nprocesses = 8  # Number of processes to use for prediction
svmCacheSize = 8000  # Cache size for the SVM model in MB
classColors = {'cloud': (150,150,150),
'cloud_free': (255,255,255)}  # Colors to represent the classes in the output map

# Preparing input and output file names for prediction

# folder_to_search = '/mnt/CEPH_PROJECTS/ALPSNOW/Riccardo/HYPERSPECTRAL/Downsampled_UAV/Subimages_56feat'

output_folder = os.path.join(os.path.dirname(fileNameList_shape[0]), 'cloud_classification_results8')
os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

    
input_fileName = [
    '/mnt/CEPH_PROJECTS/PROSNOW/MRI_Andes/Landsat_Maipo/Landsat-8/LC08_L1TP_merged_20221030_20221108_02_T1/LC08_L1TP_merged_20221030_20221108_02_T1_cloud.vrt',
    '/mnt/CEPH_PROJECTS/PROSNOW/MRI_Andes/Landsat_Maipo/Landsat-8/LC08_L1TP_merged_20230118_20230131_02_T1/LC08_L1TP_merged_20230118_20230131_02_T1_cloud.vrt',
    '/mnt/CEPH_PROJECTS/PROSNOW/MRI_Andes/Landsat_Maipo/Landsat-8/LC08_L1TP_merged_20230307_20230316_02_T1/LC08_L1TP_merged_20230307_20230316_02_T1_cloud.vrt',
    '/mnt/CEPH_PROJECTS/PROSNOW/MRI_Andes/Landsat_Maipo/Landsat-9/LC09_L1TP_merged_20220812_20230403_02_T1/LC09_L1TP_merged_20220812_20230403_02_T1_cloud.vrt'
    ]
#input_fileName = ['/mnt/CEPH_PROJECTS/PROSNOW/MRI_Andes/Landsat_Maipo/Landsat-7/LE07_L1TP_merged_20210731_20210826_02_T1/LE07_L1TP_merged_20210731_20210826_02_T1_cloud.vrt']
output_filename = []
for curr_filename in input_fileName:
    curr_output_filename = os.path.join(output_folder, os.path.basename(curr_filename)[:-4] + '_cloud_map')
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
    svmTraining(training_set_filename, cv, n_jobs, cs, xgboost_model_filename_out)

if performSVMpredict:
    print('PREDICTING CLASSES...')
    from svmPredict_image import svmPredict_image
    svmPredict_image(svm_model_filename, input_fileName, Nprocesses, svmCacheSize, noDataValue, score, classColors, output_filename)







