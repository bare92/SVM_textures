def xgboostPredict(best_model, features):
    
    return best_model.predict(features)

    
def apply_color_table(output_fileName, classColors, target_names):
    
    from osgeo import gdal
    """
    Apply a color table to an existing classified raster image.

    Args:
        image_path (str): Path to the existing classified raster.
        class_colors (dict): Dictionary mapping class labels to RGB tuples.

    Returns:
        None
    """
    # Open the raster in update mode
    dataset = gdal.Open(output_fileName, gdal.GA_Update)
    if dataset is None:
        raise FileNotFoundError(f"Unable to open {output_fileName}")
    
    # Get the first band (assuming classification raster is single-band)
    band = dataset.GetRasterBand(1)
    
    # Create a color table
    color_table = gdal.ColorTable()
    for i, values in enumerate(target_names):
      color_table.SetColorEntry(i + 1, classColors[values])
    # Assign the color table to the band
    band.SetColorTable(color_table)
    band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    
    # Close the dataset to save changes
    dataset = None
    print(f"Color table applied to {output_fileName}")

def svmPredict_image(svm_fileName, input_fileName_List, Nprocesses, svmCacheSize, inputNoDataValue, score,
                     classColors, output_fileName_List):
    """
    Function to apply SVM model on image data for classification.

    Parameters:
    svm_fileName (str): Path to the saved SVM model file.
    input_fileName_List (list): List of input image file paths.
    Nprocesses (int): Number of parallel processes to use.
    svmCacheSize (int): Cache size for SVM.
    inputNoDataValue (float): Value representing no-data in the input images.
    score (bool): Flag to decide whether to return decision function scores.
    classColors (list): List of RGB tuples representing class colors.
    output_fileName_List (list): List of output image file paths.

    Returns:
    None
    """
    import numpy as np
    from osgeo import gdal
    from joblib import Parallel, delayed
    import pickle
    from sklearn.svm import SVC
    from sklearn import preprocessing
    import sys
    import os
    from xgboost import XGBClassifier
    import rasterio
    from rasterio.enums import ColorInterp
    from rasterio.io import MemoryFile

    for input_fileName, output_fileName in zip(input_fileName_List, output_fileName_List):
        # Check if the output file already exists
        if os.path.exists(output_fileName):
            print(f"IMAGE ALREADY CLASSIFIED: {output_fileName}")
            continue
    
        # Load the SVM model
        with open(svm_fileName, 'rb') as model_file:
            svm_dict = pickle.load(model_file)
        xgboost_model = svm_dict['xgboostModel']
        normalizer = svm_dict['normalizer']
        feature_names = svm_dict['feature_names']
    
        # Open the input raster
        with rasterio.open(input_fileName) as src:
            profile = src.profile
            profile.update(dtype='uint8', count=1, nodata=0)  # Update profile for the output
            print("Descriptions of bands:", src.descriptions)
            # Read all bands and ensure they have the same dtype
            print("Reading all raster bands and normalizing dtypes...")
            bands = np.array([src.read(i + 1).astype(np.float32) for i in range(src.count)])
   
            # Create no-data mask
            print("Creating no-data mask...")
            if np.isnan(inputNoDataValue):
                noNanPixels = ~np.isnan(bands).any(axis=0)
            else:
                noNanPixels = ~(bands == inputNoDataValue).any(axis=0)
    
            if not np.any(noNanPixels):
                print(f"{input_fileName} contains only no-data values!")
                
    
            print("No-data mask created. Extracting features...")
    
            # Extract feature indices and values
            band_indices = [src.indexes[src.descriptions.index(name)] for name in feature_names]
            features = np.column_stack([bands[i - 1][noNanPixels] for i in band_indices])
    
        # Normalize features
        features = np.nan_to_num(features)
        features = normalizer.transform(features)
    
        # Split features for parallel processing
        feature_blocks = np.array_split(features, Nprocesses)
    
        # Classify in parallel using XGBoost
        print("Starting XGBoost classification...")
        def classify_block(block):
            return xgboost_model.predict(block)
    
        predictions_blocks = Parallel(n_jobs=Nprocesses, verbose=10)(
            delayed(classify_block)(block) for block in feature_blocks
        )
        predictions = np.concatenate(predictions_blocks) + 1  # Adjust class indices
    
        # Create the output raster
        print(f"Writing classified raster to {output_fileName}...")
        class_map = np.zeros((profile['height'], profile['width']), dtype='uint8')
        class_map[noNanPixels] = predictions
        # Update profile for the output file
        profile.update(
            dtype='uint8',
            count=1,  # Single band for classification output
            driver='GTiff',  # Ensure output is a GeoTIFF
            )
        print(f"Writing classified raster to {output_fileName}...")
        with rasterio.open(output_fileName, 'w', **profile) as dst:
            # Write the classified map
            dst.write(class_map.astype('uint8'), 1)
        target_names = svm_dict['target_names']
         
        apply_color_table(output_fileName, classColors, target_names)
    
        print(f"Classification completed and saved: {output_fileName}")
