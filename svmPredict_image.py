def svmPredict(svmModel, svmMatrix, score):
    """
    Function to make predictions using a trained SVM model.

    Parameters:
    svmModel (SVC): Trained SVM model.
    svmMatrix (array): Input data for prediction.
    score (bool): Flag to decide whether to return decision function scores.

    Returns:
    array: Predictions or probabilities/decision function scores based on the input parameters.
    """
    if svmModel.probability:
        return svmModel.predict_proba(svmMatrix)  # Return probabilities if probability flag is set
    elif score:
        return svmModel.decision_function(svmMatrix)  # Return decision function scores if score flag is set
    else:
        return svmModel.predict(svmMatrix)  # Return class predictions otherwise

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

    for input_fileName, output_fileName in zip(input_fileName_List, output_fileName_List):
        # Read the input image
        img = gdal.Open(input_fileName)
        if img is None:
            print('Unable to open ' + input_fileName)
            sys.exit(1)
        Ncol = img.GetRasterBand(1).XSize
        Nrow = img.GetRasterBand(1).YSize
        geoTransform = img.GetGeoTransform()
        projection = img.GetProjection()

        # Load the SVM model
        svm_dict = pickle.load(open(svm_fileName, 'rb'), encoding='latin1')

        # Set the cache size for the SVM model
        svm_dict['svmModel'].cache_size = svmCacheSize

        # Create the mask of the input to handle no-data values
        noNanPixels = np.ones((Nrow, Ncol), dtype=bool)
        if img.RasterCount >= 1:
            print('Creating the NOVALUE mask of the input image ' + input_fileName)
            for b in range(img.RasterCount):
                if np.isnan(inputNoDataValue):
                    noNanPixels = np.logical_and(noNanPixels, ~np.isnan(img.GetRasterBand(b + 1).ReadAsArray()))
                else:
                    noNanPixels = np.logical_and(noNanPixels, img.GetRasterBand(b + 1).ReadAsArray() != inputNoDataValue)

            # Check if at least one valid pixel exists
            if np.sum(noNanPixels) == 0:
                print(input_fileName + ' contains only no data values!')
                continue

        print('NOVALUE mask created. Creating the input samples list of the SVM from the input image...')

        # Create the list of band names and ordered band indexes for the SVM
        bandName_list = [img.GetRasterBand(b + 1).GetDescription() for b in range(img.RasterCount)]
        bandOrder = [bandName_list.index(b) + 1 for b in svm_dict['feature_names']]

        # Create the input samples for the SVM
        samples = np.zeros((np.sum(noNanPixels), len(bandOrder)))
        for i, b in enumerate(bandOrder):
            img_band = img.GetRasterBand(b).ReadAsArray()
            samples[:, i] = img_band[noNanPixels]
        img = None

        print('Input samples for the SVM ready. Starting the SVM classification...')

        # Normalize the samples
        samples = np.nan_to_num(samples)
        samples = svm_dict['normalizer'].transform(samples)

        # Divide samples into blocks for parallel processing
        samplesBlocks = np.array_split(samples, Nprocesses, axis=0)

        # Run the SVM prediction for each block in parallel
        classImage_arrayBlocks = Parallel(n_jobs=Nprocesses, verbose=10)(
            delayed(svmPredict)(svm_dict['svmModel'], samplesBlocks[i], score) for i in range(len(samplesBlocks)))

        # Concatenate the results
        if svm_dict['svmModel'].probability:
            probImage_array = np.concatenate(classImage_arrayBlocks, axis=0)
            classImage_array = np.argmax(probImage_array, axis=1) + 1
        elif score:
            classImage_array = np.concatenate(classImage_arrayBlocks)
            if len(classImage_array.shape) == 1:
                classImage_array = classImage_array.reshape(-1, 1)
        else:
            classImage_array = np.concatenate(classImage_arrayBlocks)

        print("Classification done. Writing the result to disk...")

        # Write the probability image if present
        if svm_dict['svmModel'].probability:
            driver = gdal.GetDriverByName('ENVI')
            prob_img = driver.Create(output_fileName + '_prob', Ncol, Nrow, len(svm_dict['target_names']), gdal.GDT_Float32)

            # Set the geographic information
            prob_img.SetGeoTransform(geoTransform)
            prob_img.SetProjection(projection)

            # Save the probability in each band
            for b in range(prob_img.RasterCount):
                layerBand = np.zeros((Nrow, Ncol), dtype='float32') * np.nan
                layerBand[noNanPixels] = probImage_array[:, b]
                prob_img_band = prob_img.GetRasterBand(b + 1)
                prob_img_band.WriteArray(layerBand)
                prob_img_band.SetDescription(svm_dict['target_names'][b])

            prob_img = None
            print('Probability image written in ' + output_fileName + '_prob')

        if score:
            driver = gdal.GetDriverByName('ENVI')
            score_img = driver.Create(output_fileName, Ncol, Nrow, len(svm_dict['target_names']) * (len(svm_dict['target_names']) - 1) // 2, gdal.GDT_Float32)

            # Set the geographic information
            score_img.SetGeoTransform(geoTransform)
            score_img.SetProjection(projection)

            # Save the decision function scores in each band
            for b in range(score_img.RasterCount):
                layerBand = np.zeros((Nrow, Ncol), dtype='float32') * np.nan
                layerBand[noNanPixels] = classImage_array[:, b]
                prob_img_band = score_img.GetRasterBand(b + 1)
                prob_img_band.WriteArray(layerBand)

            score_img = None
            print('Score image written in ' + output_fileName)

        else:
            # Initialize the classification map
            driver = gdal.GetDriverByName('ENVI')
            class_img = driver.Create(output_fileName, Ncol, Nrow, 1, gdal.GDT_Byte)

            # Set the geographic information
            class_img.SetGeoTransform(geoTransform)
            class_img.SetProjection(projection)

            # Create the raster of the classification map and write it
            layerBand = np.zeros((Nrow, Ncol), dtype='uint8')
            layerBand[noNanPixels] = classImage_array
            class_img_band = class_img.GetRasterBand(1)
            class_img_band.WriteArray(layerBand)

            # Set the class labels and the no-data value
            class_img_band.SetRasterCategoryNames(['Unclassified'] + svm_dict['target_names'])

            # Initialize the color table
            colorTable = gdal.ColorTable(gdal.GPI_RGB)
            colorTable.SetColorEntry(0, (0, 0, 0))  # Black for 'Unclassified'

            # Set the colors for all the classes
            for i, values in enumerate(svm_dict['target_names']):
                colorTable.SetColorEntry(i + 1, classColors[values])
                class_img_band.SetColorTable(colorTable)

            class_img = None
