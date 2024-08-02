def shp2data(fileNameList_raster, fileNameList_shape, fieldName, output_filename, noDataValue, target_names=[]):
    from osgeo import gdal, ogr
    import numpy as np
    from shapely.geometry import Polygon, Point
    import sys
    import pickle
    import pandas as pd

    # Check if the number of shape files is equal to the number of raster files
    if len(fileNameList_shape) != len(fileNameList_raster):
        print('Error: number of listed shape files different to number of listed rasters')
        sys.exit(1)

    # Create the feature_names list from the raster band names
    for i, fileName_raster in enumerate(fileNameList_raster):
        # Read the raster
        raster = gdal.Open(fileName_raster)
        if raster is None:
            print('Unable to open ' + fileNameList_raster[0])
            sys.exit(1)

        # Create the feature_names list from the current raster band names
        feature_names_curr = []
        for b in range(raster.RasterCount):
            feature_names_curr.append(raster.GetRasterBand(b + 1).GetDescription())
        raster = None

        # Check if feature names are consistent across all rasters
        if i == 0:
            feature_names = feature_names_curr
        else:
            if feature_names != feature_names_curr:
                print('Error: band names of ' + fileName_raster + ' no coherent with the other files')
                sys.exit(1)

    print('Feature names : ' + str(feature_names))

    # Create the target_names list from the attributes of the shape features
    if len(target_names) == 0:
        for fileName_shape in fileNameList_shape:
            # Read the shape file
            shp = ogr.Open(fileName_shape)
            if shp is None:
                print('Unable to open ' + fileName_shape)
                sys.exit(1)
            lyr = shp.GetLayer()

            # Create the target names list
            for f in range(lyr.GetFeatureCount()):
                target_name_curr = lyr.GetFeature(f).GetField(fieldName)
                if target_name_curr not in target_names:
                    target_names.append(target_name_curr)
            shp = None

    print('Target names: ' + str(target_names))

    # Initialize lists to hold coordinates, image filenames, data, and target values
    coordinates = []
    image_fileNames = []
    data = np.empty((0, len(feature_names)))
    target = np.empty((0, 1))

    # For each shape file, extract the samples from the corresponding raster
    for fileName_shape, fileName_raster in zip(fileNameList_shape, fileNameList_raster):
        # Read the raster and the shape file
        raster = gdal.Open(fileName_raster)
        shp = ogr.Open(fileName_shape)
        lyr = shp.GetLayer()

        # Extract geographic information from the raster
        geotransform = raster.GetGeoTransform()
        ulX = geotransform[0]
        ulY = geotransform[3]
        xRes = geotransform[1]
        yRes = geotransform[5]

        # For each feature in the shape, extract the points from the raster
        for f in range(lyr.GetFeatureCount()):
            feat = lyr.GetFeature(f)
            target_curr = target_names.index(feat.GetField(fieldName)) + 1
            geom = feat.GetGeometryRef()

            if geom.GetGeometryName() == 'POINT':
                # Extract the point coordinates
                point = geom.GetPoints()[0]
                x = int((point[0] - ulX) / xRes)
                y = int((point[1] - ulY) / yRes)
                coordinates.append((x * xRes + ulX, y * yRes + ulY))

                # Extract the raster values at the point location
                newSample = np.empty((1, len(feature_names)))
                for b in range(raster.RasterCount):
                    band = raster.GetRasterBand(b + 1)
                    newSample[0, b] = band.ReadAsArray(x, y, 1, 1)

                # Append the new sample to the data array
                data = np.concatenate((data, newSample), axis=0)
                target = np.append(target, target_curr)
                image_fileNames.append(fileName_raster)

            else:
                # For each geometry in the feature
                for g in range(geom.GetGeometryCount()):
                    currGeom = geom.GetGeometryRef(g)
                    if currGeom.GetGeometryName() == 'POLYGON':
                        currGeom = currGeom.GetGeometryRef(0)

                    elif currGeom.GetGeometryName() == 'LINEARRING':
                        points = np.array(currGeom.GetPoints())

                        # Extract the part of the raster limited by the polygon extent
                        topLeftBox_coordXY = np.min(points[:, 0]), np.max(points[:, 1])
                        bottomRightBox_coordXY = np.max(points[:, 0]), np.min(points[:, 1])
                        xOffset = int((topLeftBox_coordXY[0] - ulX) / xRes)
                        yOffset = int((topLeftBox_coordXY[1] - ulY) / yRes)
                        xSize = int((bottomRightBox_coordXY[0] - topLeftBox_coordXY[0]) / xRes) + 1
                        ySize = int((bottomRightBox_coordXY[1] - topLeftBox_coordXY[1]) / yRes) + 1

                        # Extract the pixels inside the box for each band
                        box = np.empty((ySize, xSize, raster.RasterCount))
                        for b in range(raster.RasterCount):
                            band = raster.GetRasterBand(b + 1)
                            box[:, :, b] = band.ReadAsArray(xOffset, yOffset, xSize, ySize)

                        # Save the pixels inside the polygon
                        poly = Polygon(points)
                        for x in range(xSize):
                            for y in range(ySize):
                                xCoord = (xOffset * xRes + ulX) + x * xRes
                                yCoord = (yOffset * yRes + ulY) + y * yRes
                                currPoint = Point(xCoord + xRes / 2.0, yCoord + yRes / 2.0)

                                if currPoint.within(poly):
                                    coordinates.append((xCoord, yCoord))
                                    newSample = np.reshape(box[y, x, :], (1, len(feature_names)))
                                    data = np.concatenate((data, newSample), axis=0)
                                    target = np.append(target, target_curr)
                                    image_fileNames.append(fileName_raster)

                    else:
                        print('ERROR: ' + geom.GetGeometryName() + ' --> invalid geometry type')
                        sys.exit(1)

        raster = None
        shp = None

    # Print the number of samples for each target
    for i, name_target in enumerate(target_names):
        print(name_target + ': ' + str(sum(target == i + 1)))

    # Store the data in a dictionary
    data_set = {'data': data, 'feature_names': feature_names, 'target': target, 'target_names': target_names, 'coordinates': coordinates, 'image_fileNames': image_fileNames}

    # Clean the data by removing samples with no data values
    data_clean = []
    target_clean = []
    coordinates_clean = []
    image_fileNames_clean = []

    for i, d in enumerate(data_set['data']):
        d[d == noDataValue] = np.nan
        if not np.isnan(d).any():
            data_clean.append(d[d != noDataValue])
            target_clean.append(data_set['target'][i])
            coordinates_clean.append(data_set['coordinates'][i])
            image_fileNames_clean = np.append(image_fileNames_clean, data_set['image_fileNames'][i])

    data_set_clean = {'data': data_clean, 'target': target_clean, 'coordinates': coordinates_clean, 'image_fileNames': image_fileNames_clean}

    # Sort the data_set_clean by 'target' values
    df = pd.DataFrame(data_set_clean)
    df = df.sort_values(by='target')
    data_set_clean = df.to_dict('list')
    data_set_clean['feature_names'] = data_set['feature_names']
    data_set_clean['target_names'] = data_set['target_names']

    # Print the number of samples for each target after cleaning
    print('After no data removal:')
    for i, name_target in enumerate(data_set_clean['target_names']):
        print(name_target + ': ' + str(np.sum(np.array(data_set_clean['target']) == i + 1)))

    # Save the cleaned data set to a pickle file
    pickle.dump(data_set_clean, open(output_filename, "wb"))

