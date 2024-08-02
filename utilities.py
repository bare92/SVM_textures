#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:30:25 2024

@author: rbarella
"""


def open_image (image_path):
    
    from osgeo import gdal
    import numpy as np

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

def save_image (image_to_save, path_to_save, driver_name, datatype, geotransform, proj, NoDataValue=None):
    
 
    '''
    adfGeoTransform[0] / * top left x * /
    adfGeoTransform[1] / * w - e pixel resolution * /
    adfGeoTransform[2] / * rotation, 0 if image is "north up" * /
    adfGeoTransform[3] / * top left y * /
    adfGeoTransform[4] / * rotation, 0 if image is "north up" * /
    adfGeoTransform[5] / * n - s pixel resolution * /
    

    enum  	GDALDataType {
  GDT_Unknown = 0, GDT_Byte = 1, GDT_UInt16 = 2, GDT_Int16 = 3,
  GDT_UInt32 = 4, GDT_Int32 = 5, GDT_Float32 = 6, GDT_Float64 = 7,
  GDT_CInt16 = 8, GDT_CInt32 = 9, GDT_CFloat32 = 10, GDT_CFloat64 = 11,
  GDT_TypeCount = 12
}
    '''
    from osgeo import gdal
    import numpy as np
    
    
    driver = gdal.GetDriverByName(driver_name)

    if len(np.shape(image_to_save)) == 2:
        bands = 1
        cols = np.shape(image_to_save)[1]
        rows = np.shape(image_to_save)[0]

    if len(np.shape(image_to_save)) > 2:
        bands = np.shape(image_to_save)[0]
        cols = np.shape(image_to_save)[2]
        rows = np.shape(image_to_save)[1]

    outDataset = driver.Create(path_to_save, cols, rows, bands, datatype)

    outDataset.SetGeoTransform(geotransform)

    if proj != None:
        outDataset.SetProjection(proj)

    if bands > 1:

        for i in range(1, bands + 1):
            outDataset.GetRasterBand(i).WriteArray(image_to_save[(i - 1), :, :], 0, 0)
            if NoDataValue != None:
                outDataset.GetRasterBand(i).SetNoDataValue(NoDataValue)

    else:
        outDataset.GetRasterBand(1).WriteArray(image_to_save, 0, 0)
        if NoDataValue != None:
                outDataset.GetRasterBand(1).SetNoDataValue(NoDataValue)
        

    outDataset = None

    print('Image Saved')

    return;

def create_vrt(file_list, band_name_list, resolution = 0.25, overwrite = False):
    
    import os
    from osgeo import gdal
    
    vrtname = os.path.join(os.path.dirname(file_list[0]), '00_features.vrt')
   
    if os.path.exists(vrtname) and overwrite == False:
        print(vrtname + ' has already been created')
        
    else:
        print('Elaborating %s' % vrtname)
        
       
            
            
           


        file_string = " ".join(file_list)
        
        cmd = "gdalbuildvrt -separate -r bilinear -tr " + str(resolution) + ' ' + str(resolution) + ' ' + vrtname + ' ' + file_string
        print(cmd)
        os.system(cmd)


        # Set Band Description
    
        VRT_dataset = gdal.Open(vrtname, gdal.GA_Update)
        for band_name, idx in zip(band_name_list, range(1, len(band_name_list) + 1)):
            VRT_dataset.GetRasterBand(idx).SetDescription(band_name)
    
        VRT_dataset = None
   