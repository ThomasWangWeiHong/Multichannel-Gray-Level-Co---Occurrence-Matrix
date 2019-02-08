import numpy as np
import pandas as pd
from osgeo import gdal
from scipy.stats import rankdata
from skimage.feature import greycomatrix, greycoprops
from sklearn.cluster import KMeans
from tqdm import tqdm



def mglcm_generation(input_filename, output_filename, cluster_size, window_size, displ_dist):
    """ 
    This function generates the multichannel GLCM for multispectral image texture representation via clustering - based 
    quantization. Full technical details can be found in the paper 'A Multichannel Gray Level Co - Occurence Matrix for 
    Multi/Hyperspectral Image Texture Representation' by X. Huang, X. Liu and L. Zhang (2014). This implementation is slightly
    different from the one in the paper, with the inverse difference feature replaced by the 'homogeneity' feature in the 
    skimage module, and the entropy feature replaced by the 'dissimilarity' feature in the skimage module. This is done to 
    speed up the calculation process.
    
    Inputs:
    - input_filename: File path of input multi - channel (4 - channels) image for which the multichannel GLCM is to be 
                      calculated
    - output_filename: File path of output multichannel GLCM which is to be written to file
    - cluster_size: Number of clusters for which the multi - channel input image should be partitioned into
    - window_size: Size of window to be used for the calculation of the GLCM
    - displ_dist: Magnitude of displacement vector to be used for the GLCM calculation
    
    Outputs:
    - glcm_measures: 
    
    
    """
    
    if (window_size % 2 == 0) :
        raise ValueError('Please input an odd number for window_size.')
    
    if (window_size + 1) < displ_dist:
        raise ValueError('Please make sure that displ_dist is less than or equal to window_size.')
    
    buffer = int((window_size - 1) / 2)    
    
    img = np.transpose(gdal.Open(input_filename).ReadAsArray(), [1, 2, 0])
    img_2d = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    kmeans = KMeans(n_clusters = cluster_size, max_iter = 10000, random_state = 2018).fit(img_2d)
    img_total = np.sum(img, axis = 2)
    
    df = pd.DataFrame({'Sum': img_total.flatten(), 
                       'Labels': kmeans.labels_})
    df_inter = df.pivot_table(index = ['Labels'], aggfunc = np.mean)
    df_inter_1 = df_inter.reset_index()
    df_inter_1.columns = ['Labels', 'Mean']
    df_inter_1['Quantized'] = rankdata(df_inter_1.Mean.values, method = 'dense')
    df_final = df.merge(df_inter_1, how = 'left', on = 'Labels')
    quantized = df_final.Quantized.values.reshape(img.shape[0], img.shape[1]) - 1
    
    quantized_padded = np.zeros(((quantized.shape[0] + 2 * buffer), (quantized.shape[1] + 2 * buffer)))
    quantized_padded[buffer : (buffer + quantized.shape[0]), buffer : (buffer + quantized.shape[1])] = quantized
    
    glcm_measures = np.zeros((img.shape[0], img.shape[1], 4))
   
    for i in tqdm(range(buffer, img.shape[0] - buffer, 1), mininterval = 600) :            
        for j in range(buffer, img.shape[1] - buffer, 1) :                                                                                                                                   
            array = quantized_padded[(i - buffer) : (i + buffer + 1), (j - buffer) : (j + buffer + 1)].astype(int)
            glcm = greycomatrix(array, [displ_dist], [0, np.pi / 4, np.pi / 2, (3 / 4) * np.pi], levels = cluster_size, 
                               normed = True)
            contrast = greycoprops(glcm, prop = 'contrast')
            energy = greycoprops(glcm, prop = 'ASM')
            homogeneity = greycoprops(glcm, prop = 'homogeneity')
            dissimilarity = greycoprops(glcm, prop = 'dissimilarity')
            glcm_measures[i - buffer, j - buffer, 0] = contrast.mean()
            glcm_measures[i - buffer, j - buffer, 1] = energy.mean()
            glcm_measures[i - buffer, j - buffer, 2] = homogeneity.mean()
            glcm_measures[i - buffer, j - buffer, 3] = dissimilarity.mean()
                    
    input_dataset = gdal.Open(input_filename)
    input_band = input_dataset.GetRasterBand(1)
    gtiff_driver = gdal.GetDriverByName('GTiff')
    output_dataset = gtiff_driver.Create(output_filename, input_band.XSize, input_band.YSize, 4, gdal.GDT_Float32)
    output_dataset.SetProjection(input_dataset.GetProjection())
    output_dataset.SetGeoTransform(input_dataset.GetGeoTransform())
    for i in range(1, 5):
        output_dataset.GetRasterBand(i).WriteArray(glcm_measures[:, :, i - 1])    
    output_dataset.FlushCache()
    for i in range(1, 5):
        output_dataset.GetRasterBand(i).ComputeStatistics(False)
    del output_dataset
    
    return glcm_measures
