# Multichannel-Gray-Level-Co-Occurrence-Matrix
Python implementation of multichannel gray level co - occurrence matrix

This function creates the multichannel gray level co - occurrence matrix (k - means clustering based) as described in the paper
'A Multichannel Gray Level Co-Occurrence Matrix for Multi/Hyperspectral Image Texture Representation' by X. Huang, 
X. Liu and L. Zhang (2014). 

Requirements:
- numpy
- pandas
- rasterio
- scipy
- skimage
- sklearn
- tqdm

Test Image (courtesy of USGS NAIP):

![alt text](https://github.com/ThomasWangWeiHong/Multichannel-Gray-Level-Co---Occurrence-Matrix/blob/master/Test_Image.JPG)

MGLCM Mean Contrast Image:

![alt text](https://github.com/ThomasWangWeiHong/Multichannel-Gray-Level-Co---Occurrence-Matrix/blob/master/MGLCM_Mean_Contrast.JPG)

MGLCM Mean Energy Image:

![alt text](https://github.com/ThomasWangWeiHong/Multichannel-Gray-Level-Co---Occurrence-Matrix/blob/master/MGLCM_Mean_Energy.JPG)

MGLCM Mean Homogeneity Image:

![alt text](https://github.com/ThomasWangWeiHong/Multichannel-Gray-Level-Co---Occurrence-Matrix/blob/master/MGLCM_Mean_Homogeneity.JPG)

MGLCM Mean Dissimilarity Image:

![alt text](https://github.com/ThomasWangWeiHong/Multichannel-Gray-Level-Co---Occurrence-Matrix/blob/master/MGLCM_Mean_Dissimilarity.JPG)
