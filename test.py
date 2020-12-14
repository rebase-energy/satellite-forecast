import pandas as pd
import time
#import pvlib
#from pvlib import clearsky, atmosphere, solarposition
#from pvlib.location import Location
import os
from glob import glob
#import xarray as xr
import numpy as np
#import satpy
#from satpy import Scene
#import pyproj as proj
#import multiprocessing as mp
#from tqdm import tqdm
import functions as fn
import cv2

import warnings
# To suppress some warnings (like proj4 projection and "invalid value encountered in cos"):
warnings.filterwarnings('ignore')

#def isinteger(x):
#    return np.equal(np.mod(x, 1), 0)

def ci_to_8bit(img,MIN,MAX):
    '''
    Convert the cloud index image to an 8-bit image pixelwise.
    Arguments:
    - img: cloud index image.
    - MIN: minimum value of img.
    - MAX: maximum value of img.
    Output:
    - Returns numpy array containing the cloud index in 8-bit.
    '''
    img = img - MIN
    img = img / (MAX - MIN)
    img = img * 255
    img = img.astype(np.uint8)
    return(img)

def bit_to_ci(img,MIN,MAX):
    '''
    Convert the 8-bit cloud index image back to the original image pixelwise.
    Arguments:
    - img: cloud index image.
    - MIN: minimum value of img.
    - MAX: maximum value of img.
    Output:
    - Returns numpy array containing the cloud index.
    '''
    img = img/255
    #img = np.divide(img,255)#,casting='unsafe'
    #img *= (MAX - MIN)
    img = img * (MAX - MIN)
    img = img + MIN
    #img = img/255
    return(img)

def warp_flow(img, flow):
    '''
    Function that extrapolates an image (img) given optical flow numpy array.
    From: https://github.com/opencv/opencv/blob/master/samples/python/opt_flow.py
    Arguments:
    - img: cloud index image in 8 bit.
    - flow: A numpy array with three dimensions. The first two dimensions are the
      image width and height from the source image. The last dimension is of size
      two and represents the x and y displacements.
    Output:
    - Returns numpy array containing the cloud index moved with flow.
    '''
    h, w = flow.shape[:2] # heigth and width
    flow = -flow # Make it negative (probably because we add integers to "flow" starting in the upper-left corner)
    flow[:,:,0] += np.arange(w) # Add increasing values to each column, repeat for each row. This is because we add the flow to the coordinates
    flow[:,:,1] += np.arange(h)[:,np.newaxis] # Add increasing values to each row, repeat for each column. Similar as above.
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return(res)

start_time = time.time()

UNTARRED_DATA_PATH = r"D:\EUMETSAT\untarred"
RESULTS_PATH = r"C:\Users\denva787\Documents\dennis\Greenlytics\Results"
DATA_PATH = r"C:\Users\denva787\Documents\dennis\Greenlytics\Data" # For ghi measurements

def fc_cloudIndex(issuetime,wndw,K):
    '''
    This function forecasts the cloud index for the specified issue time. It
    uses the blur function to apply the smoothing filter to the image that
    increases with increasing forecast horizon.
    Arguments:
    - datetime: pandas datetime Timestamp.
    - wndw: scalar with the training window length.
    - K: vector of integers representing the forecast horizons (1,...K).
    Output:
    - Returns list of numpy arrays containing the forecast cloud index.
    '''
    img_1 = fn.calc_cloudIndex(issuetime,wndw) # Image at t
    img_0 = fn.calc_cloudIndex(issuetime-pd.Timedelta(15, unit='min'),wndw) # Image at t-1

    if img_1 is None:
        return
    else:
        fcs = [] # List to store forecasts
        flow = cv2.calcOpticalFlowFarneback(img_0, img_1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        for k in K: # Loop over forecast horizons
            if k==1:
                fc = warp_flow(img_1, flow)
                fc = cv2.blur(fc,(2*k,2*k)) # Smoothing as a postprocess step, should be optimized as well)
            else:
                fc = warp_flow(fc, flow) # Forecast from the previous forecast
                fc = cv2.blur(fc,(2*k,2*k)) # Smoothing as a postprocess step, should be optimized as well)
            fcs.append(fc)
        return(fcs)

def fc_clearskyIndex(issuetime,wndw,K):
    '''
    This function calculates the clear-sky index for the specified datetime. The
    nonlinear equation comes from Global horizontal irradiance forecast for Finland
    based on geostationary weather satellite data (eq 6). It has been validated
    with a toy example.
    Arguments:
    - datetime: pandas datetime Timestamp.
    - wndw: scalar with the training window length.
    - K: vector of integers representing the forecast horizons (1,...,K).
    Output:
    - Returns numpy array containing the instantaneous clear-sky index.
    '''
    fcs_cloudindex = fc_cloudIndex(issuetime,wndw,K) # Returns list of numpy arrays
    if fcs_cloudindex is None: # Check whether azimuth is lower than 85, skips if not.
        return
    else:
        fcs = []
        for k in K:
            # Calculate clear-sky index using a nonlinear function:
            # https://stackoverflow.com/questions/45768262/numpy-equivalent-of-if-else-without-loop
            fc_csi = np.where(fcs_cloudindex[k-1] <= -0.2, 1.2, \
                    np.where(fcs_cloudindex[k-1] <= 0.8, 1-fcs_cloudindex[k-1], \
                    np.where(fcs_cloudindex[k-1] <= 1.05, 1.1661-1.781*fcs_cloudindex[k-1]+0.73*fcs_cloudindex[k-1]**2, \
                    np.where(fcs_cloudindex[k-1] > 1.05, 0.09, fcs_cloudindex[k-1]))))
            fcs.append(fc_csi)
        return(fcs)




K=np.arange(1,4,1) # Forecast horizons
issue_time = pd.to_datetime("2019-03-01 12:12:00") # Because sat imgs are taken at 12, 27, 42 and 57.
#valid_time = issue_time+pd.Timedelta(15*k, unit='min')
fcs = fc_clearskyIndex(issue_time,5,K)
print(fcs)
print(type(fcs))
print(len(fcs))


'''
ci_0 = fn.calc_cloudIndex(issue_time_0,5) # Image at t-1
ci_1 = fn.calc_cloudIndex(issue_time-pd.Timedelta(15, unit='min'),5) # Image at t

flow = cv2.calcOpticalFlowFarneback(ci_0, ci_1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
k=1 # Forecast horizon
if k==1:
    fc = warp_flow(ci_1, flow)
else:
    fc = warp_flow(fc, flow) # Forecast from the previous forecast

print(fc)
fc = cv2.blur(fc,(2*k,2*k)) # Smoothing as a postprocess step, should be optimized as well)
print(fc)

fc_csi = np.where(fc <= -0.2, 1.2, \
        np.where(fc <= 0.8, 1-fc, \
        np.where(fc <= 1.05, 1.1661-1.781*fc+0.73*fc**2, \
        np.where(fc > 1.05, 0.09, fc))))
'''


print("--- %s seconds ---" % (time.time() - start_time))
