import pandas as pd
import time
#import pvlib
#from pvlib import clearsky, atmosphere, solarposition
#from pvlib.location import Location
import os
#from glob import glob
#import xarray as xr
import numpy as np
#import satpy
#from satpy import Scene
#import pyproj as proj
#import multiprocessing as mp
#from tqdm import tqdm
import functions as fn
#import cv2

import warnings
# To suppress some warnings (like proj4 projection and "invalid value encountered in cos"):
warnings.filterwarnings('ignore')

start_time = time.time()

FORECAST_PATH = r"D:\EUMETSAT\forecasts" # Where forecasts are stored
issuetime = pd.to_datetime("2019-03-01 06:12:00") # Because sat imgs are taken at 12, 27, 42 and 57.
filenames = os.listdir(FORECAST_PATH)
results = pd.read_csv(os.path.join(FORECAST_PATH,filenames[0]), sep='\t', parse_dates=True, index_col=0)
print(results)

print("--- %s seconds ---" % (time.time() - start_time))
