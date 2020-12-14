import pandas as pd
import time
import pvlib
from pvlib import clearsky, atmosphere, solarposition
from pvlib.location import Location
import os
from glob import glob
#import xarray as xr
import numpy as np
#import satpy
#from satpy import Scene
#import pyproj as proj
import multiprocessing as mp
from tqdm import tqdm
'''
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from matplotlib import rcParams
from matplotlib import rc
rc('font', size=8)
rc('font', weight='light')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.figsize"] = [2,2]
'''
import functions as fn

import warnings
# To suppress some warnings (like proj4 projection and "invalid value encountered in cos"):
warnings.filterwarnings('ignore')

UNTARRED_DATA_PATH = r"D:\EUMETSAT\untarred"
RESULTS_PATH = r"C:\Users\denva787\Documents\dennis\Greenlytics\Results"
DATA_PATH = r"C:\Users\denva787\Documents\dennis\Greenlytics\Data" # For ghi measurements

# Create dictionaries with latitudes and longitudes
latitudes = {}
latitudes['Tarfala'] = 67.9123
#latitudes['Kiruna'] = 67.8406
latitudes['Lulea'] = 65.5436
latitudes['Umea'] = 63.8111
latitudes['Ostersund'] = 63.1970
latitudes['Storlien'] = 63.3016
latitudes['Borlange'] = 60.4879
latitudes['Hogarna'] = 59.4432
latitudes['Stockholm'] = 59.3534
latitudes['Karlstad'] = 59.3591
latitudes['Nordkoster'] = 58.8920
#latitudes['Norrkoping'] = 58.5824
latitudes['Gothenburg'] = 57.6879
#latitudes['Visby'] = 57.6728
latitudes['Vaxjo'] = 56.9269
latitudes['Hoburg'] = 56.9210
latitudes['Lund'] = 55.7137

longitudes = {}
longitudes['Tarfala'] = 18.6101
#longitudes['Kiruna'] = 20.4106
longitudes['Lulea'] = 22.1113
longitudes['Umea'] = 20.2398
longitudes['Ostersund'] = 14.4798
longitudes['Storlien'] = 12.1241
longitudes['Borlange'] = 15.4290
longitudes['Hogarna'] = 19.5022
longitudes['Stockholm'] = 18.0634
longitudes['Karlstad'] = 13.4719
longitudes['Nordkoster'] = 11.0039
#longitudes['Norrkoping'] = 16.1485
longitudes['Gothenburg'] = 11.9797
#longitudes['Visby'] = 18.3448
longitudes['Vaxjo'] = 14.7305
longitudes['Hoburg'] = 18.1507
longitudes['Lund'] = 13.2124

def update(*a):
    pbar.update()
'''
if __name__ == '__main__':
    start_time = time.time()
    start_date = pd.to_datetime("2019-03-01 06:12:00") # Because sat imgs are taken at 12, 27, 42 and 57.
    end_date = pd.to_datetime("2019-10-31 23:12:00")
    period = pd.date_range(start=start_date,end=end_date,freq='15min')
    pbar = tqdm(total=len(period))
    pool = mp.Pool(processes=32)
    my_res = [pool.apply_async(calc_ghi, args=(datetime,latitudes,longitudes), callback=update) for datetime in period]
    my_res = [p.get() for p in my_res]
    pool.close()
    df = pd.concat(my_res,axis=0)#
    df = df.sort_index(inplace=False)
    df.to_csv(os.path.join(RESULTS_PATH,"GHI.txt"), sep="\t", header=True, index=True, float_format='%.3f')
    print("--- %s seconds ---" % (time.time() - start_time))
'''
wndws = [5,10,15,20,25,30] # Training window lengths 
if __name__ == '__main__':
    for wndw in wndws:
        start_time = time.time()
        start_date = pd.to_datetime("2019-03-01 06:12:00") # Because sat imgs are taken at 12, 27, 42 and 57.
        end_date = pd.to_datetime("2019-10-31 23:12:00")
        period = pd.date_range(start=start_date,end=end_date,freq='15min')
        pbar = tqdm(total=len(period))
        pool = mp.Pool(processes=32)
        my_res = [pool.apply_async(fn.calc_ghi, args=(datetime,latitudes,longitudes,wndw), callback=update) for datetime in period]
        my_res = [p.get() for p in my_res]
        pool.close()
        df = pd.concat(my_res,axis=0)#
        df = df.sort_index(inplace=False)
        filename = "GHI_"+str(wndw)+".txt"
        df.to_csv(os.path.join(RESULTS_PATH,filename), sep="\t", header=True, index=True, float_format='%.3f')
        print("--- %s seconds ---" % (time.time() - start_time))
