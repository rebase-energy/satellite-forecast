'''
Author: Dennis van der Meer
Email: denniswillemvandermeer@gmail.com

This script generates GHI forecasts over a specific period, using the functions
in functions.py. The second part of the script loops over training window lengths
to find out what window produces the most accurate results.

'''
import pandas as pd
import time
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import functions as fn
import warnings
# To suppress some warnings (like proj4 projection and "invalid value encountered in cos"):
warnings.filterwarnings('ignore')

def update(*a):
    pbar.update()

if __name__ == '__main__':
    start_time = time.time()
    window = 15
    K=np.arange(1,25,1) # Forecast horizons
    start_date = pd.to_datetime("2019-03-01 06:12:00") # Because sat imgs are taken at 12, 27, 42 and 57.
    end_date = pd.to_datetime("2019-10-31 23:12:00")
    period = pd.date_range(start=start_date,end=end_date,freq='15min')
    pbar = tqdm(total=len(period))
    pool = mp.Pool(processes=32)
    #my_res = [pool.apply_async(fn.fc_ghi, args=(issue_time,fn.latitudes,fn.longitudes,window,K), callback=update) for issue_time in period]
    my_res = [pool.apply_async(fn.fc_ghi_ps, args=(issue_time,fn.latitudes,fn.longitudes,window,K), callback=update) for issue_time in period]
    my_res = [p.get() for p in my_res]
    pool.close()
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
'''
