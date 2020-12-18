import pandas as pd
import time
import pvlib
from pvlib import clearsky, atmosphere, solarposition
from pvlib.location import Location
import os
from glob import glob
import xarray as xr
import numpy as np
import satpy
from satpy import Scene
import pyproj as proj
#import multiprocessing as mp
from tqdm import tqdm
import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from matplotlib import rcParams
from matplotlib import rc
rc('font', size=8)
rc('font', weight='light')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.figsize"] = [2,2]

import warnings
# To suppress some warnings (like proj4 projection and "invalid value encountered in cos"):
warnings.filterwarnings('ignore')

UNTARRED_DATA_PATH = r"D:\EUMETSAT\untarred"
RESULTS_PATH = r"C:\Users\denva787\Documents\dennis\Greenlytics\Results"
DATA_PATH = r"C:\Users\denva787\Documents\dennis\Greenlytics\Data" # For ghi measurements
FORECAST_PATH = r"D:\EUMETSAT\forecasts" # Where forecasts are stored
# Load GHI data:
ghi_m = pd.read_csv(os.path.join(DATA_PATH,"smhi_GHI.txt"), sep="\t", parse_dates=[0], index_col=0) # measured ghi

########################################################################################
############################### Forecast GHI functions #################################
########################################################################################

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

def fc_cloudIndex(issuetime,wndw,K):
    '''
    This function forecasts the cloud index for the specified issue time. It
    uses the blur function to apply the smoothing filter to the image that
    increases with increasing forecast horizon (ideally, the filter box size
    is optimized).
    Arguments:
    - issuetime: pandas datetime Timestamp.
    - wndw: scalar with the training window length.
    - K: vector of integers representing the forecast horizons (1,...K).
    Output:
    - Returns list of K numpy arrays containing the forecast cloud index.
    '''
    img_1 = calc_cloudIndex(issuetime,wndw) # Image at t
    img_0 = calc_cloudIndex(issuetime-pd.Timedelta(15, unit='min'),wndw) # Image at t-1

    if (img_0 is None) or (img_1 is None): # check if either image exist
        return
    else:
        fcs = [] # List to store forecasts
        flow = cv2.calcOpticalFlowFarneback(img_0, img_1, None, 0.5, 3, 15, 3, 5, 1.2, 0) # args can be optimized
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
    This function forecasts the clear-sky index for the specified issue time. It
    uses the blur function to apply the smoothing filter to the image that
    increases with increasing forecast horizon (ideally, the filter box size
    is optimized).
    Arguments:
    - issuetime: pandas datetime Timestamp.
    - wndw: scalar with the training window length.
    - K: vector of integers representing the forecast horizons (1,...,K).
    Output:
    - Returns list of K numpy arrays containing the forecast clear-sky index.
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

def fc_ghi(issuetime,latitudes,longitudes,wndw,K):
    '''
    Forecast the GHI starting at a certain issue time for certain combinations
    of latitudes and longitudes. The function first generates a forecast for
    1,...,K forecast horizons of the clear-sky index. Then, the matrix indices
    that belong to the coordinates of the stations are calculated, which is used
    to select the right clear-sky index.
    Then, the function loops over the locations in the dictionary of latitudes
    and calculates the clear-sky irradiance and then the GHI measured by the
    satellite. The GHI is then combined with the measured GHI, which finally
    results in a data frame with K rows and columns: Az_loc, Elev_loc,
    G_loc, G_sat_loc.
    Arguments:
    - issuetime: pandas datetime Timestamp.
    - lats and lons: dictionaries containing the coordinates of sites in
      the coordinate reference system WGS84 (standard coordinate system).
    - wndw: scalar with the training window length
    - K: vector of integers representing the forecast horizons (1,...,K).
    Output:
    - Data frame with K rows and columns: Az_loc, Elev_loc, G_loc, G_sat_loc.
    '''
    # Calculate clear-sky index on date
    fcs = fc_clearskyIndex(issuetime,wndw,K) # list of K numpy arrays
    if fcs is None: # Check whether azimuth is lower than 85, skips if not.
        return
    else:
        # Find indices closest to lats and lons
        indices = find_indices(latitudes,longitudes)
        ys = indices['y'].to_list() # Get the y-coordinates
        xs = indices['x'].to_list() # Get the x-coordinates
        lst_of_dfs_outer = []
        for k in K:
            lst_of_csi = fcs[k-1][ys,xs] # Take the forecasts at the indices
            d = {'csi':lst_of_csi,'Location':list(latitudes.keys())}
            csi_at_loc = pd.DataFrame(d)
            # I need to use a rolling window for the clear sky irradiance because the image is
            # taken in an interval of 15 minutes while the clear-sky irradiance is instantaneous.
            # Valid time of forecast:
            valid_time = issuetime+pd.Timedelta(15*k, unit='min')
            time_idx = pd.date_range(start=valid_time, freq='15min', periods=1, tz='UTC').round('15min')
            # Time interval of the image:
            time_int = pd.date_range(end=valid_time, freq='1min', periods=15, tz='UTC')#.round('15min')
            lst_of_dfs_inner = []
            for loc in latitudes.keys(): # Loop over the locations
                lat,lon=latitudes[loc],longitudes[loc]
                current_loc = Location(lat,lon,'UTC',0)
                clearsky = current_loc.get_clearsky(time_int).resample('15min',closed='right',label='right').mean() # DataFrame with two rows.
                clearsky = clearsky.loc[time_idx]['ghi'] # DataFrame with one row, take ghi
                ghi_sat = clearsky.values * csi_at_loc.loc[csi_at_loc['Location'] == loc]['csi'].values
                # Read the station specific columns:
                ghi_m_loc = ghi_m.filter(regex=(loc)) # ghi_m is a .txt file with measurements
                ghi_loc =  ghi_m_loc.loc[time_idx]
                ghi_loc['G_sat_{}'.format(loc)] = ghi_sat
                #ghi_loc['csi_{}'.format(loc)] = csi_at_loc.loc[csi_at_loc['Location'] == loc]['csi'].values
                lst_of_dfs_inner.append(ghi_loc)
            df = pd.concat(lst_of_dfs_inner,axis=1) # DF with 1 row and column for each location
            lst_of_dfs_outer.append(df)
        fc = pd.concat(lst_of_dfs_outer,axis=0).round(2) # DF with K rows and column for each location
        #dir_name = os.path.join(FORECAST_PATH, issuetime.strftime("%Y/%m/%d/%H/%M"))
        #if not os.path.exists(dir_name):
        #    os.makedirs(dir_name)
        #fc.to_csv(os.path.join(dir_name,"fc.txt"), sep="\t")
        fname = os.path.join(FORECAST_PATH, issuetime.strftime("%Y%m%dT%H%M")+".txt")
        fc.to_csv(fname, sep="\t")

########################################################################################
################################ Compute GHI functions #################################
########################################################################################

def calc_historical_rho(datetime,wndw):
    '''
    This function returns ground and cloud reflectance matrices for the datetime
    entered by the user. The first for loop iterates over the time instances,
    which is defined by tr_window_length. Although currently fixed, tr_window_length
    could be made a function argument.
    The function appends all the preceding satellite images and calculates the
    5% and 95% for the rho_ground and rho_cloud, respectively.
    Arguments:
    - period: pandas datetime Timestamp.
    - wndw: scalar with the training window length (for validation purposes)
    Output:
    - Returns numpy arrays of rho_ground and rho_cloud.
    '''
    dirs = create_historical_filenames(datetime,wndw)
    if dirs is None: # Check whether azimuth is lower than 85, skips if not.
        return
    else:
        sat_imgs = []
        for filename in dirs: # Loop over the trailing window tr_window_length.
            filename = glob(filename+'*.nat') # Because dirs was a list of lists extract first element.
            if not filename: # If filename is empty, continue to the next iteration.
                continue
            else:
                global_scene = Scene(reader="seviri_l1b_native", filenames=filename)
                # Load the HRV channel:
                global_scene.load(['HRV'])
                # Resample:
                local_scene = global_scene.resample("scan1",radius_of_influence=50e3,resampler='nearest',neighbours=16) # nearest='bilinear',cache_dir=REFLECTANCE_DATA_PATH
                # radius_of_influence: maximum distance to search for a neighbour for each point in the target grid
                hrv = local_scene['HRV'] # xarray.DataArray
                # Add time dimension and coordinate so I can easily slice the resulting xArray:
                hrv = hrv.assign_coords(time=hrv.attrs['end_time'])
                hrv = hrv.expand_dims('time')
                sat_imgs.append(hrv)
        # Concatenate the list above on the 'time' dimension
        combined = xr.concat(sat_imgs, dim='time')
        # Extract the reflectances (this takes a while)
        reflectance = combined.values
        # Calculate the lowest 5% for each pixel
        rho_ground = np.percentile(reflectance,5,0)
        # Calculate the lowest 5% for each pixel (added 2020-09-14)
        rho_cloud = np.percentile(reflectance,95,0)
        return(rho_ground,rho_cloud)

def calc_instant_rho(datetime,wndw):
    '''
    This function returns returns the reflectance for the datetime entered by the
    user (achieved by setting "historical" to False).
    Arguments:
    - period: pandas datetime Timestamp.
    - wndw: scalar with the training window length (for validation purposes)
    Output:
    - Returns numpy array containing the instantaneous reflectance.
    '''
    dirs = create_instant_filenames(datetime,wndw)
    if dirs is None: # Check whether azimuth is lower than 85, skips if not.
        return
    else:
        filename = glob(dirs+'*.nat') # Because dirs was a list of lists extract first element.
        if not filename: # If filename is empty, continue to the next iteration.
            print("This satellite image is missing.")
        else:
            global_scene = Scene(reader="seviri_l1b_native", filenames=filename)
            # Load the HRV channel:
            global_scene.load(['HRV'])
            # Resample:
            local_scene = global_scene.resample("scan1",radius_of_influence=50e3,resampler='nearest',neighbours=16) # nearest='bilinear',cache_dir=REFLECTANCE_DATA_PATH
            #local_scene.show('HRV') # If I want to produce a picture
            # radius_of_influence: maximum distance to search for a neighbour for each point in the target grid
            hrv = local_scene['HRV'] # xarray.DataArray
            return(hrv.values)

def calc_cloudIndex(datetime,wndw):
    '''
    This function calculates the cloud index for the specified datetime.
    Arguments:
    - datetime: pandas datetime Timestamp.
    - wndw: scalar with the training window length (for validation purposes)
    Output:
    - Returns numpy array containing the instantaneous cloud index.
    '''
    rho_instant = calc_instant_rho(datetime,wndw)
    if rho_instant is None: # Check whether azimuth is lower than 85, skips if not.
        return
    else:
        rho_ground, rho_cloud = calc_historical_rho(datetime,wndw)
        cloudIndex = (rho_instant - rho_ground) / (rho_cloud - rho_ground)
        return(cloudIndex)

def calc_clearskyIndex(datetime,wndw):
    '''
    This function calculates the clear-sky index for the specified datetime. The
    nonlinear equation comes from Global horizontal irradiance forecast for Finland
    based on geostationary weather satellite data (eq 6). It has been validated
    with a toy example.
    Arguments:
    - datetime: pandas datetime Timestamp.
    - wndw: scalar with the training window length (for validation purposes)
    Output:
    - Returns numpy array containing the instantaneous clear-sky index.
    '''
    cloudIndex = calc_cloudIndex(datetime,wndw)
    if cloudIndex is None: # Check whether azimuth is lower than 85, skips if not.
        return
    else:
        # Calculate clear-sky index using a nonlinear function:
        # https://stackoverflow.com/questions/45768262/numpy-equivalent-of-if-else-without-loop
        clearSkyIndex = np.where(cloudIndex <= -0.2, 1.2, \
                np.where(cloudIndex <= 0.8, 1-cloudIndex, \
                np.where(cloudIndex <= 1.05, 1.1661-1.781*cloudIndex+0.73*cloudIndex**2, \
                np.where(cloudIndex > 1.05, 0.09, cloudIndex))))
        return(clearSkyIndex)

def calc_ghi(datetime,latitudes,longitudes,wndw):
    '''
    Calculate the GHI at a certain datetime for certain combinations of latitudes
    and longitudes. The function first calculates the clear-sky index at the
    datetime. Then, the matrix indices that belong to the coordinates of the
    stations are calculated, which is used to select the right clear-sky index.
    Then, the function loops over the locations in the dictionary of latitudes
    and calculates the clear-sky irradiance and then the GHI measured by the
    satellite. The GHI is then combined with the measured GHI, which finally
    results in a data frame with one row and columns: Az_loc, Elev_loc,
    G_loc, G_sat_loc.
    Arguments:
    - datetime: pandas datetime Timestamp.
    - lats and lons: dictionaries containing the coordinates of sites in
      the coordinate reference system WGS84 (standard coordinate system).
    - wndw: scalar with the training window length (for validation purposes)
    Output:
    - Data frame with one row and columns: Az_loc, Elev_loc, G_loc, G_sat_loc.
    '''
    # Calculate clear-sky index on date
    csi = calc_clearskyIndex(datetime,wndw) # Note that the output is (y,x)
    if csi is None: # Check whether azimuth is lower than 85, skips if not.
        return
    else:
        # Find indices closest to lats and lons
        indices = find_indices(latitudes,longitudes)
        ys = indices['y'].to_list() # Get the y-coordinates
        xs = indices['x'].to_list() # Get the x-coordinates
        lst_of_csi = csi[ys,xs]
        d = {'csi':lst_of_csi,'Location':list(latitudes.keys())}
        csi_at_loc = pd.DataFrame(d)
        # I need to use a rolling window for the clear sky irradiance because the image is
        # taken in an interval of 15 minutes while the clear-sky irradiance is instantaneous.
        # Current time:
        time_idx = pd.date_range(start=datetime, freq='15min', periods=1, tz='UTC').round('15min')
        # Time interval of the image:
        time_int = pd.date_range(end=datetime, freq='1min', periods=15, tz='UTC')#.round('15min')
        lst_of_dfs = []
        for loc in latitudes.keys(): # Loop over the locations
            lat,lon=latitudes[loc],longitudes[loc]
            current_loc = Location(lat,lon,'UTC',0)
            clearsky = current_loc.get_clearsky(time_int).resample('15min',closed='right',label='right').mean() # DataFrame with two rows.
            clearsky = clearsky.loc[time_idx]['ghi'] # DataFrame with one row, take ghi
            ghi_sat = clearsky.values * csi_at_loc.loc[csi_at_loc['Location'] == loc]['csi'].values
            # Read the station specific columns:
            ghi_m_loc = ghi_m.filter(regex=(loc)) # ghi_m is a .txt file with measurements
            ghi_loc =  ghi_m_loc.loc[time_idx]
            ghi_loc['G_sat_{}'.format(loc)] = ghi_sat
            ghi_loc['csi_{}'.format(loc)] = csi_at_loc.loc[csi_at_loc['Location'] == loc]['csi'].values
            lst_of_dfs.append(ghi_loc)
        df = pd.concat(lst_of_dfs,axis=1)
        return(df)

########################################################################################
################################### Helper functions ###################################
########################################################################################

def define_trailing_and_target(datetime,wndw):
    '''
    When a user wants to estimate GHI from satellite images during a time instant,
    this function subsets first looks at the past tr_window_length time instances
    to check if the azimuth is higher than 85 degrees. The function builds a list
    of time instances with daily frequency and calculates the average zenith angle
    over the list of time instances. If the average is below 85 degrees, this
    function returns a dictionary with keys: (i) trailing window datetimes and
    (ii) the target datetime.
    Specifically, the function looks at each date in the period,
    creates a date_range of the preceding "tr_window_length" days at
    that time and calculates the average zenith angle over the days.
    If the zenith angle is lower than 85 degrees, the date is in-
    cluded.
    Arguments:
    - datetime: pandas datetime Timestap.
    - wndw: scalar with the training window length (for validation purposes)
    Output:
    - A dictionary with keys tr_window and targets. tr_window contains a list of
      DatetimeIndexes that satisfy the condition and for which we want to estimate
      GHI. target contains a list with Timestamp with the target datetime (argument).
    '''
    tr_window_length = wndw # 20 seems to work nicely.
    latitude, longitude, tz, altitude, name = 62, 16, None, 0, 'midSweden'
    res=[] # List containing the DatetimeIndexes
    tgts=[]
    target = datetime
    start = datetime-pd.Timedelta(tr_window_length, unit='d')
    end = datetime-pd.Timedelta(1, unit='d')
    tr_window = pd.date_range(start=start, end=end, freq='d', tz=tz)
    solpos = pvlib.solarposition.get_solarposition(tr_window, latitude, longitude)
    apparent_zenith = solpos['apparent_zenith']
    if apparent_zenith.mean() <= 85:
        return{'tr_window': tr_window, 'targets': tr_window[-1]+pd.Timedelta(1, unit='d')}

def create_historical_filenames(datetime,wndw):
    '''
    This function first uses the function define_trailing_and_target to create a
    a list of filenames for the trailing window (historical=True) or the target
    (historical=False), if the datetime is valid (depending on the azimuth).
    Arguments:
    - datetime: pandas datetime Timestamp.
    - wndw: scalar with the training window length (for validation purposes)
    Output:
    - A list of lists containing the directories where the untarred satellite images
      can be found. One list for each valid time step in the subsetted period.
    '''
    checked_date = define_trailing_and_target(datetime,wndw)
    if checked_date is None: # Check whether azimuth is lower than 85, skips if not.
        return
    else:
        untarred_data_filenames = []
        month = checked_date.get("tr_window").month
        day = checked_date.get("tr_window").day
        hour = checked_date.get("tr_window").hour[0] # Take 0th element because these are constant for each iteration
        minute = checked_date.get("tr_window").minute[0] # Take 0th element because these are constant for each iteration
        lst = [month,day] # These are the variables, hour and minute is constant for each iteration.
        arr = np.asarray(lst) # Convert to array with first row of month and second row of day.
        md_range = list(map(tuple, arr.T)) # Create a list of tuples of the columns of arr (i.e. month and day).
        tmp = [os.path.join(UNTARRED_DATA_PATH, "2019/{:02d}/{:02d}/{:02d}/{:02d}/".format(month,day,hour,minute)) for month,day in md_range]
        return(tmp)

def create_instant_filenames(datetime,wndw):
    '''
    This function first uses the function define_trailing_and_target to create a
    a list of filenames for the trailing window (historical=True) or the target
    (historical=False), if the datetime is valid (depending on the azimuth).
    Arguments:
    - datetime: pandas datetime Timestamp.
    - wndw: scalar with the training window length (for validation purposes)
    Output:
    - A list of lists containing the directories where the untarred satellite images
      can be found. One list for each valid time step in the subsetted period.
    '''
    checked_date = define_trailing_and_target(datetime,wndw)
    if checked_date is None: # Check whether azimuth is lower than 85, skips if not.
        return
    else:
        untarred_data_filenames = []
        month = checked_date.get("targets").month
        day = checked_date.get("targets").day
        hour = checked_date.get("targets").hour # Take 0th element because these are constant for each iteration
        minute = checked_date.get("targets").minute # Take 0th element because these are constant for each iteration
        tmp = os.path.join(UNTARRED_DATA_PATH, "2019/{:02d}/{:02d}/{:02d}/{:02d}/".format(month,day,hour,minute))
        return(tmp)

def find_indices(latitudes,longitudes):
    '''
    Function that returns a numpy array with matrix indices closest to the
    coordinates in lats and lons. Lats and lons are dictionaries of coordinates
    in the WGS84 geographic and are therefore first transformed to the coordinate
    reference system of the satellite.
    Arguments:
    - lats and lons: dictionaries containing the coordinates of sites in
      the coordinate reference system WGS84 (standard coordinate system).
    Output:
    - Numpy array with y indices in the first column and x indices in the second
      column.
    '''
    date = pd.to_datetime("2019-03-01 08:12:00") # One date will suffice
    PROJ = get_projection(date) # The projection is the same for each day.
    # Create a meshgrid of the coordinates
    xx,yy = np.meshgrid(PROJ['x'].values,PROJ['y'].values)
    crs_4326 = proj.Proj(init='epsg:4326') # assuming you're using WGS84 geographic
    crs_proj = proj.Proj(str(PROJ['crs'].values))
    x, y = proj.transform(crs_4326, crs_proj, list(longitudes.values()), list(latitudes.values()))

    idxs = []
    idys = []
    for i in range(len(x)): # Loop over all coorinates
        distance = (yy-y[i])**2 + (xx-x[i])**2 # Distance closest to x and y on xx and yy
        idy,idx = np.where(distance==distance.min())
        idys.append(idy[0]) # The first element of an array of length 1
        idxs.append(idx[0]) # The first element of an array of length 1
    d = {'y':idys,'x':idxs,'Location':list(latitudes.keys())}
    df = pd.DataFrame(data=d)
    return(df)

def get_coordinates(datetime,wndw):
    '''
    Function to get the coordinates of the resampled satellite image (for plotting
    purposes).
    Arguments:
    - period: pandas datetime Timestamp.
    - wndw: scalar with the training window length (for validation purposes)
    Output:
    - Returns crs.
    '''
    dirs = create_instant_filenames(datetime,wndw)
    filename = glob(dirs+'*.nat') # Because dirs was a list of lists extract first element.
    if not filename: # If filename is empty, continue to the next iteration.
        print("This satellite image is missing.")
    else:
        global_scene = Scene(reader="seviri_l1b_native", filenames=filename)
        # Load the HRV channel:
        global_scene.load(['HRV'])
        # Resample:
        local_scene = global_scene.resample("scan1",radius_of_influence=50e3,resampler='nearest',neighbours=16) # nearest='bilinear',cache_dir=REFLECTANCE_DATA_PATH
        hrv = local_scene['HRV'] # Extract reflectance values into xarray.DataArray
        # radius_of_influence: maximum distance to search for a neighbour for each point in the target grid
        crs = local_scene['HRV'].attrs['area'].to_cartopy_crs()
        return(crs)

def get_projection(datetime):
    '''
    Function to get the projection of the resampled satellite image (for lon,lat
    calculation). There must be a way to store this globally instead of having
    to run this function every time.
    2020-11-09: I changed the function so that it only gets the projection for
    one datetime and is therefore quite fast.
    Arguments:
    - period: pandas datetime Timestamp.
    Output:
    - Returns projection of the image.
    '''
    month = datetime.month
    day = datetime.day
    hour = datetime.hour
    minute = datetime.minute
    dirs = os.path.join(UNTARRED_DATA_PATH, "2019/{:02d}/{:02d}/{:02d}/{:02d}/".format(month,day,hour,minute))
    filename = glob(dirs+'*.nat') # Because dirs was a list of lists extract first element.
    if not filename: # If filename is empty, continue to the next iteration.
        print("This satellite image is missing.")
    else:
        global_scene = Scene(reader="seviri_l1b_native", filenames=filename)
        # Load the HRV channel:
        global_scene.load(['HRV'])
        # Resample:
        local_scene = global_scene.resample("scan1",radius_of_influence=50e3,resampler='nearest',neighbours=16) # nearest='bilinear',cache_dir=REFLECTANCE_DATA_PATH
        hrv = local_scene['HRV'] # Extract reflectance values into xarray.DataArray
        #proj = str(hrv.coords['crs'].values)
        proj = hrv.coords
        return(proj)

########################################################################################
################################## Plotting functions ##################################
########################################################################################

def plot_CSI(datetime,wndw):
    '''
    Function to plot the clear-sky index at datetime. Currently, this function is
    not very smart as it has to run twice but it works for now.
    Arguments:
    - period: pandas datetime Timestamp.
    - wndw: scalar with the training window length (for validation purposes)
    Output:
    - Returns numpy array containing the instantaneous reflectance.
    '''
    crs = get_coordinates(datetime,wndw)
    img = calc_clearskyIndex(datetime,wndw)
    ax = plt.axes(projection=crs)
    ax.coastlines(color='grey')
    #ax.gridlines()
    ax.set_global()
    plt.imshow(img, transform=crs, extent=crs.bounds, origin='upper', cmap='plasma')
    cbar = plt.colorbar(pad=0.025, orientation="horizontal", fraction=0.046)
    #cbar.ax.tick_params(labelsize="small")
    cbar.set_label("Clear-sky index (-)")#,size="small")
    #plt.show()
    plt.savefig("clearskyIndex.pdf", bbox_inches='tight')

def plot_Cloudindex(datetime,wndw):
    '''
    Function to plot the cloud index at datetime. Currently, this function is
    not very smart as it has to run twice but it works for now.
    Arguments:
    - period: pandas datetime Timestamp.
    - wndw: scalar with the training window length (for validation purposes)
    Output:
    - Returns numpy array containing the instantaneous reflectance.
    '''
    crs = get_coordinates(datetime,wndw)
    img = calc_cloudIndex(datetime,wndw)
    ax = plt.axes(projection=crs)
    ax.coastlines(color='grey')
    #ax.gridlines()
    ax.set_global()
    plt.imshow(img, transform=crs, extent=crs.bounds, origin='upper', cmap='viridis')
    cbar = plt.colorbar(pad=0.025, orientation="horizontal", fraction=0.046)
    #cbar.ax.tick_params(labelsize="small")
    cbar.set_label("Cloud index (-)")#,size="small")
    #plt.show()
    plt.savefig("cloudIndex.pdf", bbox_inches='tight')

def plot_sat_img(datetime,wndw):
    '''
    This function plots the original satellite image at datetime.
    Arguments:
    - period: pandas datetime Timestamp.
    - wndw: scalar with the training window length (for validation purposes)
    Output:
    - Saves satellite image.
    '''
    dirs = create_instant_filenames(datetime,wndw)
    filename = glob(dirs+'*.nat') # Because dirs was a list of lists extract first element.
    if not filename: # If filename is empty, continue to the next iteration.
        print("This satellite image is missing.")
    else:

        global_scene = Scene(reader="seviri_l1b_native", filenames=filename)
        # Load the HRV channel:
        global_scene.load(['HRV'])
        # Resample:
        local_scene = global_scene.resample("scan1",radius_of_influence=50e3,resampler='nearest',neighbours=16) # nearest='bilinear',cache_dir=REFLECTANCE_DATA_PATH
        # Get coordinate reference system from satellite
        crs = local_scene['HRV'].attrs['area'].to_cartopy_crs()


        PROJ = local_scene['HRV'].coords # Extract reflectance values into xarray.DataArray
        crs_4326 = proj.Proj(init='epsg:4326') # assuming you're using WGS84 geographic
        crs_proj = proj.Proj(str(PROJ['crs'].values))
        x, y = proj.transform(crs_4326, crs_proj, list(longitudes.values()), list(latitudes.values()))
        #x, y = proj.transform(crs_4326, crs_proj, 18.6101, 67.9123)


        #img = calc_cloudIndex(datetime)
        ax = plt.axes(projection=crs)
        ax.scatter(x, y, zorder=1, alpha= 1, c='b', s=15)
        ax.coastlines(color='grey')
        #ax.gridlines()
        ax.set_global()
        plt.imshow(local_scene['HRV'], transform=crs, extent=crs.bounds, origin='upper', cmap='gist_gray')
        cbar = plt.colorbar(pad=0.025, orientation="horizontal", fraction=0.046)
        #cbar.ax.tick_params(labelsize="small")
        cbar.set_label("Reflectance (%)")#,size="small")
        plt.show()
        #plt.savefig("sat_img.pdf", bbox_inches='tight')

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
