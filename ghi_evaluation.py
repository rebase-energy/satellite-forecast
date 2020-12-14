'''
Author: Dennis van der Meer
Email: dennis.vandermeer@angstrom.uu.se

This script validates instantaneous GHI derived from the satellite against
GHI data at 14 locations across Sweden, measured by SMHI.

The script relies on the cloud and ground reflectance matrices that are computed
for each available time instant to compute the clear-sky index and the clear-sky
GHI from pvlib.

'''
import pandas as pd
import numpy as np
import os
from glob import glob
import warnings
from sklearn.metrics import r2_score
from scipy import stats

def MBE(obs,sat):
    return(np.mean(sat-obs))

def rMBE(obs,sat):
    return(100*MBE(obs,sat)/np.mean(obs))

def MAE(obs,sat):
    return(np.mean(np.abs(sat-obs)))

def RMSE(obs,sat):
    return(np.sqrt(np.mean((sat-obs)**2)))

def rRMSE(obs,sat):
    return(100*RMSE(obs,sat)/np.mean(obs))

def SD(obs,sat):
    term1 = np.sum(len(obs)*(sat-obs)**2)
    term2 = (np.sum(sat-obs))**2
    return( (100/np.mean(obs)) * np.sqrt(term1 - term2) / len(obs) )

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

RESULTS_PATH = r"C:\Users\denva787\Documents\dennis\Greenlytics\Results"
SAT_GHI_PATH = r"C:\Users\denva787\Documents\dennis\Greenlytics\Results\satDerivedGHI"
DATA_PATH = r"C:\Users\denva787\Documents\dennis\Greenlytics\Data"
UNTARRED_DATA_PATH = r"D:\EUMETSAT\untarred"
REFLECTANCE_DATA_PATH = r"D:\EUMETSAT\reflectance"

#skies = ['all']#['clear','cloudy','all']
#elevations = [10]#[40, 5] # Reflect an zenith angle of 50 and 85 respectively
elevation = 10
lst = []
wndws = [5,10,15,20,25,30] # Training window lengths
wndw=5

for wndw in wndws:
    filename = "GHI_"+str(wndw)+".txt"
    results = pd.read_csv(os.path.join(RESULTS_PATH,filename), sep='\t', parse_dates=True, index_col=0)
    # remove NAs
    results.dropna(axis=0,how='any',inplace=True)
    # subset based on date (see V. Kallio-Meyers et al.) --> This doesn't lead to better results
    #results = results.loc['2019-05-01':'2019-08-31']

    for loc in latitudes:
        # Subset based on location
        df = results.filter(like=loc,axis=1)
        # Subset based on elevation:
        df = df.loc[(df['Elev_'+loc] >= elevation)]
        # Remove negative GHI measurements
        df = df.loc[(df['G_'+loc] >= 0)]
        # Error metrics
        mbe=np.around(MBE(df['G_'+loc].values,df['G_sat_'+loc].values),decimals=1)
        rmbe=np.around(rMBE(df['G_'+loc].values,df['G_sat_'+loc].values),decimals=1)
        mae=np.around(MAE(df['G_'+loc].values,df['G_sat_'+loc].values),decimals=1)
        rmse=np.around(RMSE(df['G_'+loc].values,df['G_sat_'+loc].values),decimals=1)
        rrmse=np.around(rRMSE(df['G_'+loc].values,df['G_sat_'+loc].values),decimals=1)
        y_mu=np.around(np.mean(df['G_'+loc].values),decimals=1)
        r2=np.around(100*r2_score(df['G_'+loc].values,df['G_sat_'+loc].values),decimals=1)
        sd=np.around(SD(df['G_'+loc].values,df['G_sat_'+loc].values),decimals=1)
        sizeN=np.around(df.shape[0],decimals=1)
        rho=np.around(stats.spearmanr(df['G_'+loc].values,df['G_sat_'+loc].values)[0],decimals=2)
        # Append the results
        lst.append([wndw,loc,mbe,rmbe,mae,rmse,rrmse,y_mu,r2,sd,sizeN,rho])
        # Just to make sure that we do not reuse results
        del(mbe,rmbe,mae,rmse,rrmse,y_mu,r2,sd,sizeN)

res = pd.DataFrame(lst, columns =['Window','Location','MBE','rMBE','MAE','RMSE','rRMSE','$\\bar{y}$','$R^2$','SD','$N$','$\\rho$'])
res = res.set_index('Location')
mbe_wide = res.pivot_table(index="Location",columns='Window',values='MBE')
mbe_wide = mbe_wide.loc[latitudes.keys()]
rmbe_wide = res.pivot_table(index="Location",columns='Window',values='rMBE')
rmbe_wide = rmbe_wide.loc[latitudes.keys()]
rmse_wide = res.pivot_table(index="Location",columns='Window',values='RMSE')
rmse_wide = rmse_wide.loc[latitudes.keys()]
rrmse_wide = res.pivot_table(index="Location",columns='Window',values='rRMSE')
rrmse_wide = rrmse_wide.loc[latitudes.keys()]
rho_wide = res.pivot_table(index="Location",columns='Window',values='$\\rho$')
rho_wide = rho_wide.loc[latitudes.keys()]

#print(mbe_wide.to_latex(index=True,longtable=False,sparsify=True,escape=False))
#print(rmbe_wide.to_latex(index=True,longtable=False,sparsify=True,escape=False))
#print(rmse_wide.to_latex(index=True,longtable=False,sparsify=True,escape=False))
#print(rrmse_wide.to_latex(index=True,longtable=False,sparsify=True,escape=False))
print(rho_wide.to_latex(index=True,longtable=False,sparsify=True,escape=False))
#res.to_csv(os.path.join(RESULTS_PATH,'test.txt'), header=True, sep='\t', index=False)



'''
Make a contour plot (do this in R instead)
import matplotlib.pyplot as plt
import scipy.stats as st

x=df['GHI_m_'+loc].values
y=df['GHI_sat_'+loc].values
xmin=ymin=0
xmax=ymax=1000

# Peform the kernel density estimate
xx, yy = np.mgrid[xmin:xmax:100j, xmin:xmax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = st.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)

fig = plt.figure()
ax = fig.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
# Contourf plot
cfset = ax.contourf(xx, yy, f, cmap='Blues')
## Or kernel density estimate plot instead of the contourf plot
#ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
# Contour plot
cset = ax.contour(xx, yy, f, colors='k')
# Label plot
#ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('Observed GHI')
ax.set_ylabel('Satellite derived GHI')

plt.show()
'''
