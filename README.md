# satellite-forecast
This project mainly focuses on estimating the global horizontal irradiance (GHI) from satellite images. Specifically, images recorded 
by the SEVIRI instrument aboard Meteosat Second Generation (MSG) satellite, which is operated by the European Organisation for the 
Exploitation of Meteorological Satellites (EUMETSAT).

The approach is based on the Heliosat method, in which it is assumed that the reflectance of a pixel is caused by a linear combination 
of ground reflectance and cloud reflectance, which results in the cloud-index. The cloud-index can be converted to the clear-sky index
using an approximate linear relation, from which the GHI can be computed using a clear-sky model (Ineichen-Perez implemented in 
[pvlib](https://pvlib-python.readthedocs.io/en/stable/) in this case).

The library also includes functions to generate GHI forecasts, as well as plotting functions. Raw images can be downloaded via EUMETSAT's 
[data portal](https://eoportal.eumetsat.int) and are read using [Satpy](https://satpy.readthedocs.io/en/stable/).

Relevant literature:
1. R. Perez, T. Cebecauer, and M. Su ́ri. Semi-empirical satellite models.
In J. Kleissl, editor, Solar Energy Forecasting and Resource Assessment,
chapter 2, pages 21–48. Academic Press, 2013.
2. J. Kühnert,E. Lorenz,and D. Heinemann. Satellite-based irradianceand power forecasting for the german energy market. 
In J. Kleissl, editor, Solar Energy Forecasting and Resource Assessment, chapter 11, pages
267–297. Academic Press, 2013.
3. V. Kallio-Myers, A. Riihelä, P. Lahtinen, A. Lindfors,
Global horizontal irradiance forecast for Finland based on geostationary weather satellite data,
Solar Energy,
Volume 198,
2020,
Pages 68-80,
ISSN 0038-092X.
4. S. Cros, J. Badosa, A. Szantaï, M. Haeffelin, Reliability Predictors for Solar Irradiance Satellite-Based Forecast,
Energies, Volume 13, 2020, ISSN 1996-1073.
