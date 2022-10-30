import warnings
from astropy.wcs import FITSFixedWarning
import spice_utils.ias_spice_utils.utils as spu, numpy as np, scipy as sc, pandas as pd, matplotlib.pyplot as plt
from sunraster.instr.spice import read_spice_l2_fits
from os import getenv
from astropy.visualization import SqrtStretch,PowerStretch, AsymmetricPercentileInterval, ImageNormalize, MinMaxInterval
import astropy
from sys import modules
from SlimPy import * 
import pickle
import csv
import seaborn as sns
warnings.filterwarnings('ignore' )
plt.ion()
Npool = 40



file1 = "solo_L2_spice-n-ras_20220403T015832_V06_100664015-000.fits"
if True:
    stp = 196
    std_name="SCI_DYN-SLOW_SC_SL04_5.0S_FF"
    spu.write_studies_files_list_for_stp('{:03d}'.format(stp),"/archive/SOLAR-ORBITER/SPICE/","./")
    files = spu.read_studies_files_list_for_stp('./STP{:03d}/files_list.txt'.format(stp)) [std_name]   
    data_path = getenv('ARCHIVE_DATA_PATH', default='/archive/SOLAR-ORBITER/SPICE')

file_path1 = spu.full_file_path(file1, data_path)
raster1 = read_spice_l2_fits(file_path1)


window_size = np.array([[500,550],[80,130]])
# window_size = np.array([[110,720],[0,-1]])
quite_sun = np.array([0,-1,0,-1])

init_params = [
    np.array(
    [0.08,702.5,0.5,
     0.10,703.5,0.5,
     0.04,705.5,0.5,
     0.01],
    ),
    np.array(
    [0.03,749.5,0.5,
     0.01]),
    np.array(
    [0.5 ,770.0,0.5,
     0.05,772.0,0.5,
     0.01]
    ),
    np.array(
    [0.30,780  ,0.5,
     0.05,782  ,0.5,
     0.01]),
    np.array(
    [0.20,786  ,0.5,
     0.30,787.5,0.5,
     0.01]),
    # np.array(
    # [0.15,988.5,0.5,
    #  0.20,989.5,0.5,
    #  0.35,991.3,0.5,
    #  0.5]),    
]
segmentation = [
    np.array([ 700  , 707. ]),
    np.array([ 749  , 751  ]),
    np.array([ 767  , 774  ]),
    np.array([ 777  , 783  ]),
    np.array([ 784  , 790  ]),
    # np.array([ 987  , 993  ])
]
segmentation = [
    np.array([ 0  , 10**4 ]),
    np.array([ 0  , 10**4 ]),
    np.array([ 0  , 10**4 ]),
    np.array([ 0  , 10**4 ]),
    np.array([ 0  , 10**4 ]),
]
quentities = [
    ["I","x","s","I","x","s","I","x","s","B"],
    ["I","x","s","B"],
    ["I","x","s","I","x","s","B"],
    ["I","x","s","I","x","s","B"],
    ["I","x","s","I","x","s","B"],
]
convolution_threshold = [
    np.array([0.1,10**-4,0.1,
              0.1,10**-4,0.1,
              0.1,10**-4,0.1,
              100]),
    np.array([0.1,10**-4,0.1,
              100]),
    np.array([0.1,10**-4,0.1,
              0.1,10**-4,0.1,
              100]),
    np.array([0.1,10**-4,0.1,
              0.1,10**-4,0.1,
              100]),
    np.array([0.1,10**-4,0.1,
              0.1,10**-4,0.1,
              100]),
]
convolution_extent_list = np.array([0,1,2])

from SlimPy.fit_models import flat_inArg_multiGauss
from SlimPy.fit_functions import _fit_raster
res = _fit_raster(
    path_or_raster = file_path1,
    init_params = init_params,
    fit_func = flat_inArg_multiGauss,
    convolution_threshold=convolution_threshold,
    quentities = quentities,
    window_size = window_size,
    convolution_extent_list=convolution_extent_list,
    Jobs= {"windows":6,"pixels":8},
    verbose=-1
    
)