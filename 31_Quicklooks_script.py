from SlimPy.fit_functions import _fit_raster
from SlimPy.fit_models    import flat_inArg_multiGauss
from SlimPy.utils import getfiles
from SlimPy.init_handler import gen_fit_inits
from time import sleep
import os

import numpy as np
if False: #Get all the files by selection
    Compo_files = []
    L2_folder = "/archive/SOLAR-ORBITER/SPICE/fits/level2/"
    YEAR      = 2022
    MONTH     = 10
    DAY       = "All" 
    STD_TYP  = "DYN" #"DYN" "ALL"
    selected_fits = getfiles(L2_folder,YEAR,MONTH,DAY,STD_TYP,verbose = 0)

if False: #Get all the files by selection
    Compo_files = []
    L2_folder = "/archive/SOLAR-ORBITER/SPICE/fits/level2/"
    YEAR      = 2022
    MONTH     = 10
    DAY       = [i for i in range(17,32)]#"All" 
    STD_TYP  = "DYN" #"DYN" "ALL"   
    selected_fits = []
    for DAY in range(17,32):
        try :
            files =getfiles(L2_folder,YEAR,MONTH,DAY,STD_TYP,STP_NUM=228,MISOSTUD_NUM = 2155,verbose = 0)
            
            selected_fits.append(files[0])
            if len(files)!=1:selected_fits.append(files[-1])
            
        except:pass


from  pathlib import Path, PosixPath
from astropy.io import fits as fits_reader
from SlimPy.utils import quickview
import os
from SlimPy.init_handler import getfiles
from datetime import datetime

if False: #Get all the files by selection
    Compo_files = []
    L2_folder = "/archive/SOLAR-ORBITER/SPICE/fits/level2/"
    YEAR      = 2022
    MONTH     = 10
    DAY       = [i for i in range(17,32)]#"All" 
    STD_TYP  = "DYN" #"DYN" "ALL"   
    selected_fits = []
    for DAY in range(17,32):
        try :
            selected_fits.append(getfiles(L2_folder,YEAR,MONTH,DAY,STD_TYP,STP_NUM=228,MISOSTUD_NUM = 2155,verbose = 0)[0])
        except:pass
if True: #Get all the files by selection
    Compo_files = []
    L2_folder = "/archive/SOLAR-ORBITER/SPICE/fits/level2/"
    YEAR      = 2022
    MONTH     = 10
    DAY       = [26]#"All" 
    STD_TYP  = "DYN" #"DYN" "ALL"   
    selected_fits = []
    try :
        selected_fits = (getfiles(L2_folder,YEAR,MONTH,DAY,STD_TYP,STP_NUM=228,MISOSTUD_NUM = 2155,verbose = 0))
    except:pass
    
if False: #Get all the files by selection
    Compo_files = []
    L2_folder = "/archive/SOLAR-ORBITER/SPICE/fits/level2/"
    YEAR      = 2022
    MONTH     = 10
    DAY       = [i for i in range(16,32)]#"All" 
    STD_TYP  = "DYN" #"DYN" "ALL"       j
    selected_fits = getfiles(L2_folder,YEAR,MONTH,DAY,STD_TYP,STP_NUM=228,MISOSTUD_NUM = 2155,verbose = 0)

    HRI_timeformat= '%y-%m-%d %H:%M:%S'
    target_times_str = [
    "2022-10-17 07:00:00" , "2022-10-17 07:30:00",
    "2022-10-18 14:00:00" , "2022-10-18 14:30:00",
    "2022-10-19 19:00:00" , "2022-10-19 20:00:00",
    "2022-10-20 19:00:00" , "2022-10-20 20:00:00",
    "2022-10-21 19:00:00" , "2022-10-21 20:00:00",
    "2022-10-22 19:00:00" , "2022-10-22 19:30:00",
    "2022-10-23 19:00:00" , "2022-10-23 19:30:00",
    "2022-10-24 19:00:00" , "2022-10-24 19:30:00",
    "2022-10-25 19:00:00" , "2022-10-25 19:30:00",
    "2022-10-26 19:00:00" , "2022-10-26 19:30:00",
    ]
    target_times = [datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S') for datetime_str in target_times_str]
    times_str = []
    for i ,fits in enumerate(selected_fits) :
        raster = fits_reader.open(str(fits))
        times_str.append(raster[0].header["DATE-OBS"])
    SPICE_timeformat = "%Y-%m-%dT%H:%M:%S.%f"
    times = [datetime.strptime(datetime_str, SPICE_timeformat) for datetime_str in times_str]
    selected_fits2 = []
    for i in range(len(target_times)):
        delta = np.array([np.abs( (time-target_times[i]).total_seconds() ) for time in (times)])
        index = np.nanargmin(delta)
        if selected_fits[index] not in selected_fits2: selected_fits2.append(selected_fits[index])
    selected_fits = selected_fits2

for fits in selected_fits:
    print(fits)
sleep(10)
    
bad_fits = []
    
for fits_file in selected_fits:
    try:
        fit_args = gen_fit_inits(fits_file,verbose=4)
    except:
        bad_fits.append(fits_file) 
    _fit_raster(                                                                 
            path_or_raster          = str(fits_file)                       ,                               
            init_params             = fit_args["init_params"]              ,                                
            fit_func                = flat_inArg_multiGauss                ,                               
            quentities              = fit_args["quentities"]               ,                                 
            bounds                  = np.array([np.nan])                   ,                                    
            window_size             = np.array([[110,720],[0,-1]])         ,                                       
            convolution_function    = lambda lst:np.zeros_like(lst[:,2])+1 ,                                   
            convolution_threshold   = fit_args['convolution_threshold']    ,                                   
            convolution_extent_list = np.array([0,1,2])                    ,                                   
            weights                 = None                                 ,                                  
            counter_percent         = 10                                   ,                                   
            preclean                = True                                 ,                                        
            preadjust               = False                                ,                                     
            save_data               = True                                 ,                                            
            save_plot               = True                                 ,                                            
            prefix                  = None                                 ,                                         
            plot_filename           = fits_file.stem                       ,                  
            data_filename           = fits_file.stem                       ,                  
            quite_sun               = np.array([0,-1,110,720])             ,              
            plot_save_dir           = "./data_storage/pipeline_v-00.07_2022-11-26",                  
            data_save_dir           = "./data_storage/pipeline_v-00.07_2022-11-26",                  
            plot_kwargs             = {}                                   ,                
            show_ini_infos          = True                                 ,                   
            forced_order            = None                                 ,                 
            Jobs                    = {"windows":1,"pixels":96}            ,         
            verbose                 = -2                                   ,            
            describe_verbose        = True                                 , 
        )

print(bad_fits)   
        
   
   
   
   
   
   
   
   
   
