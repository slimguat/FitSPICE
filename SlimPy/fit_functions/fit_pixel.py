import numpy as np
from typing import Callable
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


from ..utils import clean_nans
def fit_pixel(*args,**kwargs):
    #this part will adapt to select the old and the new fast format in the future
    #for now it works with the old algorithm
    return(fit_pixel_multi(*args,**kwargs))

def fit_pixel_multi(x:np.ndarray,
                    y:np.ndarray,
                    ini_params:np.ndarray,
                    quentities,
                    fit_func:Callable,
                    bounds:np.ndarray=[np.nan],
                    plotit: bool=False,
                    weights: str = None,
                    verbose=0,
                    describe_verbose=False,
                    **kwargs
                    ):
    """Function that Fits a pixel data into a gaussian.

    Args:
        x(np.ndarray): spectrum axis. 
        pixel_data (np.ndarray): pixel intensity values as function of wavelength.
        Gauss_ini_params (np.ndarray, shape:4,optional): a list of initiale parameters if not initialized. 
        bounds (np.array,np.array): The bounds of the parameters
        plotit (bool, optional): In case you want to plot it. Defaults to False.
        weights (str, optional): string ["I": for a linear weight depend on the value of intensity]. Defaults to None.
    Return:
        coeff (np.ndarray): Fitted parameters.
        var_matrix (np.ndarray): Variation matrix that represent calculated fitting error.
    """
    assert len(ini_params.shape)==1
    assert x.shape==y.shape
    _s = ini_params.shape[0]
    if callable(bounds):
        bounds = bounds(par=ini_params,que=quentities)
        
    elif not (False in  (np.isnan(bounds))):
        bounds = np.zeros((2,_s))
        for i in range(_s):
            if quentities[i] == "B":
                bounds[:,i] = [-5,5]
            if quentities[i] == "I":
                bounds[:,i] = [ 0,1.e4]
            if quentities[i] == "x":
                bounds[:,i] = [ini_params[i]-2,ini_params[i]+2]
            if quentities[i] == "s":
                bounds[:,i] = [0.28,2]
        if verbose>=2: print(f"bounds were internally set:\n{bounds}")
                
        
    _s = ini_params.shape[0]
    _x,_y,w =  clean_nans(x,y,weights)

    if _y.shape[0]<=_s:
        if verbose>=0: print("after cleaning data the number of parameters is greater than data points")
        return (np.ones((_s  ))*np.nan,
                np.ones((_s,_s))*np.nan)
    try:
        res = curve_fit(fit_func,_x,_y,p0=ini_params,bounds=bounds,sigma=w)
    except RuntimeError:
        if verbose>=1: print("couldn't find the minimum")
        if verbose>=2: print(f"x     : {_x}\ny     : {_y}\ninipar: {ini_params}\nsigma : {w}\nbounds:\n{bounds}")
        
        res =  (np.ones((_s  ))*np.nan,
                np.ones((_s,_s))*np.nan)
        plotit =False
    except:
        if verbose>=0: print( "this value is not feasable")
        if verbose>=2: print(f"x     : {_x}\ny     : {_y}\nIniPar: {ini_params}\nsigma : {w}\nbounds:\n{bounds}")
        
        res =  (np.ones((_s  ))*np.nan,
                np.ones((_s,_s))*np.nan)
        plotit =False
        
    if plotit or verbose>=3 or verbose==-3:
        fig,axis = plt.subplots(1,1,figsize=(6,4))
        axis.step(_x,_y,where="mid")
        spectrum_title = "spectrum"
        if 'plot_title_prefix' in kwargs.keys():
            spectrum_title = spectrum_title + "\n" + kwargs['plot_title_prefix']
        axis.set_title(spectrum_title)
        if verbose>=2: print(f"fit_func:\n{fit_func}")
        
        axis.plot(_x,fit_func(_x,*ini_params),":",label="initial params")
        xlabel = "wavelength $(\AA)$\n"
        for i in range(len(quentities)):
            if quentities[i]=="I":
                value = f"I: {res[0][i]:06.1f}$\pm${np.sqrt(res[1][i,i]):04.2f}-"
            if quentities[i]=="B":
                value = f"B: {res[0][i]:03.1f}$\pm${np.sqrt(res[1][i,i]):04.2f}-"
            if quentities[i]=="x":
                value = f"x: {res[0][i]:07.2f}$\pm${np.sqrt(res[1][i,i]):04.2f}-"
            if quentities[i]=="s":
                value = f"s: {res[0][i]:03.1f}$\pm${np.sqrt(res[1][i,i]):04.2f}\n"
            xlabel+=value
        axis.plot(_x,fit_func(_x,*res[0]),
                  label="fitted params:\n".format()
                  )
        axis.set_xlabel(xlabel)        
        axis.legend(fontsize = 12)
        plt.tight_layout()
        plt.savefig(f"./tmp/{kwargs['plot_title_prefix']}.jpg")
        
        if verbose>=1: print("ini_params",ini_params,"\n","bounds",bounds)
        
    return res
