import math
import numpy as np
from numba import jit
from multiprocess.shared_memory import SharedMemory 
import matplotlib.pyplot as plt
import os

def prepare_filenames(prefix=None, 
                      plot_filename=None, 
                      data_save_dir="./.p/",
                      plot_save_dir="./imgs/", 
                      i=None,
                      verbose=0):
    if type(prefix)==str:
        filename = prefix+"_window_{:03d}_"+"{:}.p"
    elif prefix==None:
        dir = data_save_dir
        if not os.path.isdir(dir):
            os.mkdir(dir)
        dir_list = os.listdir(dir); j=0
        for file in dir_list:
            try:
                j2 = int(file[0:3])
                if j2>=j:
                    j=j2+1
                        
            except Exception:
                pass
        j3 = j
        dir2 = dir
    if type(plot_filename)==str:
        if plot_filename.format(" ",0,0) == plot_filename: #make sure this passed variable is subscriptable 
            filename_a = plot_filename+"plot_{:03d}_{}_{}.jpg"
            filename_b = plot_filename+"hist_{:03d}_{}_{}.jpg"
            
    elif prefix==None:
        dir = plot_save_dir
        if not os.path.isdir(dir):
            os.mkdir(dir)
        dir_list = os.listdir(dir); j=0
        for file in dir_list:
            try:
                j2 = int(file[0:3])
                    
                if j2>=j:
                    j=j2+1
                        
                    
            except Exception:
                pass
        j = max(j3,j)
            #Delete these later------
        j=(i if type(i)!=type(None) else j)
        if verbose>=1:print("working in the file with prefix i={:03d} ".format(j))
            #------------------------
        filename_a = dir + "{:03d}_".format(j)+"plot_{:03d}_"+"{}_{}.jpg"
        filename_b = dir + "{:03d}_".format(j)+"hits_{:03d}_"+"{}_{}.jpg"
        filename = dir2+"{:03d}_".format(j)+"window_{:03d}_"+"{}_{}.p"
    return filename,filename_a,filename_b


def clean_nans(xdata:np.ndarray,
               ydata:np.ndarray,
               weights=None,
              )-> np.ndarray[np.ndarray,np.ndarray]:
    """
    Function that returns a cleaned version of x and y arrays from "np.nan" values.
    
    Args:
        xdata   (np.ndarray): x data.
        ydata   (np.ndarray): y data.
        weights (np.ndarray): weights of y data.
    Return:
        xdata_cleaned (np.ndarray): cleaned x data
        ydata_cleaned (np.ndarray): cleaned y data
        wdata_cleaned (np.ndarray): cleaned weights
    """
    assert xdata.shape==ydata.shape
    num_elements = np.zeros(xdata.shape)
    num_elements = np.logical_not((np.isnan(xdata)) | (np.isinf(xdata)) | (np.isinf(ydata)) | (np.isnan(ydata)) | (ydata<0))
    clean_x = xdata[num_elements]; clean_y = ydata[num_elements]
    if type(weights) not in [str,type(None)]:
        weights = np.array(weights)
        assert xdata.shape==weights.shape
        sigma = np.sum(weights[num_elements])/weights[num_elements]
        if sigma[np.where(clean_y == np.max(clean_y))] < sigma[np.where(clean_x == np.min(clean_x))]:
            print("We found that the weights injected aren't decreasing with Intensity\n if you want to continue supress this message by deleting it from:\n SlimPy.clean_nans")
    elif  type(weights) == str:
        if weights == "1/sqrtI":
            weights = 1./np.sqrt(clean_y.copy())
        elif weights == "I":
            weights = clean_y.copy()
        elif weights == "expI":
            weights = clean_y.copy()**2
        elif weights == "I2":
            weights = np.exp(clean_y.copy())
        elif weights == "sqrtI":
            weights = np.sqrt(clean_y.copy())
        else: 
            raise ValueError ("the weights are unknown make sure you give the right ones\n current value: {} {} \n the allowed ones are: I, expI, I2, sqrtI".format(type(weights),weights))

        try:
            weights2=weights - np.nanmin(weights)
            weights = weights2
        except:
            pass
        sigma = 1/(weights.copy()/np.sum(weights))
    elif type(weights)== type(None):
        sigma = 1/(np.ones(len(clean_y))/len(clean_y))
    return clean_x,clean_y,sigma  

@jit(nopython=True)
def fst_neigbors(
        extent: float
    ):
    """Generates a list of first neiboors in a square lattice and returns inside the list
        [n,m,n**2+m**2]

    Args:
        extent (float): how far the pixels will extend
    Return:
        nm_list (np.ndarray): list of data [n,m,n**2+m**2]
    """
    nm_list = np.array([
        [0,0,0] 
               ],dtype=np.int64)
    N= 0
    while True:
        N+=1
        Rmin = N
        Rmax = N+1
        if extent!=0:
            for n in range(int(np.sqrt(Rmax))+1):
                min_m = int(np.sqrt(Rmin-n**2) if Rmin-n**2>0 else 0 )
                max_m = int(np.sqrt(Rmax-n**2)+1)
                for m in range(min_m,max_m):    
                    s = n**2 + m**2
                    if s==Rmin:
                        number_element = 1 
                        if n!=0 and m!=0:
                            number_element = 4 
                        elif n!=0 or m!=0: 
                            number_element = 2
                        
                        _list = np.zeros((nm_list.shape[0]+number_element,nm_list.shape[1]),
                                        dtype=np.int64)
                        _list[:-number_element] = nm_list
                        _list[ -number_element] = n,m,s
                        
                        if n!=0 and m!=0:
                            _list[ -3] = -n, m,s
                            _list[ -1] = -n,-m,s
                            _list[ -2] =  n,-m,s
                            
                        elif n!=0 or m!=0: 
                            if n==0:
                                _list[ -1] =  n,-m,s
                            else :
                                _list[ -1] =  -n,m,s
                        
                        nm_list=_list
                        
                        
                    
        if nm_list[-1,2] == extent**2:
            # print(set(nm_list[:,2])) 
            # for i in nm_list:
            #     print(i)
            return(nm_list)

@jit(nopython=True)
def join_px(data,i,j,ijc_list):
    res_px = float(0.)
    s = float(0.)
    
    for n_layer in ijc_list:
        i2,j2,c = n_layer 
        
        if (data.shape[0] - (i+i2) > 0
            and 
            data.shape[1] - (j+j2) > 0
            and
            i+i2 >= 0 
            and
            j+j2 >= 0
            ):
            if not np.isnan(data[i+i2,j+j2]):
                res_px += float(c*data[i+i2,j+j2])
                s += float(c)
    if s!=0:
        return (res_px/ s)
    else: return np.nan

@jit(nopython=True)   
def join_dt(data,ijc_list):
    # data_new = np.zeros_like(data,dtype=float) #numba has no zeros_like 
    data_new = data.copy()*np.nan
    for k in range(data.shape[0]):
        for l in range(data.shape[1]):
            for i in range(data.shape[2]):
                for j in range(data.shape[3]):
                    data_new[k,l,i,j] = join_px(data[k,l],i,j,ijc_list)
    return data_new

# @jit(nopython=True) #not tryed yet
def Preclean(cube):
    cube2 = cube.copy()
    # logic=np.logical_or(np.isinf(cube2),cube2<-10**10)
    logic=np.logical_or(cube2>490,cube2<-10)
    cube2[logic]=np.nan
    if False: #this part is for elemenating the cosmic effected values but it's not well done (see it again)
        mean_cube = np.nanmean(cube2, axis=1)*1000
        for i in range(cube2.shape[1]):
            cube2[:,i,:,:][cube2[:,i,:,:]>mean_cube] = np.nan
        
    return cube2

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

def gen_shmm(create = False,name=None,ndarray=None,size=0,shape=None,dtype=float):
    assert (type(ndarray)!=type(None) or size!=0) or type(name)!=type(None)
    assert type(ndarray)!=type(None) or type(shape)!=type(None)
    size = size if type(ndarray) == type(None) else ndarray.nbytes
    shmm = SharedMemory(create = create,size=size,name=name)
    shmm_data = np.ndarray(shape = shape if type(ndarray)==type(None) else ndarray.shape  
                           , buffer = shmm.buf , dtype=dtype)
    
    if create and type(ndarray)!=type(None):
        shmm_data[:] = ndarray[:]
    elif create:
        shmm_data[:] = np.nan
        
    return shmm,shmm_data

def verbose_description(verbose):
    print(f"level {verbose:01d} verbosity")
    if verbose ==-2:
        print("On-screen information mode: Dead \nNo information including warnings  (CAREFUL DUDE!!)")
    elif verbose ==-1:
        print("On-screen information mode: Minimal\nHighly important ones only and wornings (Don't have a blind faith please)")
        
    elif verbose ==0:
        print("On-screen information mode: Normie\nBasic information any normie needs")
    
    elif verbose ==1:
        print("On-screen information mode: Extra\nmore detailed information for tracking and debugging")
    elif verbose ==2:
        print("On-screen information mode: Stupid\nUnless you are as stupid as the writer of this script, you don't need this much information for debugging an error")
    elif verbose ==3:
        print("On-screen information mode: Visual\nPlot extra figures in a ./tmp file with ")
    
def gen_velocity(doppler_data,quite_sun=[60,150,550,600],correction=False,verbose=0,get_0lbdd = False):
    qs = quite_sun
    mean_doppler = np.nanmedian(doppler_data[qs[2]:qs[3],qs[0]:qs[1]]) 
    results = (doppler_data-mean_doppler) / mean_doppler * 3*10**5
    if correction:
        if verbose >0: print("Correcting")
        hist,bins= gen_velocity_hist(results,bins=np.arange(-600,600,1),verbose=verbose)
        vel_corr,ref = correct_velocity(hist,bins,verbose=verbose)
        if verbose >0: print(f"The correction found the distribution was off by {ref}")
        results -= ref
        
    if verbose > 1:
        fig=plt.figure()
        plt.pcolormesh(results,cmap="twilight_shifted",vmax=80,vmin=-80)
        plt.plot([qs[1],qs[0],qs[0],qs[1],qs[1]],
                    [qs[2],qs[2],qs[3],qs[3],qs[2]],color="green",label="mean value {:06.1f}".format(mean_doppler))
        plt.legend()
        # plt.savefig('fig_test.jpg')
    return results, (None if not correction else ref),(None if not get_0lbdd else mean_doppler) 

def gen_velocity_hist(velocity_data,axis=None,bins = None,verbose=0):
    hist,bins = np.histogram(velocity_data,bins = bins) 
    bins = (bins[:-1] + bins[1:])/2
    if verbose>1:
        if type(axis) == type(None): 
            fig,axis = plt.subplots(1,1)        
        axis.step(bins,hist)
        axis.set_yscale('log',base=10)
        plt.axvline(0,ls="--",color='red',alpha=0.5)
    return hist, bins
    
def correct_velocity(velocity_hist,velocity_values,verbose=0):
    if verbose>1: print("correct_velocity<func>.velocity_hist.shape: {}\n,correct_velocity<func>.velocity_values.shape: {}".format(velocity_hist.shape,velocity_values.shape))
    ref_velocity = velocity_values[np.where(velocity_hist == np.nanmax(velocity_hist))[0]]
    ref_velocity = np.mean(ref_velocity)
    if verbose>0: print(f"the velocity reference was found at {ref_velocity}\n now it will be set to 0")
    velocity_values_corr = velocity_values-ref_velocity
    return velocity_values_corr,ref_velocity