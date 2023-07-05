from multiprocessing import Pool, TimeoutError
import time,sys
import os,sys
import numpy as np
import glob,os,sys
import cv2
import matplotlib.pyplot as plt
import tifffile
from tqdm import tqdm
master_analysis_folder = r'C:\Users\BintuLabUser\Scope3AnalysisScripts\ChromatinTracing2023'
sys.path.append(master_analysis_folder)
import StandardChromatinAnalysis as sca

#Run as:
#activate cellpose
#cd C:\Users\BintuLabUser\Scope3AnalysisScripts\ChromatinTracing2023
#python worker_DNA_Tula.py 25 [12]

def f(ifov_iQs,return_obj=False):
    ifov,iQs = ifov_iQs
    if True:
        chr_ = sca.chromatin_analyzer(data_folders = [r'Z:\TK_FFBBL1_SampleA_1_27_2023'],
                          save_folder=r'Z:\TK_FFBBL1_SampleA_1_27_2023_Analysis',
                         H0folder=r'Z:\TK_FFBBL1_SampleA_1_27_2023\H15R43_44_45',extension=r'\data')
        if return_obj:
            return chr_
        nfovs = len(chr_.fls)
        if ifov<nfovs:
            #chr_.shape=[64,2048,2048]
            chr_.set_fov(ifov)
            chr_.im_meds = np.ones([chr_.ncol]+list(chr_.shape[1:]))
            if iQs is None: iQs = np.arange(len(chr_.Rfolders))
            for iQ in iQs:
                chr_.set_hybe(iQ)
                fit_completed = chr_.check_finished_file()
                if not fit_completed:
                    try:
                        
                        chr_.compute_cell_segmentation(nstart=20,nend=chr_.shape[0],rescz=800,resc=6,th_prob=0,th_flow=0,
                                                       sz_min_2d=150,use_gpu=False,force=False,expand=11)
                        plt.close('all')
                        #chr_.compute_cell_segmentation(nstart=7,nend=60,rescz=8,resc=4,sz_min_2d=150,use_gpu=False)
                        #sca.expand_segmentation(chr_,size=11,resc=5)
                        print("Loading background image")
                        chr_.load_background_image()
                        print("Loading signal image")
                        chr_.load_signal_image()
                        chr_.compute_drift()
                        chr_.fit_points_in_cells(nmax=300,subtract=False,icols=[0,1,2])
                        chr_.save_fits(plt_val=True)
                    except Exception as e:
                        print("Failed")
                        print(e)
    
    return ifov
if __name__ == '__main__':

    #Use as:
    #activate cellpose
    #cd C:\Users\bbintu\bbintu_jackfruit_scripts\XK_ChromatinTracing
    #python worker_DNA.py
    
    #for parallel computing decine number of pools
    processes = 4
    
    if len(sys.argv)>=3:
        ### test on a single fov and hybe
        ifov,iQs = [int(sys.argv[1]),eval(sys.argv[2])]
        f([ifov,iQs])
    elif len(sys.argv)==2:
        ### given a single fov run across all hybes
        ifov = int(sys.argv[1])
        chr_ = f([0,None],return_obj=True)
        nQs = len(chr_.Rfolders)
        print("Detected number of hybes:",nQs)
        chr_.set_fov(ifov)
        completed = []
        for iQ in np.arange(nQs):
            chr_.set_hybe(iQ)
            fit_completed = chr_.check_finished_file()
            completed.append(fit_completed)
        print("Fraction completed:",np.mean(completed))
        items = [(ifov,np.arange(nQs)[iP::processes]) for iP in range(processes)]
        
        with Pool(processes=processes) as pool:
            print('starting pool')
            #while True:
            start = time.time()
            result = pool.map(f, items)
            end = time.time()
            print('Sleeping for a bit')
            print(result,end-start)
    else:
        ### run across all data
        chr_ = f([0,None],return_obj=True)
        nfovs = len(chr_.fls)
        items = [(ifov,None) for ifov in range(nfovs)]
        with Pool(processes=processes) as pool:
            print('starting pool')
            #while True:
            start = time.time()
            result = pool.map(f, items)
            end = time.time()
            print('Sleeping for a bit')
            print(result,end-start)
            #time.sleep(600)
    