from multiprocessing import Pool, TimeoutError
import time,sys
import os,sys
import numpy as np
import glob,os,sys
import cv2
import matplotlib.pyplot as plt
import tifffile
from tqdm.notebook import tqdm
master_analysis_folder = r'C:\Users\bbintu\bbintu_jackfruit_scripts\XK_ChromatinTracing'
sys.path.append(master_analysis_folder)
import StandardChromatinAnalysis as sca





def f(ifov,iQs=None):
    if True:
        chr_ = sca.chromatin_analyzer(data_folders = [r'\\MERFISH6.ucsd.edu\merfish6v2\DNA_FISH\20230122_R120PVTS32RRNA'],
                              save_folder=r'\\MERFISH6.ucsd.edu\merfish6v2\DNA_FISH\20230122_R120PVTS32RDNA_Analysis',
                             H0folder=r'\\MERFISH6.ucsd.edu\merfish6v2\DNA_FISH\20230122_R120PVTS32RDNA\H0_DNA',extension=r'.dax')
        chr_.compute_flat_fields();
        nfovs = len(chr_.fls)
        if ifov<nfovs:
            chr_.shape=[64,2048,2048]
            chr_.set_fov(ifov)
            if iQs is None: iQs = np.arange(len(chr_.Qfolders))
            for iQ in iQs:
                chr_.set_hybe_RNA(iQ)
                fit_completed = chr_.check_finished_file()
                if not fit_completed:
                    try:
                        chr_.compute_cell_segmentation(nstart=7,nend=60,rescz=8,resc=4,sz_min_2d=150,use_gpu=False)
                        print("Loading background image")
                        chr_.load_background_image(rescz=1)
                        print("Loading signal image")
                        chr_.load_signal_image()
                        chr_.compute_drift()
                        chr_.fit_RNA_points_in_cells(th_keep=10,subtract_bk=True)
                        chr_.save_fits_RNA(icols=[0,1,2],plt_val=False)
                        chr_.re_th_RNA(icols=[0,1,2],th_s = [2.5,3,2.5],save='_v2')
                        chr_.save_fits_RNA(icols=[0,1,2])
                    except Exception as e:
                        print("Failed")
                        print(e)

    
    return ifov
if __name__ == '__main__':

    #Use as:
    #activate cellpose
    #cd C:\Users\bbintu\bbintu_jackfruit_scripts\XK_ChromatinTracing
    #python worker_DNA.py
    if len(sys.argv)>1:
        ifov,iQs = [int(sys.argv[1]),eval(sys.argv[2])]
        f(ifov,iQs=iQs)
    else:
        items = [ifov for ifov in range(125)]
        with Pool(processes=20) as pool:
            print('starting pool')
            while True:
                start = time.time()
                result = pool.map(f, items)
                end = time.time()
                print('Sleeping for a bit')
                print(result,end-start)
                time.sleep(3600)
    