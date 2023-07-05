from multiprocessing import Pool, TimeoutError
import time,sys
import os,sys
master_analysis_folder = r'C:\Users\BintuLab\Dropbox\MERFISH_DC_SCOPE3'
sys.path.append(master_analysis_folder)
from ioMicro import *
def f(set_ifov):
    if True:
        if True:
            set_,ifov=set_ifov#'set1'

            ### Decoding
            dec = decoder(analysis_folder = r'\\132.239.200.33\Raw_data\DCBB_MER250__12_2_2022_Analysis')
            dec.get_set_ifov(ifov=ifov,set_=set_,keepH = [1,2,3,4,5,6,7,8],ncols=3)
            if not dec.is_complete:
                ### Correct distortion
                drift_  = drift_refiner(data_folder=r'\\132.239.200.33\Raw_data\DCBB_MER250__12_2_2022',
                                 analysis_folder = r'\\132.239.200.33\Raw_data\DCBB_MER250__12_2_2022_Analysis')
                drift_.get_fov(ifov,set_)
                for iR in np.arange(len(drift_.raw_fls)):
                    analysis_folder_ = drift_.analysis_folder+os.sep+'distortion'
                    if not os.path.exists(analysis_folder_):os.makedirs(analysis_folder_)
                    fl_save = analysis_folder_+os.sep+os.path.basename(drift_.raw_fls[0]).split('.')[0]+'--'+drift_.set_+'--iR'+str(iR)+'.npy'
                    if not os.path.exists(fl_save):
                        drift_.load_images(iR)
                        drift_.normalize_ims(zm=30,zM=50)
                        drift_.get_Tmed(sz_=300,th_cor=0.6,nkeep=9)
                        try:
                            P1_,P2_ = drift_.get_P1_P2_plus();
                            P1__,P2__ = drift_.get_P1_P2_minus();
                            P1f,P2f = np.concatenate([P1_,P1__]),np.concatenate([P2_,P2__])
                        except:
                            P1f,P2f = [],[]

                        if False:
                            import napari
                            viewer = napari.view_image(drift_.im2n,name='im2',colormap='green')
                            viewer.add_image(drift_.im1n,name='im1',colormap='red')
                            viewer.add_points(P2_,face_color='g',size=10)
                            viewer.add_points(P1_,face_color='r',size=10) 
                            drift_.check_transf(P1f,P2f)
                        
                            print("Error:",np.percentile(np.abs((P1f-P2f)-np.median(P1f-P2f,axis=0)),75,axis=0))
                            P1fT = drift_.get_Xwarp(P1f,P1f,P2f-P1f,nneigh=50,sgaus=20)
                            print("Error:",np.percentile(np.abs(P1fT-P2f),75,axis=0))
                        
                        print(fl_save)
                        np.save(fl_save,np.array([P1f,P2f]))


                print("Loading the fitted molecules")
                dec.get_XH()
                print("Correcting for distortion acrossbits")
                dec.apply_distortion_correction()
                dec.load_library(lib_fl=master_analysis_folder+os.sep+'codebook_DCBB250.csv')
                dec.XH = dec.XH[dec.XH[:,-4]>0.25]
                dec.get_inters(dinstance_th=3)
                dec.pick_best_brightness(nUR_cutoff = 3,resample = 10000)
                dec.pick_best_score(nUR_cutoff = 3)
                np.savez_compressed(dec.save_file_dec,res_pruned=dec.res_pruned,icodes=dec.icodes,scores_pruned = dec.scores_pruned,Xcms=dec.Xcms)
                dec.load_segmentation()
                dec.cts_all_pm = dec.get_counts_per_cell(nbad=0)
                dec.cts_all = dec.get_counts_per_cell(nbad=1)
                np.savez(dec.save_file_cts,cts_all_pm = dec.cts_all_pm,cts_all = dec.cts_all,gns_names=dec.gns_names,cm_cells=dec.cm_cells)
            else:
                print("Is complete")
        #except:
        #    print("Failed")
    
    return set_ifov
if __name__ == '__main__':
    # start 4 worker processes
    items = [(set_,ifov)for set_ in ['set1','set2','set3','set4']
                        for ifov in range(400)]
    item = [str(sys.argv[1]),int(sys.argv[2])]
    f(item)
    #print(result,end-start)