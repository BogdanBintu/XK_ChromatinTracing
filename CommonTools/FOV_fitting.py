#Bogdan Bintu 12/2018
#This is intended to deal with OB FOVs and use pytorch for fast fitting.
import os,sys,glob
import numpy as np
import matplotlib.pylab as plt

import cv2


import IOTools_py3 as io
import MosaicTools as mt
import AlignmentTools_py3 as at

from scipy.ndimage import zoom
def med_correct(im3d,ksize = 32):
    """Divide by the median of 2d slices"""
    sz,sxo,syo = im3d.shape
    num_windows_x = int(sxo/ksize)
    num_windows_y = int(syo/ksize)
    sx = num_windows_x*ksize
    sy = num_windows_y*ksize
    im = im3d[:,:sx,:sy]

    im_reshape = im.reshape([sz,num_windows_x,ksize,num_windows_y,ksize])
    im_reshape = np.swapaxes(im_reshape,2,3)
    im_reshape = im_reshape.reshape(list(im_reshape.shape[:-2])+[ksize*ksize])
    im_med = np.median(im_reshape,axis=-1)
    sz,sx_,sy_ = im_med.shape
    
    im_medf = zoom(im_med,[1,float(sxo)/sx_,float(syo)/sy_],order=1)
    return im3d/imf
from scipy.ndimage import zoom
def med_norm3d(im3d,ksize = 25):
    """Divide by the interpolated median of boxes of size ~ksize"""
    im3d_ = np.array(im3d).astype(np.float32)
    szo,sxo,syo = im3d_.shape
    num_windows_z = int(np.round(szo/float(ksize)))
    num_windows_x = int(np.round(sxo/float(ksize)))
    num_windows_y = int(np.round(syo/float(ksize)))
    ksizex = max(int(sxo/float(num_windows_x)),1)
    ksizey = max(int(syo/float(num_windows_y)),1)
    ksizez = max(int(szo/float(num_windows_z)),1)
    sz = num_windows_z*ksizez
    sx = num_windows_x*ksizex
    sy = num_windows_y*ksizey
    im = im3d_[:sz,:sx,:sy]
    im_reshape = im.reshape([num_windows_z,ksizez,num_windows_x,ksizex,num_windows_y,ksizey])
    im_reshape = im_reshape.swapaxes(0,1).swapaxes(1,3).swapaxes(-1,2)
    im_reshape = im_reshape.reshape([np.prod(im_reshape.shape[:3])]+list(im_reshape.shape[-3:]))
    im_med = np.median(im_reshape,0).swapaxes(1,2)
    sz_,sx_,sy_ = im_med.shape
    im_medf = zoom(im_med,[float(szo)/sz_,float(sxo)/sx_,float(syo)/sy_],order=1)
    return im3d_/im_medf
from scipy.ndimage import maximum_filter
def find_local_max(im3d_,size=3,threshold=1.5):
    """Find the local maximum within the image for a size"""
    mf = maximum_filter(im3d_,size)
    max_id = (mf==im3d_)&(im3d_>threshold)
    z,x,y = np.where(max_id)
    h = im3d_[z,x,y]
    return np.array([z,x,y,h],dtype=np.float32).T
def pack(im,psize=256,return_starts=False):
    im_packs = []
    dims = im.shape
    num_packs = [int(np.ceil(float(sz)/psize)) for sz in dims]
    it_packs = np.indices(num_packs).reshape(len(num_packs),-1).T
    for starts in it_packs:
        idx = [slice(i*psize,(i+1)*psize) for i in starts]
        im_packs.append(im[tuple(idx)])
    if return_starts:
        return im_packs,np.array(it_packs)*psize
    return im_packs
class fov_decode:
    def __init__(self,hybe_flds,ifov,analysis_folder=None,num_cols=3,bead_col = 1,verbose=True,mode3d='sequential'):
        self.mode3d = mode3d
        self.bead_col=bead_col
        self.hybe_flds = hybe_flds
        self.verbose = verbose
        self.data_folder = os.path.dirname(self.hybe_flds[0])
        self.analysis_folder = analysis_folder
        if self.analysis_folder is None:
            self.analysis_folder = self.data_folder+os.sep+'Analysis'
        if not os.path.exists(self.analysis_folder):
            os.makedirs(self.analysis_folder)
        self.num_cols = num_cols
        self.fovs = list(map(os.path.basename,glob.glob(self.hybe_flds[0]+os.sep+'*.dax')))
        self.ifov = ifov
        self.fov = self.fovs[self.ifov]
        self.tags = list(map(os.path.basename,self.hybe_flds))
        self.Rs = [[int(r) for r in tag.split('R')[-1].split(';')[0].split('T')[0].split(',')][::-1] 
                   for tag in self.tags]
        
        self.construct_daxs()
    def get_corrections(self,num_ims=None):
        cor_file = self.analysis_folder+os.sep+'im_means.npy'
        if os.path.exists(cor_file):
            self.im_means = np.load(cor_file)
            return self.im_means
        fldr = self.hybe_flds[0]
        daxs = [fldr+os.sep+fov for fov in self.fovs]
        im_means=[]
        for icol in range(self.num_cols):
            ims = []
            print('Flattening color ',icol)
            for dax_fl in daxs[:num_ims]:
                dax_obj = io.dax_im(dax_fl,num_col=self.num_cols,color=icol,mode3d=self.mode3d)
                im = dax_obj.get_mid(icol,tag='mid')
                ims.append(im)
            im_means.append(np.mean(ims,0))
        self.im_means = np.array(im_means,dtype=np.uint16)
        np.save(cor_file,self.im_means)
    def construct_daxs(self):
        self.dax_objs = {}
        for ifol,folder in enumerate(self.hybe_flds):
            dax_fl = folder+os.sep+self.fov
            colors = list(range(self.num_cols))[::-1]
            colors.pop(self.bead_col)
            for icol,col in enumerate(colors):
                dax_obj = io.dax_im(dax_fl,num_col=self.num_cols,
                                    bead_col=self.bead_col,color=col,mode3d=self.mode3d)
                self.dax_objs[self.Rs[ifol][icol]] = dax_obj
    def get_ims(self):
        if self.verbose:
            print('Dealing with flat field...')
        self.get_corrections()
        for key in self.dax_objs:
            if self.verbose:
                print('Loading image R',key)
            dax_obj = self.dax_objs[key]
            im = dax_obj.get_im()
            im = im.astype(np.float32)/self.im_means[dax_obj.color]
            self.dax_objs[key].im = im/np.mean(im)
            im_dapi = dax_obj.get_mid(self.bead_col,tag='mid')
            im_dapi = im_dapi.astype(np.float32)/self.im_means[self.bead_col]
            self.dax_objs[key].im_dapi = im_dapi/np.mean(im_dapi)
    
    def get_registration_dapi(self,pack_size=512,max_disp = 200,med_sz = 51,ref = 0,overwrite=False):
        """Uses the dapi image to compute the registration for all hybe folders"""
        
        save_file = self.analysis_folder+os.sep+str(self.fov)+'__'+'dapi_drift.npz'
        if os.path.exists(save_file) and not overwrite:
            npz_file = np.load(save_file)
            self.drift_xy_R,self.drift_xy_error_R,self.cor_score_matrix = npz_file['drift_xy_R'],npz_file['drift_xy_error_R'],npz_file['cor_score_matrix']
            return self.drift_xy_R
        self.im_beads = [self.dax_objs[rs[0]].im_dapi for rs in self.Rs]
        self.im_beads = [im_bead-cv2.blur(im_bead,(med_sz,med_sz)) for im_bead in self.im_beads]
        im_packs = [pack(im,psize=pack_size) for im in self.im_beads]
        good_inds = np.argsort([np.std(im_) for im_ in im_packs[0]])[::-1]

        self.cor_score_matrix = []
        self.ts_matrix = []
        for isel in range(len(im_packs)):
            rs = self.Rs[isel]
            ts = []
            maxs = []
            for ig in range(len(good_inds)):
                im_ref = im_packs[ref][good_inds[ig]]
                im_target = im_packs[isel][good_inds[ig]]
                t,max_cor = at.fftalign_2d(im_ref,im_target,max_disp=max_disp,plt_val=False,return_cor_max=True)
                ts.append(t)
                maxs.append(max_cor)
            self.cor_score_matrix.append(maxs)
            self.ts_matrix.append(ts)

        self.cor_score_matrix = np.array(self.cor_score_matrix)
        ts_matrix = np.array(self.ts_matrix)
        cor_score_matrix = self.cor_score_matrix/np.max(self.cor_score_matrix,axis=-1)[...,np.newaxis]
        self.drift_xy = []
        for cor_score,ts in zip(cor_score_matrix,ts_matrix):
            keep = cor_score>0.5
            self.drift_xy.append(np.average(ts[keep],weights=cor_score[keep],axis=0))
        self.drift_xy = np.array(self.drift_xy)
        self.drift_xy_error = ts_matrix-self.drift_xy[:,np.newaxis]
        
        uRs = np.ravel(self.Rs)
        self.drift_xy_R,self.drift_xy_error_R= [],[]
        for iRs,Rs_ in enumerate(self.Rs):
            self.drift_xy_R.extend([self.drift_xy[iRs] for r in Rs_])
            self.drift_xy_error_R.extend([self.drift_xy_error[iRs] for r in Rs_])
        self.drift_xy_R,self.drift_xy_error_R = np.array(self.drift_xy_R),np.array(self.drift_xy_error_R) 
        self.drift_xy_R = self.drift_xy_R[np.argsort(uRs)]
        self.drift_xy_error_R = self.drift_xy_error_R[np.argsort(uRs)]
        
        np.savez(save_file,drift_xy_R=self.drift_xy_R,drift_xy_error_R=self.drift_xy_error_R,cor_score_matrix = self.cor_score_matrix)
        return self.drift_xy_R
        
    def get_Rims(self):
        return [self.dax_objs[r].im for r in np.unique(self.Rs)]
    def get_zxyh_cor(self):
        save_file = self.analysis_folder+os.sep+str(self.fov)+'__'+'fitted_spots.npz'
        if os.path.exists(save_file):
            dic = np.load(save_file)
            self.zxyh_R_cor = dic['zxyh_R_cor']
            return self.zxyh_R_cor
        self.get_ims()
        self.ims_R = self.get_Rims()
        #fit all the readouts
        self.zxyh_R=[]
        for iR,imT in enumerate(self.ims_R):
            if self.verbose: print('Fitting R'+str(iR+1))
            #imT = med_norm3d(imT,ksize=25)
            zxyh = find_local_max(imT,size=3,threshold=1.5)
            self.zxyh_R.append(zxyh)

        #perform drift correction
        self.get_registration_dapi()
        self.drift_zxy = np.concatenate([np.zeros([len(self.drift_xy_R),1]),self.drift_xy_R],axis=-1)    
        self.zxyh_R_cor = []
        for zxyh,d_zxy in zip(self.zxyh_R,self.drift_zxy):
            zxyh_ = zxyh.copy()
            zxyh_[:,:3]-=d_zxy[[0,1,2]]
            self.zxyh_R_cor.append(zxyh_)

        
        np.savez(save_file,zxyh_R_cor=self.zxyh_R_cor)
        return self.zxyh_R_cor
    def get_all_fits(self,sS=3.,ss=1.5,th_brightness= 5,overwrite=False):
        save_file = self.analysis_folder+os.sep+str(self.fov)+'__'+'fitting.npz'
        if os.path.exists(save_file) and not overwrite:
            npz_file = np.load(save_file)
            self.zxyhf_R = npz_file['zxyhf_R']
            return self.zxyhf_R
        self.im3ds = self.get_Rims()
        self.zxyhf_R = []
        for iim,im3d in enumerate(self.im3ds):
            if self.verbose:
                print("Dealing with R: "+str(iim+1))
            zxyhf_im = []
            for im3d_,starts_ in zip(*pack(im3d,psize=512,return_starts=True)):
                zxyhf = self.pixel_fit_image(im3d_,sS=sS,ss=ss,th_brightness= th_brightness)
                zxyhf+=list(starts_)+[0]
                zxyhf_im.extend(zxyhf)
            self.zxyhf_R.append(np.array(zxyhf_im))
        np.savez(save_file,zxyhf_R=self.zxyhf_R)
        return self.zxyhf_R
def get_medians(ims,func=np.median,num_cols=4):
    from tqdm import tqdm_notebook as tqdm
    im_meds = []
    for col_ind in range(num_cols):
        ims_med = []
        for im in tqdm(ims):
            ims_med.append(im.get_slice(im.start(col_ind),im.start(col_ind)+1))
        im_meds.append(func(ims_med,axis=0)[0])
    return im_meds
def pixel_fit_image(im3d,sS=3.,ss=1.5,th_brightness= 5,ksize_max=3,plt_val=False):
    input_= torch.tensor([[im3d]]).cuda()
    ### compute the big gaussian filter ##########
    gaussian_kernel_ = gaussian_kernel(sxyz = [sS,sS,sS],cut_off = 2.5)
    ksz = len(gaussian_kernel_)
    gaussian_kernel_ = torch.FloatTensor(gaussian_kernel_).cuda().view(1, 1, ksz,ksz,ksz)
    #gaussian_kernel_ = gaussian_kernel_.repeat(channels, 1, 1, 1)
    gfilt_big = DataParallel(nn.Conv3d(1,1,ksz, stride=1,padding=0,bias=False)).cuda()
    gfilt_big.module.weight.data = gaussian_kernel_
    gfilt_big.module.weight.requires_grad = False
    gfit_big_ = gfilt_big(pd.ReplicationPad3d(int(ksz/2.))(input_))

    ### compute the small gaussian filter ##########
    gaussian_kernel_ = gaussian_kernel(sxyz = [1,ss,ss],cut_off = 2.5)
    ksz = len(gaussian_kernel_)
    gaussian_kernel_ = torch.FloatTensor(gaussian_kernel_).cuda().view(1, 1, ksz,ksz,ksz)
    #gaussian_kernel_ = gaussian_kernel_.repeat(channels, 1, 1, 1)
    gfilt_sm = DataParallel(nn.Conv3d(1,1,ksz, stride=1,padding=0,bias=False)).cuda()
    gfilt_sm.module.weight.data = gaussian_kernel_
    gfilt_sm.module.weight.requires_grad = False
    gfilt_sm_ = gfilt_sm(pd.ReplicationPad3d(int(ksz/2.))(input_))

    ### compute the maximum filter ##########
    max_filt = DataParallel(nn.MaxPool3d(ksize_max, stride=1,padding=int(ksize_max/2), return_indices=False)).cuda()
    local_max = max_filt(gfilt_sm_)==gfilt_sm_

    g_dif = torch.log(gfilt_sm_)-torch.log(gfit_big_)
    std_ = torch.std(g_dif)
    g_keep = (g_dif>std_*th_brightness)*local_max
    
    #inds = torch.nonzero(g_dif)
    g_keep = g_keep.cpu().numpy()
    inds = np.array(np.nonzero(g_keep)).T
    
    zxyhf = np.array([[],[],[],[]]).T
    zf,xf,yf,hf = zxyhf.T 
    if len(inds):
        brightness = g_dif[inds[:,0],inds[:,1],inds[:,2],inds[:,3],inds[:,4]]
        # bring back to CPU
        zf,xf,yf = inds[:,-3:].T#.cpu().numpy().T
        hf = brightness.cpu().numpy()
        zxyhf = np.array([zf,xf,yf,hf]).T
    torch.cuda.empty_cache()
    if plt_val:
        plt.figure()
        plt.scatter(yf,xf,s=150,facecolor='none',edgecolor='r')
        plt.imshow(np.max(im3d,axis=0),vmax=2)
        plt.show()
    return zxyhf
def gaussian_kernel(sxyz = [2,2,2],cut_off = 2.5):
    sx,sy,sz = sxyz
    ksize = int(np.max((np.array([sx,sy,sz])*2*cut_off)))
    z,x,y = np.indices([ksize]*3)
    exp_ = (x-ksize/2.)**2/(2.*sx**2)+(y-ksize/2.)**2/(2.*sy**2)+(z-ksize/2.)**2/(2.*sz**2)
    kernel = np.exp(-exp_)
    kernel /=np.sum(kernel)
    return kernel
def get_corrections(daxs):
    from tqdm import tqdm_notebook as tqdm
    im_meds,im_means=[],[]
    
    for icol in range(num_cols):
        ims = []
        for dax_fl in tqdm(daxs):
            dax_obj = io.dax_im(dax_fl,num_col=num_cols, start_cutoff=12, end_cutoff=10)
            start = dax_obj.start(2)
            im = dax_obj.get_slice(icol+1,icol+2)[0]
            ims.append(im)
        im_meds.append(np.median(ims,0))
        im_means.append(np.mean(ims,0))

    analysis_folder = data_folder+os.sep+'Analysis'
    if not os.path.exists(analysis_folder):
        os.makedirs(analysis_folder)
    np.save(analysis_folder+os.sep+'im_meds.npy',np.array(im_meds,dtype=np.uint16))
    np.save(analysis_folder+os.sep+'im_means.npy',np.array(im_means,dtype=np.uint16))
def get_mosaic(daxs,icol = 2,zframe = 0,num_cols = 3): 
    from tqdm import tqdm_notebook as tqdm
    im_means = np.load(mean_fl)
    ims_c = im_means[icol]
    ims,xs_um,ys_um=[],[],[]
    for dax_fl in tqdm(daxs):
        dax_obj = io.dax_im(dax_fl,num_col=num_cols, start_cutoff=12, end_cutoff=10)
        im = dax_obj.get_slice(icol+1+zframe,icol+2+zframe)[0]
        im = im/np.mean(im)
        ims.append(im[::1,::-1])
        xs_um.append(dax_obj.stage_x)
        ys_um.append(dax_obj.stage_y)
    ims_c = np.mean(ims,0)
    ims = [im/ims_c for im in ims]
    return ims,mt.compose_mosaic(ims,ims_c*0+1,xs_um,ys_um,fl=None,um_per_pix=0.108,rot = 0,plt_val=False,tag='',monitor=False)