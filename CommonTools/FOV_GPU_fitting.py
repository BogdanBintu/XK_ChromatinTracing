#Bogdan Bintu 12/2018
#This is intended to deal with OB FOVs and use pytorch for fast fitting.
import os,sys,glob
import numpy as np
import matplotlib.pylab as plt

import cv2

import torch
from torch.nn.parallel.data_parallel import DataParallel
from torch import nn
import torch.nn.modules.padding as pd

import IOTools_py3 as io
import MosaicTools as mt
import AlignmentTools_py3 as at

import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

def get_info_bleed(ims,ims_,zxyh_keep,real_i=0,bleed_i=1,block_sizes=np.array([9,9,9]),th_cor = 0.75):
    """ims are the N color x zsize x xsize xysize original images
    ims_ are the N color x zsize x xsize xysize corrected images
    zxyh_keep are the positions where to inspect and get small images
    
    Return ims_real_original,ims_bleed_original,slopes_corrected,constants_corrected,zxyh"""
    from tqdm import tqdm_notebook as tqdm
    cors = []
    ims_real,ims_bleed,ims_realo,ims_bleedo = [],[],[],[]
    print('Gathering data')
    for ispot in tqdm(range(len(zxyh_keep))):
        im_sm1 = at.grab_block_im(ims_[real_i],center=zxyh_keep[ispot,[0,1,2]],block_sizes=block_sizes)
        im_sm2 = at.grab_block_im(ims_[bleed_i],center=zxyh_keep[ispot,[0,1,2]],block_sizes=block_sizes)
        if np.prod(im_sm1.shape)>0:
            ims_real.append(im_sm1)
            ims_bleed.append(im_sm2)
            cor = np.corrcoef(np.ravel(im_sm1),np.ravel(im_sm2))[0,1]
            cors.append(cor)
        
        im_sm1 = at.grab_block_im(ims[real_i],center=zxyh_keep[ispot,[0,1,2]],block_sizes=block_sizes)
        im_sm2 = at.grab_block_im(ims[bleed_i],center=zxyh_keep[ispot,[0,1,2]],block_sizes=block_sizes)
        if np.prod(im_sm1.shape)>0:
            ims_realo.append(im_sm1)
            ims_bleedo.append(im_sm2)
    cors = np.array(cors)
    
    ims_real = np.array(ims_real)
    ims_bleed = np.array(ims_bleed)
    ims_realo = np.array(ims_realo)
    ims_bleedo = np.array(ims_bleedo)
    keep = cors>th_cor #cutoff on correlation
    zxyh_keep_ = zxyh_keep[keep]
    ims_real_keep = ims_real[keep]
    ims_bleed_keep = ims_bleed[keep]
    ims_realo_keep = ims_realo[keep]
    ims_bleedo_keep = ims_bleedo[keep]
    ms,cs=[],[]
    for imR,imB in zip(ims_real_keep,ims_bleed_keep):
        x = np.ravel(imR)-1
        y = np.ravel(imB)-1
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        ms.append(m)
        cs.append(c)
    return ims_real_keep,ims_bleed_keep,ims_realo_keep,ims_bleedo_keep,ms,cs,zxyh_keep_
from scipy.ndimage import zoom
def med_correct(im3d,ksize = 32,cuda=True):
    if cuda:
        sz = im3d.shape[-2:]
        input_= torch.tensor(np.array([im3d],dtype=np.float32)).cuda()
        med2d_filt = DataParallel(MedianPool2d(ksize, stride=ksize)).cuda()
        output_ = med2d_filt(input_)
        imf = nn.functional.interpolate(output_,size =sz,mode='bilinear').cpu().numpy()[0]
        return im3d/imf
    else:
        sz,sxo,syo = im3d.shape
        #assert(sx/ksize==int(sx/ksize) and sy/ksize==int(sy/ksize))
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
class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x

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
    def __init__(self,hybe_flds,ifov,analysis_folder=None,num_cols=3,verbose=True):
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
    def get_corrections(self):
        cor_file = self.analysis_folder+os.sep+'im_means.npy'
        if os.path.exists(cor_file):
            self.im_means = np.load(cor_file)
            return self.im_means
        fldr = self.hybe_flds[0]
        daxs = [fldr+os.sep+fov for fov in self.fovs]
        im_means=[]
        for icol in range(self.num_cols):
            ims = []
            print('color ',icol)
            for dax_fl in daxs:
                dax_obj = io.dax_im(dax_fl,num_col=num_cols, start_cutoff=12, end_cutoff=10)
                im = dax_obj.get_slice(icol+1,icol+2)[0]
                ims.append(im)
            im_means.append(np.mean(ims,0))
        self.im_means = np.array(im_means,dtype=np.uint16)
        np.save(cor_file,self.im_means)
    def construct_daxs(self):
        self.dax_objs = {}
        for ifol,folder in enumerate(self.hybe_flds):
            dax_fl = folder+os.sep+self.fov
            for icol in range(self.num_cols-1):
                dax_obj = io.dax_im(dax_fl,num_col=self.num_cols, start_cutoff=12, end_cutoff=10,
                                    bead_col=self.num_cols-1,color=icol)
                self.dax_objs[self.Rs[ifol][icol]] = dax_obj
    def get_ims(self):
        if self.verbose:
            print('Dealing with flat field...')
        self.get_corrections()
        for key in self.dax_objs:
            if self.verbose:
                print('Dealing with R',key)
            dax_obj = self.dax_objs[key]
            im = dax_obj.get_im()
            im = im.astype(np.float32)/self.im_means[dax_obj.color]
            self.dax_objs[key].im = im/np.mean(im)
            im_dapi = dax_obj.get_slice(self.num_cols,self.num_cols+1)[0]
            im_dapi = im_dapi.astype(np.float32)/self.im_means[self.num_cols-1]
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
    def pixel_fit_image(self,im3d,sS=3.,ss=1.5,th_brightness= 5,plt_val=False):
        self.ss,self.sS = ss,sS
        self.th_brightness = th_brightness
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
        ksize_max = 3#local maximum in 3x3x3 range
        max_filt = DataParallel(nn.MaxPool3d(ksize_max, stride=1,padding=int(ksize_max/2), return_indices=False)).cuda()
        local_max = max_filt(gfilt_sm_)==gfilt_sm_

        g_dif = torch.log(gfilt_sm_)-torch.log(gfit_big_)
        std_ = torch.std(g_dif)
        inds = torch.nonzero((g_dif>std_*th_brightness)*local_max)
        zxyhf = np.array([[],[],[],[]]).T
        zf,xf,yf,hf = zxyhf.T 
        if len(inds):
            brightness = g_dif[inds[:,0],inds[:,1],inds[:,2],inds[:,3],inds[:,4]]
            # bring back to CPU
            torch.cuda.empty_cache()
            zf,xf,yf = inds[:,-3:].cpu().numpy().T
            hf = brightness.cpu().numpy()
            zxyhf = np.array([zf,xf,yf,hf]).T
  
        if plt_val:
            plt.figure()
            plt.scatter(yf,xf,s=150,facecolor='none',edgecolor='r')
            plt.imshow(np.max(im3d,axis=0),vmax=2)
            plt.show()
        return zxyhf
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