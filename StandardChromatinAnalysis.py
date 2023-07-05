#%matplotlib inline

import numpy as np
import glob,os,sys
import cv2
import matplotlib.pyplot as plt
import tifffile

sys.path.append(r'C:\Users\bbintu\bbintu_jackfruit_scripts\XK_ChromatinTracing\CommonTools')
sys.path.append(r'C:\Users\BintuLabUser\Scope3AnalysisScripts\ChromatinTracing2023\CommonTools')
sys.path.append(r'C:\Users\Bogdan\Dropbox\ChromatinTracing2023\CommonTools')
import AlignmentTools_py3 as at
import Fitting_v4 as ft

### Usefull functions


def get_fov(self,fl=None):
    if fl is None: fl =self.fl
    self.fov = os.path.basename(fl.replace(self.extension,'')).split('_')[-1]
def get_xml(self,fl=None):
    if fl is None: fl =self.fl
    get_fov(self,fl=fl)
    xmls = glob.glob(os.path.dirname(fl.replace(self.extension,''))+os.sep+'*.xml')
    self.xml = [xml for xml in xmls if self.fov in os.path.basename(xml)][0]
def get_metadata(self,fl=None,tags = ['stage_position','z_offsets','x_pixels','y_pixels']):
    get_xml(self,fl=fl)
    self.dic_meta = {}
    for tag in tags:
        lns = [ln.split('>')[1].split('<')[0] for ln in open(self.xml,'r') if tag in ln]
        if len(lns)>0:
            self.dic_meta[tag]=lns[0]
    if 'stage_position' in self.dic_meta: self.dic_meta['stage_position']=eval(self.dic_meta['stage_position'])
    if 'z_offsets' in self.dic_meta:
        start,end,zpix,ncols=self.dic_meta['z_offsets'].split(':')
        sizez = (float(end)-float(start))/float(zpix)
        self.dic_meta['ncols'] = int(ncols)
        self.dic_meta['shape'] = [int(sizez),int(self.dic_meta['x_pixels']),int(self.dic_meta['y_pixels'])]
        self.shape = self.dic_meta['shape']
        self.ncol = self.dic_meta['ncols']
def get_xmlS(fl,extension):
    fov = os.path.basename(fl.replace(extension,'')).split('_')[-1]
    xmls = glob.glob(os.path.dirname(fl.replace(extension,''))+os.sep+'*.xml')
    xml = [xml for xml in xmls if fov in os.path.basename(xml)][0]
    return xml
def get_zpix_size(file_sig,extension):
    xml = get_xmlS(file_sig,extension)
    tag = '<z_offsets type="string">'
    z_off = [np.array(ln.split(tag)[-1].split('<')[0].split(':')).astype(float) 
     for ln in open(xml,'r') if tag in ln][0]
    return z_off[2]
def get_H(fld):
    Hbase = os.path.basename(fld)
    if ('R' not in Hbase) and  ('Q' not in Hbase):
        return np.inf
    else:
        try:
            return int(Hbase.split('R')[0].split('Q')[0][1:])
        except:
            return np.inf
def readdax(fl):
    """Read dax file - old imaging format"""
    extension = os.path.basename(fl).split('.')[-1]
    if extension=='dax':
        return np.frombuffer(open(fl,'rb').read(),dtype=np.uint16).reshape((-1,2048,2048)).swapaxes(1,2)
    elif extension=='tif' or extension=='tiff':
        return tifffile.imread(fl)
def get_frame(dax_fl,ind_z=1,sx=2048,sy=2048):
    "returns single frame of a dax/tiff file"
    extension = os.path.basename(dax_fl).split('.')[-1]
    if extension=='dax':
        f = open(dax_fl, "rb")
        bytes_frame = sx*sy*2
        f.seek(bytes_frame*ind_z)
        im_ = np.fromfile(f,dtype=np.uint16,count=sx*sy).reshape([sx,sy]).swapaxes(0,1)
        f.close()
    elif extension=='tif' or extension=='tiff':
        im_ = tifffile.imread(dax_fl,key=ind_z)#.swapaxes(0,1)
    elif extension=='data':
        im_ = zarr.open(dax_fl,'r')[ind_z+1]
    return im_
def get_local_max(im_dif,th_fit,delta=2,delta_fit=3,dbscan=True,return_centers=False,mins=None):
    """Given a 3D image <im_dif> as numpy array, get the local maxima in cube -<delta>_to_<delta> in 3D.
    Optional a dbscan can be used to couple connected pixels with the same local maximum. 
    (This is important if saturating the camera values.)
    Returns: Xh - a list of z,x,y and brightness of the local maxima
    """
    
    z,x,y = np.where(im_dif>th_fit)
    zmax,xmax,ymax = im_dif.shape
    in_im = im_dif[z,x,y]
    keep = np.ones(len(x))>0
    for d1 in range(-delta,delta+1):
        for d2 in range(-delta,delta+1):
            for d3 in range(-delta,delta+1):
                keep &= (in_im>=im_dif[(z+d1)%zmax,(x+d2)%xmax,(y+d3)%ymax])
    z,x,y = z[keep],x[keep],y[keep]
    h = in_im[keep]
    Xh = np.array([z,x,y,h]).T
    if dbscan and len(Xh)>0:
        from scipy import ndimage
        im_keep = np.zeros(im_dif.shape,dtype=bool)
        im_keep[z,x,y]=True
        lbl, nlbl = ndimage.label(im_keep,structure=np.ones([3]*3))
        l=lbl[z,x,y]#labels after reconnection
        ul = np.arange(1,nlbl+1)
        il = np.argsort(l)
        l=l[il]
        z,x,y,h = z[il],x[il],y[il],h[il]
        inds = np.searchsorted(l,ul)
        Xh = np.array([z,x,y,h]).T
        Xh_ = []
        for i_ in range(len(inds)):
            j_=inds[i_+1] if i_<len(inds)-1 else len(Xh)
            Xh_.append(np.mean(Xh[inds[i_]:j_],0))
        Xh=np.array(Xh_)
        z,x,y,h = Xh.T
    im_centers=[]
    if delta_fit!=0 and len(Xh)>0:
        z,x,y,h = Xh.T
        z,x,y = z.astype(int),x.astype(int),y.astype(int)
        im_centers = [[],[],[],[]]
        for d1 in range(-delta_fit,delta_fit+1):
            for d2 in range(-delta_fit,delta_fit+1):
                for d3 in range(-delta_fit,delta_fit+1):
                    if (d1*d1+d2*d2+d3*d3)<=(delta_fit*delta_fit):
                        im_centers[0].append((z+d1))
                        im_centers[1].append((x+d2))
                        im_centers[2].append((y+d3))
                        im_centers[3].append(im_dif[(z+d1)%zmax,(x+d2)%xmax,(y+d3)%ymax])

        im_centers_ = np.array(im_centers)
        im_centers_[-1] -= np.min(im_centers_[-1],axis=0)
        zc = np.sum(im_centers_[0]*im_centers_[-1],axis=0)/np.sum(im_centers_[-1],axis=0)
        xc = np.sum(im_centers_[1]*im_centers_[-1],axis=0)/np.sum(im_centers_[-1],axis=0)
        yc = np.sum(im_centers_[2]*im_centers_[-1],axis=0)/np.sum(im_centers_[-1],axis=0)
        Xh = np.array([zc,xc,yc,h]).T
    if return_centers:
        return Xh,np.array(im_centers)
    return Xh
from tqdm.notebook import tqdm

def resize(im,shape_ = [50,2048,2048]):
    """Given an 3d image <im> this provides a quick way to resize based on nneighbor sampling"""
    z_int = np.round(np.linspace(0,im.shape[0]-1,shape_[0])).astype(int)
    x_int = np.round(np.linspace(0,im.shape[1]-1,shape_[1])).astype(int)
    y_int = np.round(np.linspace(0,im.shape[2]-1,shape_[2])).astype(int)
    return im[z_int][:,x_int][:,:,y_int]

import scipy.ndimage as ndimage
def get_final_cells_cyto(im_polyA,final_cells,icells_keep=None,ires = 4,iresf=10,dist_cutoff=10):
    """Given a 3D im_polyA signal and a segmentation fie final_cells """
    incell = final_cells>0
    med_polyA = np.median(im_polyA[incell])
    med_nonpolyA = np.median(im_polyA[~incell])
    im_ext_cells = im_polyA>(med_polyA+med_nonpolyA)/2


    X = np.array(np.where(im_ext_cells[:,::ires,::ires])).T
    Xcells = np.array(np.where(final_cells[:,::ires,::ires]>0)).T
    from sklearn.neighbors import KDTree

    kdt = KDTree(Xcells[::iresf], leaf_size=30, metric='euclidean')
    icells_neigh = final_cells[:,::ires,::ires][Xcells[::iresf,0],Xcells[::iresf,1],Xcells[::iresf,2]]
    dist,neighs = kdt.query(X, k=1, return_distance=True)
    dist,neighs = np.squeeze(dist),np.squeeze(neighs)

    final_cells_cyto = im_ext_cells[:,::ires,::ires]*0
    if icells_keep is not None:
        keep_cyto = (dist<dist_cutoff)&np.in1d(icells_neigh[neighs],icells_keep)
    else:
        keep_cyto = (dist<dist_cutoff)
    final_cells_cyto[X[keep_cyto,0],X[keep_cyto,1],X[keep_cyto,2]] = icells_neigh[neighs[keep_cyto]]
    final_cells_cyto = resize(final_cells_cyto,im_polyA.shape)
    return final_cells_cyto
def expand_segmentation(imseg_,size=11,resc=5):
    from scipy import ndimage as ndim
    A = imseg_[::resc,::resc,::resc]
    B = ndim.maximum_filter(A,size=size)
    B[A != 0] = A[A != 0]
    B = resize(B,imseg_.shape)
    return B
def slice_pair_to_info(pair):
    sl1,sl2 = pair
    xm,ym,sx,sy = sl2.start,sl1.start,sl2.stop-sl2.start,sl1.stop-sl1.start
    A = sx*sy
    return [xm,ym,sx,sy,A]
def get_coords(imlab1,infos1,cell1):
    xm,ym,sx,sy,A,icl = infos1[cell1-1]
    return np.array(np.where(imlab1[ym:ym+sy,xm:xm+sx]==icl)).T+[ym,xm]
def cells_to_coords(imlab1,return_labs=False):
    """return the coordinates of cells with some additional info"""
    infos1 = [slice_pair_to_info(pair)+[icell+1] for icell,pair in enumerate(ndimage.find_objects(imlab1))
    if pair is not None]
    cms1 = np.array([np.mean(get_coords(imlab1,infos1,cl+1),0) for cl in range(len(infos1))])
    ies=[]
    if len(cms1)>0:
        cms1 = cms1[:,::-1]
    ies = [info[-1] for info in infos1]
    
    if return_labs:
        return imlab1.copy(),infos1,cms1,ies
    return imlab1.copy(),infos1,cms1
def resplit(cells1,cells2,nmin=100):
    """intermediate function used by standard_segmentation.
    Decide when comparing two planes which cells to split"""
    imlab1,infos1,cms1 = cells_to_coords(cells1)
    imlab2,infos2,cms2 = cells_to_coords(cells2)
    if len(cms1)>0 and len(cms2)>0:
        #find centers 2 within the cells1 and split cells1
        cms2_ = np.round(cms2).astype(int)
        cells2_1 = imlab1[cms2_[:,1],cms2_[:,0]]
        imlab1_cells = [0]+[info[-1] for info in infos1]
        cells2_1 = [imlab1_cells.index(cl_) for cl_ in cells2_1]#reorder coords
        #[for e1,e2 in zip(np.unique(cells2_1,return_counts=True)) if e1>0]
        dic_cell2_1={}
        for cell1,cell2 in enumerate(cells2_1):
            dic_cell2_1[cell2] = dic_cell2_1.get(cell2,[])+[cell1+1]
        dic_cell2_1_split = {cell:dic_cell2_1[cell] for cell in dic_cell2_1 if len(dic_cell2_1[cell])>1 and cell>0}
        cells1_split = list(dic_cell2_1_split.keys())
        imlab1_cp = imlab1.copy()
        number_of_cells_to_split = len(cells1_split)
        for cell1_split in cells1_split:
            count = np.max(imlab1_cp)+1
            cells2_to1 = dic_cell2_1_split[cell1_split]
            X1 = get_coords(imlab1,infos1,cell1_split)
            X2s = [get_coords(imlab2,infos2,cell2) for cell2 in cells2_to1]
            from scipy.spatial.distance import cdist
            X1_K = np.argmin([np.min(cdist(X1,X2),axis=-1) for X2 in X2s],0)

            for k in range(len(X2s)):
                X_ = X1[X1_K==k]
                if len(X_)>nmin:
                    imlab1_cp[X_[:,0],X_[:,1]]=count+k
                else:
                    #number_of_cells_to_split-=1
                    pass
        imlab1_,infos1_,cms1_ = cells_to_coords(imlab1_cp)
    
    else:
        imlab1_,infos1_,cms1_ = cells_to_coords(imlab1)
        number_of_cells_to_split = 0
    
    return imlab1_,infos1_,cms1_,number_of_cells_to_split

def converge(cells1,cells2):
    imlab1,infos1,cms1,labs1 = cells_to_coords(cells1,return_labs=True)
    imlab2,infos2,cms2 = cells_to_coords(cells2)

    if len(cms1)>0 and len(cms2)>0:
        #find centers 2 within the cells1 and split cells1
        cms2_ = np.round(cms2).astype(int)
        cells2_1 = imlab1[cms2_[:,1],cms2_[:,0]]
        imlab1_cells = [0]+[info[-1] for info in infos1]
        cells2_1 = [imlab1_cells.index(cl_) for cl_ in cells2_1]#reorder coords
        #[for e1,e2 in zip(np.unique(cells2_1,return_counts=True)) if e1>0]
        dic_cell2_1={}
        for cell1,cell2 in enumerate(cells2_1):
            dic_cell2_1[cell2] = dic_cell2_1.get(cell2,[])+[cell1+1]
            
        dic_cell2_1_match = {cell:dic_cell2_1[cell] for cell in dic_cell2_1 if cell>0}
        cells2_kp = [e_ for e in dic_cell2_1_match for e_ in dic_cell2_1_match[e]]
        modify_cells2 = np.setdiff1d(np.arange(len(cms2)),cells2_kp)
        imlab2_ = imlab2*0
        for cell1 in dic_cell2_1_match:
            for cell2 in dic_cell2_1_match[cell1]:
                xm,ym,sx,sy,A,icl = infos2[cell2-1]
                im_sm = imlab2[ym:ym+sy,xm:xm+sx]
                imlab2_[ym:ym+sy,xm:xm+sx][im_sm==icl]=labs1[cell1-1]
        count_cell = max(np.max(imlab2_),np.max(labs1))
        for cell2 in modify_cells2:
            count_cell+=1
            xm,ym,sx,sy,A,icl = infos2[cell2-1]
            im_sm = imlab2[ym:ym+sy,xm:xm+sx]
            imlab2_[ym:ym+sy,xm:xm+sx][im_sm==icl]=count_cell
    else:
        return imlab1,imlab2
    return imlab1,imlab2_

def standard_segmentation(im_dapi,sz_m=2,sz_M=100,rescz=4,resc=2,sz_min_2d=150,use_gpu=True,th_prob=-20,th_flow=10):
    """Using cellpose with nuclei mode"""
    from cellpose import models, io
    model = models.Cellpose(gpu=use_gpu, model_type='nuclei')
    #decided that resampling to the 4-2-2 will make it faster
    #im_dapi_3d = im_dapi[::rescz,::resc,::resc].astype(np.float32)
    
    
    
    chan = [0,0]
    masks_all = []
    flows_all = []
    for im in im_dapi[::rescz]:
        im_ = im.astype(np.float32)
        img = (cv2.blur(im,(sz_m,sz_m))/cv2.blur(im,(sz_M,sz_M)))[::resc,::resc]
        p1,p99 = np.percentile(img,1),np.percentile(img,99.9)
        print("im limits:",p1,p99)
        img = np.array(np.clip((img-p1)/(p99-p1),0,1),dtype=np.float32)
        masks, flows, styles, diams = model.eval(img, diameter=20, channels=chan,
                                             flow_threshold=th_flow,cellprob_threshold=th_prob,min_size=50,normalize=False)
        masks_all.append(utils.fill_holes_and_remove_small_masks(masks,min_size=sz_min_2d))
        flows_all.append(flows[0])
    masks_all = np.array(masks_all)

    sec_half = list(np.arange(int(len(masks_all)/2),len(masks_all)-1))
    first_half = list(np.arange(0,int(len(masks_all)/2)))[::-1]
    indexes = first_half+sec_half
    masks_all_cp = masks_all.copy()
    max_split = 1
    niter = 0
    while max_split>0 and niter<2:
        max_split = 0
        for index in tqdm(indexes):
            cells1,cells2 = masks_all_cp[index],masks_all_cp[index+1]
            imlab1_,infos1_,cms1_,no1 = resplit(cells1,cells2)
            imlab2_,infos2_,cms2_,no2 = resplit(cells2,cells1)
            masks_all_cp[index],masks_all_cp[index+1] = imlab1_,imlab2_
            max_split += max(no1,no2)
            #print(no1,no2)
        niter+=1
    masks_all_cpf = masks_all_cp.copy()
    for index in tqdm(range(len(masks_all_cpf)-1)):
        cells1,cells2 = masks_all_cpf[index],masks_all_cpf[index+1]
        cells1_,cells2_ = converge(cells1,cells2)
        masks_all_cpf[index+1]=cells2_
    return masks_all_cpf
import pickle


def get_tiles(im_3d,size=256,delete_edges=False):
    sz,sx,sy = im_3d.shape
    if not delete_edges:
        Mz = int(np.ceil(sz/float(size)))
        Mx = int(np.ceil(sx/float(size)))
        My = int(np.ceil(sy/float(size)))
    else:
        Mz = np.max([1,int(sz/float(size))])
        Mx = np.max([1,int(sx/float(size))])
        My = np.max([1,int(sy/float(size))])
    ims_dic = {}
    for iz in range(Mz):
        for ix in range(Mx):
            for iy in range(My):
                ims_dic[(iz,ix,iy)]=ims_dic.get((iz,ix,iy),[])+[im_3d[iz*size:(iz+1)*size,ix*size:(ix+1)*size,iy*size:(iy+1)*size]] 
    return ims_dic
from scipy.spatial.distance import cdist
def get_best_trans(Xh1,Xh2,th_h=1,th_dist = 2,return_pairs=False):
    mdelta = np.array([np.nan,np.nan,np.nan])
    if len(Xh1)==0 or len(Xh2)==0:
        if return_pairs:
            return mdelta,[],[]
        return mdelta
    X1,X2 = Xh1[:,:3],Xh2[:,:3]
    h1,h2 = Xh1[:,-1],Xh2[:,-1]
    i1 = np.where(h1>th_h)[0]
    i2 = np.where(h2>th_h)[0]
    if len(i1)==0 or len(i2)==0:
        if return_pairs:
            return mdelta,[],[]
        return mdelta
    i2_ = np.argmin(cdist(X1[i1],X2[i2]),axis=-1)
    i2 = i2[i2_]
    deltas = X1[i1]-X2[i2]
    dif_ = deltas
    bins = [np.arange(m,M+th_dist*2+1,th_dist*2) for m,M in zip(np.min(dif_,0),np.max(dif_,0))]
    hhist,bins_ = np.histogramdd(dif_,bins)
    max_i = np.unravel_index(np.argmax(hhist),hhist.shape)
    #plt.figure()
    #plt.imshow(np.max(hhist,0))
    center_ = [(bin_[iM_]+bin_[iM_+1])/2. for iM_,bin_ in zip(max_i,bins_)]
    keep = np.all(np.abs(dif_-center_)<=th_dist,-1)
    center_ = np.mean(dif_[keep],0)
    for i in range(5):
        keep = np.all(np.abs(dif_-center_)<=th_dist,-1)
        center_ = np.mean(dif_[keep],0)
    mdelta = center_
    keep = np.all(np.abs(deltas-mdelta)<=th_dist,1)
    if return_pairs:
        return mdelta,Xh1[i1[keep]],Xh2[i2[keep]]
    return mdelta
def get_uniform_points(im_raw,coords=None,sz_big = 21,sz_small=5,size=128,delta_fit=11,delta_fit_z=5,plt_val=False):
    """Normaly used on a dapi image to extract sub-pixel features. Returns Xh"""
    #normalize image
    
    im_raw = im_raw.astype(np.float32)
    sz=sz_big
    im_n = np.array([im_/cv2.GaussianBlur(im_,ksize= (sz*4+1,sz*4+1),sigmaX = sz,sigmaY = sz) for im_ in im_raw])
    sz=sz_small
    im_nn = np.array([cv2.GaussianBlur(im_,ksize= (sz*4+1,sz*4+1),sigmaX = sz,sigmaY = sz) for im_ in im_n])
    if coords is None:
        dic_ims = get_tiles(im_nn,size=size,delete_edges=True)
        coords = []
        for key in dic_ims:
            im_ = dic_ims[key][0]
            coords+=[np.unravel_index(np.argmax(im_),im_.shape)+np.array(key)*size]

    z,x,y = np.array(coords).T
    im_centers = [[],[],[],[]]
    zmax,xmax,ymax = im_nn.shape
    for d1 in range(-delta_fit_z,delta_fit_z+1):
        for d2 in range(-delta_fit,delta_fit+1):
            for d3 in range(-delta_fit,delta_fit+1):
                if (d1*d1+d2*d2+d3*d3)<=(delta_fit*delta_fit):
                    im_centers[0].append((z+d1))
                    im_centers[1].append((x+d2))
                    im_centers[2].append((y+d3))
                    im_centers[3].append(im_nn[(z+d1)%zmax,(x+d2)%xmax,(y+d3)%ymax])

    im_centers_ = np.array(im_centers)
    im_centers_[-1] -= np.min(im_centers_[-1],axis=0)
    zc = np.sum(im_centers_[0]*im_centers_[-1],axis=0)/np.sum(im_centers_[-1],axis=0)
    xc = np.sum(im_centers_[1]*im_centers_[-1],axis=0)/np.sum(im_centers_[-1],axis=0)
    yc = np.sum(im_centers_[2]*im_centers_[-1],axis=0)/np.sum(im_centers_[-1],axis=0)
    h = np.max(im_centers[-1],axis=0)
    Xh = np.array([zc,xc,yc,h]).T
    if plt_val:
        plt.figure()
        plt.imshow(np.max(im_nn,0),vmax=np.median(h)*1.5)
        plt.plot(yc,xc,'rx')
    return Xh
    
    
import cv2
def norm_slice(im,s=50):
    im_=im.astype(np.float32)
    return np.array([im__-cv2.blur(im__,(s,s)) for im__ in im_],dtype=np.float32)

def get_txyz(im_dapi0,im_dapi1,sz_norm=20,sz = 200,nelems=5,plt_val=False):
    """
    Given two 3D images im_dapi0,im_dapi1, this normalizes them by subtracting local background (gaussian size sz_norm)
    and then computes correlations on <nelemes> blocks with highest  std of signal of size sz
    It will return median value and a list of single values.
    """
    im_dapi0_ = norm_slice(im_dapi0,sz_norm)
    im_dapi1_ = norm_slice(im_dapi1,sz_norm)
    dic_ims0 = get_tiles(im_dapi0_,size=sz,delete_edges=True)
    dic_ims1 = get_tiles(im_dapi1_,size=sz,delete_edges=True)
    keys = list(dic_ims0.keys())
    best = np.argsort([np.std(dic_ims0[key]) for key in keys])[::-1]
    txyzs = []
    im_cors = []
    for ib in range(min(nelems,len(best))):
        im0 = dic_ims0[keys[best[ib]]][0].copy()
        im1 = dic_ims1[keys[best[ib]]][0].copy()
        im0-=np.mean(im0)
        im1-=np.mean(im1)
        from scipy.signal import fftconvolve
        im_cor = fftconvolve(im0[::-1,::-1,::-1],im1, mode='full')
        if plt_val:
            plt.figure()
            plt.imshow(np.max(im_cor,0))
            print(txyz)
        txyz = np.unravel_index(np.argmax(im_cor), im_cor.shape)-np.array(im0.shape)+1
        
        im_cors.append(im_cor)
        txyzs.append(txyz)
    txyz = np.median(txyzs,0).astype(int)
    return txyz,txyzs



def get_new_ims(im_dapi0_,im_dapi1_,txyz,ib=0,sz=100):
    dic_ims0 = get_tiles(im_dapi0_,size=sz)
    keys = list(dic_ims0.keys())
    best = np.argsort([np.std(dic_ims0[key]) for key in keys])[::-1]
    start = np.array(keys[best[ib]])*sz
    szs = im_dapi0_.shape
    szf = np.array(im_dapi0_.shape)
    szs = np.min([szs,[sz]*3],axis=0)
    start1,end1 = start+txyz,start+szs+txyz
    start2,end2 = start,start+szs
    start2[start1<0]-=start1[start1<0]
    start1[start1<0]=0
    end2[end1>szf]-=end1[end1>szf]-szf[end1>szf]
    end1[end1>szf]=szf[end1>szf]
    im1=im_dapi1_[start1[0]:end1[0],start1[1]:end1[1],start1[2]:end1[2]]
    im0=im_dapi0_[start2[0]:end2[0],start2[1]:end2[1],start2[2]:end2[2]]
    return im0,im1
def rescale_gaus(im0,resc=4,gauss=False):
    sx,sy,sz = im0.shape
    im0_ = im0[np.arange(0,sx,1./resc).astype(int)]
    im0_ = im0_[:,np.arange(0,sy,1./resc).astype(int)]
    im0_ = im0_[:,:,np.arange(0,sz,1./resc).astype(int)]
    if gauss:
        sigma=resc
        sz0=sigma*4
        X,Y,Z = (np.indices([2*sz0+1]*3)-sz0)
        im_ker = np.exp(-(X*X+Y*Y+Z*Z)/2/sigma**2)

        im0_ = fftconvolve(im0_,im_ker, mode='same')
    return im0_


from tqdm.notebook import tqdm  
    
def align_dapis(ims_align,ims_align_ref,sz_big = 21,sz_small=5,size=128,delta_fit=11,th_h=1.15,th_dist=2,return_pairs=True):
    """Given a set of dapi images, get subpixel features across voxels of size <size> and then calculate the best translation"""

    Xh2 = get_uniform_points(ims_align_ref,sz_big = sz_big,sz_small=sz_small,size=size,delta_fit=delta_fit,plt_val=False)#Xhs[ref]
    Ds,DEs,D_pairs = [],[],[]
    for iIm,im_ in enumerate(ims_align):
        Xh1 = get_uniform_points(im_,sz_big = sz_big,sz_small=sz_small,size=size,delta_fit=delta_fit,plt_val=False)
        dtf,Xh1__,Xh2__ = get_best_trans(Xh1,Xh2,th_h=th_h,th_dist=th_dist,return_pairs=True)
        if np.max(np.abs(dtf))>5:
            coords = (Xh2[:,:3]+dtf).astype(int)
            Xh1 = get_uniform_points(im_,coords,sz_big = sz_big,sz_small=sz_small,
                                    size=size,delta_fit=delta_fit,plt_val=False)
            dtf,Xh1__,Xh2__ = get_best_trans(Xh1,Xh2,th_h=th_h,th_dist=th_dist,return_pairs=True)
        print(dtf)
        dt1 = get_best_trans(Xh1[0::2],Xh2[0::2],th_h=th_h,th_dist=th_dist)
        dt2 = get_best_trans(Xh1[1::2],Xh2[1::2],th_h=th_h,th_dist=th_dist)
        #dt1-dt2
        Ds.append(dtf)
        DEs.append(dt1-dt2)
        D_pairs.append([Xh1__,Xh2__])
    return Ds,DEs,D_pairs
import cv2
def norm_slice(im,s=50):
    
    im_=im.astype(np.float32)
    return np.array([im__-cv2.blur(im__,(s,s)) for im__ in im_],dtype=np.float32)
def fftalign_2d(im1,im2,center=[0,0],max_disp=50,return_cor_max=False,plt_val=False):
    """
    Inputs: 2 2D images <im1>, <im2>, the expected displacement <center>, the maximum displacement <max_disp> around the expected vector.
    This computes the cross-cor between im1 and im2 using fftconvolve (fast) and determines the maximum
    """    
    from scipy.signal import fftconvolve
    im2_=np.array(im2[::-1,::-1],dtype=float)
    im2_-=np.mean(im2_)
    im2_/=np.std(im2_)
    im1_=np.array(im1,dtype=float)
    im1_-=np.mean(im1_)
    im1_/=np.std(im1_)
    im_cor = fftconvolve(im1_,im2_, mode='full')

    sx_cor,sy_cor = im_cor.shape
    center_ = np.array(center)+np.array([sx_cor,sy_cor])/2.
    
    x_min = int(min(max(center_[0]-max_disp,0),sx_cor))
    x_max = int(min(max(center_[0]+max_disp,0),sx_cor))
    y_min = int(min(max(center_[1]-max_disp,0),sy_cor))
    y_max = int(min(max(center_[1]+max_disp,0),sy_cor))
    
    im_cor0=np.zeros_like(im_cor)
    im_cor0[x_min:x_max,y_min:y_max]=1
    im_cor = im_cor*im_cor0
       
    y, x = np.unravel_index(np.argmax(im_cor), im_cor.shape)
    if np.sum(im_cor>0)>0:
        im_cor[im_cor==0]=np.min(im_cor[im_cor>0])
    else:
        im_cor[im_cor==0]=0
    if plt_val:
        plt.figure()
        plt.plot([x],[y],'k+')
        plt.imshow(im_cor,interpolation='nearest')
        plt.xlim(im_cor.shape[0]/2-max_disp,im_cor.shape[0]/2+max_disp)
        plt.ylim(im_cor.shape[1]/2-max_disp,im_cor.shape[1]/2+max_disp)
        plt.show()
    xt,yt=(-(np.array(im_cor.shape)-1)/2.+[y,x]).astype(int)
    if return_cor_max:
        return (xt,yt),np.max(im_cor)
    return xt,yt


def norm_im(im,s=30):
    return np.array([im_.astype(np.float32)/cv2.blur(im_.astype(np.float32),(s,s))for im_ in im])
def norm_im_med(im,im_med):
    if len(im_med)==2:
        return (im.astype(np.float32)-im_med[0])/im_med[1]
    else:
        return im.astype(np.float32)/im_med

def load_fl(file_ = r'Z:\Glass_MERFISH\CGBB_1_25_2022\H10_R28,29,30\Conv_zscan__024.tif',
           bk_file = r'Z:\Glass_MERFISH\CGBB_1_25_2022\H0\Conv_zscan__024.tif',ncols = None,h_dif = 1.25,setz0=True,im_meds = None):
    if ncols is None:
        ncols = os.path.basename(os.path.dirname(file_)).count(',')+2
    print(ncols)
    im = tifffile.imread(file_)
    if type(bk_file) is str:
        im0 = tifffile.imread(bk_file)
    else:
        im0 = bk_file
    ### Drift correction
    im_beads = [im_[ncols-1::ncols,::,::] for im_ in [im,im0]]
    txyz = at.fft3d_from2d(im_beads[0],im_beads[1],gb=33)
    print(txyz)
    if setz0:
        txyz[0]=0
    
    from tqdm import tqdm_notebook as tqdm
    im_sigs = []
    for iim in tqdm(np.arange(ncols)):
        if im_meds is not None:
            im_sig = im[iim::ncols].astype(np.float32)/im_meds[iim%len(im_meds)]
        else:
            im_sig = im[iim::ncols].astype(np.float32)
        if iim!=ncols-1:
            if im_meds is not None:
                im_noise = im0[iim::ncols]/im_meds[iim%len(im_meds)]
            else:
                im_noise = im0[iim::ncols]
            im_sig = at.translate(im_sig, -txyz)
            im_sig = im_sig-im_noise*h_dif
        #else:
        #    im_sig = at.translate(im_sig, -txyz)
        im_sigs.append(im_sig)
    
    im_sigs = np.array(im_sigs)
    return im_sigs,txyz

def read_n_frames(fl_final,extension='.tif'):
    extension_ = os.path.basename(fl_final).split('.')[-1]
    if extension_=='data':
        fl_final_ = os.path.dirname(fl_final)
        fov_ = os.path.basename(fl_final_)
        fl_final_ = os.path.dirname(fl_final_)+os.sep+'Conv_zscan__'+fov_+'.xml'
    else:
        fl_final_ = fl_final.replace(extension,'.xml')
       
    tags = ['<number_frames type="int">','<x_pixels type="int">','<y_pixels type="int">']
    return [[int(ln.split(tag)[-1].split('<')[0]) for ln in open(fl_final_,'r') if tag in ln][0]
           for tag in tags]

import zarr
def load_image(fl):
    extension = os.path.basename(fl).split('.')[-1]
    if extension=='tif' or extension=='tiff':
        im = tifffile.load(fl)
    elif extension=='dax':
        im = readdax(fl)
    elif extension=='data':
        im = zarr.load(fl)[1:]
    return im
from cellpose import utils,plot
class chromatin_analyzer():
    def __init__(self,data_folders,save_folder=None,H0folder=None,extension='.tif',ncol=5,nend=60):
        
        #extract files based on the background image
        data_folder = data_folders[0]
        if H0folder is None:
            H0folder = data_folder+os.sep+'H0'
        fls = np.sort(glob.glob(H0folder+os.sep+r'*'+extension))
        
        #construct save folder
        if save_folder is None:
            save_folder = data_folder+'_Analysis'
        self.save_folder = save_folder
        if not os.path.exists(save_folder): os.makedirs(save_folder)

        folders = np.array([fld for data_folder in data_folders for fld in glob.glob(data_folder+os.sep+r'H*R*') 
                            if not np.any([tg in os.path.basename(fld) for tg in ['rep','T','blank']])])
        #Hs = [int(os.path.basename(fold).split('_R')[-1].split('_PR')[-1].split(',')[-1])for fold in folders]
        
            
        fovs = [os.path.basename(fl) for fl in fls]
        #if len(fovs)==0:
        #   fovs = [os.path.basename(os.path.dirname(fl))+r'/data' for fl in glob.glob(folders[0]+os.sep+r'*/data')]
        ### update self
        self.fovs = fovs
        print('Detected number of fovs: '+str(len(self.fovs)))
        
        self.Rfolders = folders
        self.Rfolders = np.array(self.Rfolders)[np.argsort([get_H(fld) for fld in self.Rfolders])]
        print('Detected number of hybe folders for DNA: '+str(len(self.Rfolders)))
        if True:
            folders_ = []
            for tag in ['Q']:
                folders = np.array([fld for data_folder in data_folders for fld in glob.glob(data_folder+os.sep+r'H*'+tag+'*')]) 
                folders_.extend(folders)
            self.Qfolders = folders_
            self.Qfolders = np.array(self.Qfolders)[np.argsort([get_H(fld) for fld in self.Qfolders])]
            print('Detected number of hybe folders for marker RNA: '+str(len(self.Qfolders)))
        
        self.H0folder = H0folder
        self.fls = fls
        self.ncol=ncol
        self.nend=nend
        self.extension = extension
        
    def set_fov(self,ifl):
        self.fl = self.fls[ifl]
        get_metadata(self)
        print("Setting file: "+self.fl)
    def compute_flat_fields(self,fls=None,nend=None,ncol=None):
        if fls is None: fls = self.fls
        if nend is None: nend = self.nend
        if ncol is None: ncol = self.ncol
        #fls,nend,ncol = self.fls,self.nend,self.ncol
        im_meds = []
        for icol in range(ncol):
            save_med_fl = self.save_folder+r'\median_col'+str(icol)+'.npy'
            if not os.path.exists(save_med_fl):
                im_med = np.median([get_frame(fl,int((nend//2)*ncol+icol)) for fl in tqdm(fls)],0)
                np.save(save_med_fl,im_med)
            else:
                im_med = np.load(save_med_fl)
            im_meds.append(im_med)
        self.im_meds = im_meds
        return  im_meds
    def save_image_dapi_segmentation(self,fr=20):
        img = get_frame(self.fl,fr*self.ncol-1)
        masks_ = self.imseg_[fr]
        from cellpose import utils
        outlines = utils.masks_to_outlines(masks_)
        img = (cv2.blur(img,(10,10))/cv2.blur(img,(100,100)))#[::resc,::resc]
        p1,p99 = np.percentile(img,1),np.percentile(img,99.9)
        img = np.array(np.clip((img-p1)/(p99-p1),0,1),dtype=np.float32)
        outX, outY = np.nonzero(outlines)
        imgout= np.dstack([img]*3)
        imgout[outX, outY] = np.array([1,0,0]) # pure red
        fig = plt.figure(figsize=(20,20))
        plt.imshow(imgout)
        fig.savefig(self.save_seg_fl.replace('.npz','__segim'+str(fr)+'.png'))
    def compute_cell_segmentation(self,nstart=1,nend=60,rescz=6,resc=2,sz_min_2d=150,th_prob=-20,th_flow=10,force=False,check_3Dseg=False,use_gpu=True,expand=11):
        self.prev_fl_dapi = getattr(self,'prev_fl_dapi','None')
        if self.prev_fl_dapi!=self.fl:
            
            #print("Here")
            self.prev_fl_dapi = self.fl
            ncol = self.ncol
            fl_final = self.fl
            save_folder = self.save_folder
            fl_final_ = fl_final.replace(r'\data','')
            save_seg_fl = save_folder+os.sep+os.path.basename(fl_final_).split('.')[0]+'--dapi_seg.npz'
            self.save_seg_fl = save_seg_fl
            print(save_seg_fl)
            save_seg_fl_add = save_folder+os.sep+os.path.basename(fl_final_).split('.')[0]+'--dapi_seg_add.npy'
            redo = force
            redo = redo or (not os.path.exists(save_seg_fl))
            redo = redo or (not os.path.exists(save_seg_fl_add))
            if not redo:
                redo = redo or (self.fl!=str(np.load(save_seg_fl_add)))
            
            
            ### ---- check file existence and decide whether to recompute segmentation
            if redo:
                print("Computing the segmentation...")
                ###----save dapiseg filename for reference
                np.save(save_seg_fl_add,self.fl)
                ###-------load dapi flat field
                im_med_dapi = self.im_meds[-1]#np.load(save_folder+r'\median_col'+str(ncol-1)+'.npy')
                
                ###-------load dapi
                
                im_dapi = np.array([norm_im_med(get_frame(fl_final,ifr*ncol-1),im_med_dapi) for ifr in range(nstart,nend,rescz)])
                
                ###-------apply segmentation
                imseg = standard_segmentation(im_dapi,rescz=1,resc=resc,sz_min_2d=sz_min_2d,use_gpu=use_gpu,th_prob=th_prob,th_flow=th_flow)
                #np.save(save_seg_fl,imseg)
                np.savez_compressed(save_seg_fl,imseg)
                #self.save_image_dapi_segmentation()
                ###------inspect 3d segmentation
                if check_3Dseg:
                    imoutline = np.array([utils.masks_to_edges(im_) for im_ in imseg])
                    imoutline = resize(imoutline,im_dapi.shape)

                    fig = plt.figure()
                    im_dapi_ = im_dapi.copy()
                    im_dapi_[imoutline>0]=np.max(im_dapi_)
                    tifffile.imshow(im_dapi_,cmap='gray',figure=fig)
                    #tifffile.imshow(imoutline,alpha=0.1,figure=fig)
            else:
                print("Detecting existing segmentation file and loading it...")
                print(save_seg_fl)
                imseg = np.load(save_seg_fl)['arr_0']
            self.shape = read_n_frames(fl_final,extension=self.extension)
            self.shape[0] = self.shape[0]//self.ncol

            
            self.imseg_ = resize(imseg,self.shape)
            
            self.save_image_dapi_segmentation()
            if expand!=0:
                self.imseg_ = expand_segmentation(self.imseg_,size=expand,resc=5)
            self.outpix = utils.outlines_list(self.imseg_[len(self.imseg_)//2])
            
    def load_background_image(self,rescz=1):
        self.prev_fl = getattr(self,'prev_fl','None')
        if self.prev_fl!=self.fl:
            self.prev_fl = self.fl
            fl_final,ncol = self.fl,self.ncol
            save_folder = self.save_folder

            self.bk_im = load_image(fl_final) ### raw background image
            self.bk_im = self.bk_im[:(len(self.bk_im)//ncol)*ncol]
            shape_2d = list(self.bk_im.shape[1:])
            self.bk_im = self.bk_im.reshape([-1,ncol]+shape_2d)
            self.bk_im = self.bk_im[::rescz].reshape([-1]+shape_2d)
            self.im_dapi_bk =norm_im_med(self.bk_im[ncol-1::ncol],self.im_meds[-1]) ### flatfield corrected image
    def set_hybe(self,ihybe):
        self.Rfolder = self.Rfolders[ihybe]
        self.file_sig = self.fl.replace(self.H0folder,self.Rfolder)
        print("Focusing on file: "+str(self.file_sig))
    def set_hybe_RNA(self,ihybe):
        self.Rfolder = self.Qfolders[ihybe]
        self.file_sig = self.fl.replace(self.H0folder,self.Rfolder)
        print("Focusing on file: "+str(self.file_sig))
    def load_signal_image(self):
        self.signal_im = load_image(self.file_sig) ### raw signal file
        ncol = self.ncol
        self.im_dapi_signal = norm_im_med(self.signal_im[ncol-1::ncol],self.im_meds[-1])
    def compute_drift(self,sz=512,nelems=5):
        #txyz = at.fft3d_from2d(,self.im_dapi_bk,gb=33)
        #txyz,txyzs = get_txyz(self.im_dapi_bk,self.im_dapi_signal,sz_norm=20,sz = 400,nelems=7)
        im_bk = self.im_dapi_bk
        im_sig = self.im_dapi_signal
        len_bk = len(im_bk)
        len_sig = len(im_sig)
        pix_sig = get_zpix_size(self.file_sig,self.extension)
        pix_bk = get_zpix_size(self.fl,self.extension)
        nbk = int(np.round(len_bk/pix_sig*pix_bk))
        inds_bk = np.round(np.linspace(0,len_bk-1,nbk)).astype(int)
        inds_sig = np.arange(len_sig)
        nmin = np.min([len(inds_bk),len(inds_sig)])
        inds_bk,inds_sig = inds_bk[:nmin],inds_sig[:nmin]
        txyz,txyzs = get_txyz(self.im_dapi_bk[inds_bk],self.im_dapi_signal[inds_sig],sz_norm=30,sz = sz,nelems=5)
        #at.fft3d_from2d(chr_.im_dapi_signal,chr_.im_dapi_bk,gb=30,max_disp=400)
        error = np.median(np.abs(np.array(txyzs)-txyz))
        print('Pixel based registrtion:'+str(txyz),error)
        #Ds,DEs,D_pairs = align_dapis([self.im_dapi_signal],self.im_dapi_bk,sz_big = 21,
        #                             sz_small=5,size=256,delta_fit=11,th_h=1.15,th_dist=2,return_pairs=True)
                                     
        Ds = [txyz]
        DEs = None
        D_pairs = txyzs
        self.txyz,self.Ds,self.DEs,self.D_pairs = txyz,Ds,DEs,D_pairs
        if np.abs(np.round(self.Ds[0][0])-self.txyz[0])>=2:################
            self.Ds[0][0]=self.txyz[0]
        
        if np.any(np.isnan(self.Ds)):
            self.Ds = [self.txyz]

        self.dic_drift = {'txyz':self.txyz,'Ds':self.Ds,'DEs':self.DEs,'D_pairs':self.D_pairs,'drift_fl':self.fl}
        """
        ### check rotation
        Xh1,Xh2 = self.D_pairs[0]
        X1,X2 = Xh1[:,:3],Xh2[:,:3]
        X1c = X1-np.mean(X1,0)
        X2c = X2-np.mean(X2,0)
        U,S,V = np.linalg.svd(np.dot(X1c.T,X2c))
        R = np.dot(U,V)

        Rs = []
        for isub in range(2):
            X1,X2 = Xh1[isub::2,:3],Xh2[isub::2,:3]
            X1c = X1-np.mean(X1,0)
            X2c = X2-np.mean(X2,0)
            U,S,V = np.linalg.svd(np.dot(X1c.T,X2c))
            R_ = np.dot(U,V)
            Rs.append(R_)
        vec = np.array(self.shape)/2
        self.R_rot=R
        self.R_rots=Rs
        estimate_error = np.round(np.max(np.abs(np.dot(vec[:,np.newaxis],np.array([R[0,1],R[0,2],R[1,2]])[np.newaxis]))),2)
        print("Max estimate error based on rotation (in pixels):" + str(estimate_error))
        """

    def fit_points_in_cells(self,nmax=50,nbetter=6,subtract=True,icols=None):
        ncol = self.ncol
        def norm_im_subtr(im,s=30):
            return np.array([im_.astype(np.float32)-cv2.blur(im_.astype(np.float32),(s,s)) for im_ in im])
        
        self.Tref = np.round(self.Ds[0]).astype(int)
        Tref = self.Tref
        slices_bk = tuple([slice(-t_,None,None) if t_<=0 else slice(None,-t_,None) for t_ in Tref])
        slices_sig = tuple([slice(t_,None,None) if t_>=0 else slice(None,t_,None) for t_ in Tref])
        
        from scipy.ndimage import find_objects
        imseg__ = self.imseg_[slices_bk][1:]
        slices_cells = find_objects(imseg__)
        
        
        dic_pts_cell = {}
        self.ims_signal_final_norm = []
        ### iterate through colors
        if icols is None: icols = np.arange(ncol-1)
        for icol in icols:
            im_sig = self.signal_im[icol::self.ncol][1:]
            im_bk = self.bk_im[icol::self.ncol][1:]
            len_bk = len(im_bk)
            len_sig = len(im_sig)
            pix_sig = get_zpix_size(self.file_sig,self.extension)
            pix_bk = get_zpix_size(self.fl,self.extension)
            nbk = int(np.round(len_bk/pix_sig*pix_bk))
            inds_bk = np.round(np.linspace(0,len_bk-1,nbk)).astype(int)
            inds_sig = np.arange(len_sig)
            nmin = np.min([len(inds_bk),len(inds_sig)])
            inds_bk,inds_sig = inds_bk[:nmin],inds_sig[:nmin]
            im_sig_ = norm_im_med(im_sig[inds_sig],self.im_meds[icol])
            im_bk_ = norm_im_med(im_bk[inds_bk],self.im_meds[icol])
            im_sig__ = im_sig_[slices_sig]
            im_bk__ = im_bk_[slices_bk]
            if subtract:
                ratio = np.median(im_sig__[::5,::10,::10]/im_bk__[::5,::10,::10])
                ratio = np.min([ratio,1.25])
                im_sig__f = im_sig__-im_bk__*ratio
            else:
                im_sig__f = im_sig__
            im_sig__ff  = norm_im_subtr(im_sig__f,s=30)
            
            self.ims_signal_final_norm.append(im_sig__ff)
            dic_pts_cell[icol] = {}
            
            ### iterate through cells
            for icell in tqdm(np.arange(len(slices_cells))):
                slice_cell = slices_cells[icell]
                if slice_cell is not None:

                    im_sig__f_ = im_sig__f[slice_cell].copy() #zoom in on cell not normalized signal
                    im_sig__ff_ = im_sig__ff[slice_cell].copy() #zoom in on cell normalized signal 
                    inside_cell = imseg__[slice_cell]==(icell+1) #pixels inside cell
                    volume = np.sum(inside_cell ) #cell volume
                    im_th = np.std(im_sig__ff_[inside_cell])*3 #stop at 3 times the std
                    Xh_cell = get_local_max(im_sig__ff_,im_th,delta_fit=0) #find local maxima in the box of the cell
                    pfits_ = []
                    pfits_better = []
                    if len(Xh_cell)>0:
                        X_cell = Xh_cell[:,:3].astype(int) 

                        ### select the top nmax points in the cell in order of brightness and fit those with gaussians
                        is_inside = inside_cell[X_cell[:,0],X_cell[:,1],X_cell[:,2]]
                        Xh_cell = Xh_cell[is_inside]
                        h_cell = Xh_cell[:,-1]
                        XT=Xh_cell[np.argsort(h_cell)[::-1]][:nmax]

                        centers_zxy = XT[:,:-1]
                        pfits_ = ft.fast_fit_big_image(im_sig__f_,centers_zxy,radius_fit = 4,better_fit = False,avoid_neigbors=False,verbose=False)
                        
                        if len(pfits_)>0:
                            starts = [sl.start for sl in slice_cell]
                            pfits_[:,1:4]=pfits_[:,1:4]+starts+[t_ if t_>0 else 0 for t_ in Tref] ### replace in original image coordinates

                            
                        
                        pfits_better = ft.fast_fit_big_image(im_sig__f_,centers_zxy[:nbetter],radius_fit = 4,
                                                    better_fit = True,avoid_neigbors=False,verbose=False)
                        if len(pfits_better)>0:
                            starts = [sl.start for sl in slice_cell]
                            pfits_better[:,1:4]=pfits_better[:,1:4]+starts+[t_ if t_>0 else 0 for t_ in Tref] ### replace in original image coordinates
                        
                            
                    dic_pts_cell[icol][icell+1] = {'fits':pfits_,'volume':volume,'slice':slice_cell,'fits_better':pfits_better}
                    #dic_pts_cell[icol][icell] = pfits
                    if False:
                        plt.figure()
                        plt.imshow(np.max(im_sig__f_,0))
                        plt.plot(pfits_[:,3],pfits_[:,2],'r.')
                        
        self.dic_pts_cell=dic_pts_cell
        
    def check_finished_file(self):
        file_sig = self.file_sig
        ext = os.path.basename(self.file_sig).split('.')[-1]
        if ext=='data':
            file_sig = os.path.dirname(self.file_sig)
        save_folder = self.save_folder
        fov_ = os.path.basename(file_sig).split('.')[0]
        hfld_ = os.path.basename(os.path.dirname(file_sig))
        self.base_save = self.save_folder+os.sep+fov_+'--'+hfld_
        self.dic_pts_cell_fl = self.base_save+'--'+'dic_pts_cell.pkl'
        
        fl_final = self.fl
        save_folder = self.save_folder
        fl_final_ = fl_final.replace(r'\data','')
        save_seg_fl = save_folder+os.sep+os.path.basename(fl_final_).split('.')[0]+'--dapi_seg.npz'
        self.save_seg_fl = save_seg_fl
        
        
        return os.path.exists(self.dic_pts_cell_fl) and os.path.exists(self.save_seg_fl)
    
    def save_fits(self,plt_val=True):
        if plt_val:
            for icol in range(self.ncol-1):
                dic_pts_cell = self.dic_pts_cell
                if icol in dic_pts_cell:
                    im_sig = self.ims_signal_final_norm[icol]

                    vmax,vmin = np.median([(dic_pts_cell[icol][icell]['fits'][0][0],dic_pts_cell[icol][icell]['fits'][0][4]) 
                                           for icell in dic_pts_cell[icol] 
                                           if dic_pts_cell[icol][icell]['volume']>200 and len(dic_pts_cell[icol][icell]['fits'])>0],axis=0)
                    fig = plt.figure(figsize=(30,30))
                    
                    outpix = self.outpix
                    #tx_,ty_ = [t_ if t_>0 else 0 for t_ in self.Tref][1:]
                    ty_,tx_ = [t_ if t_>0 else 0 for t_ in -self.Tref][1:]
                    for k in range(len(outpix)):
                        plt.plot(outpix[k][:,0]-tx_, outpix[k][:,1]-ty_, color='r')
                    plt.imshow(np.max(im_sig,axis=0),vmin=0,vmax=vmax,cmap='gray')
                    fig.savefig(self.base_save+'_signal-col'+str(icol)+'.png')
                    plt.close('all')
        pickle.dump([self.dic_drift,self.dic_pts_cell],open(self.dic_pts_cell_fl,'wb'))
        
    def fit_RNA_points_in_cells(self,th_keep=12,subtract_bk=False):
        ncol = self.ncol
        def norm_im_subtr(im,s=30):
            return np.array([im_.astype(np.float32)-cv2.blur(im_.astype(np.float32),(s,s)) for im_ in im])

        self.Tref = np.round(self.Ds[0]).astype(int)
        Tref = self.Tref
        slices_bk = tuple([slice(-t_,None,None) if t_<=0 else slice(None,-t_,None) for t_ in Tref])
        slices_sig = tuple([slice(t_,None,None) if t_>=0 else slice(None,t_,None) for t_ in Tref])

        from scipy.ndimage import find_objects
        imseg__ = self.imseg_[slices_bk][1:]
        slices_cells = find_objects(imseg__)


        dic_pts_cell = {}
        self.ims_signal_final_norm = []
        self.th_fits,self.med_fits = [],[]
        self.Xh_RNAs = []
        ### iterate through colors
        for icol in range(ncol-1):
        
            im_sig = self.signal_im[icol::self.ncol][1:]
            im_bk = self.bk_im[icol::self.ncol][1:]
            len_bk = len(im_bk)
            len_sig = len(im_sig)
            pix_sig = get_zpix_size(self.file_sig,self.extension)
            pix_bk = get_zpix_size(self.fl,self.extension)
            nbk = int(np.round(len_bk/pix_sig*pix_bk))
            inds_bk = np.round(np.linspace(0,len_bk-1,nbk)).astype(int)
            inds_sig = np.arange(len_sig)
            nmin = np.min([len(inds_bk),len(inds_sig)])
            inds_bk,inds_sig = inds_bk[:nmin],inds_sig[:nmin]
            im_sig_ = norm_im_med(im_sig[inds_sig],self.im_meds[icol])
            im_bk_ = norm_im_med(im_bk[inds_bk],self.im_meds[icol])
            im_sig__ = im_sig_[slices_sig]
            im_bk__ = im_bk_[slices_bk]
            
            if subtract_bk:
                
                ratio = np.median(im_sig__[::5,::10,::10]/im_bk__[::5,::10,::10])
                ratio = np.min([ratio,1.25])
                im_sig__f = im_sig__-im_bk__*ratio
                #im_sig__f = im_sig__-im_bk__*1.25
            im_sig__ff  = norm_im_subtr(im_sig__f,s=30)

            med_fit = np.median(im_sig__ff)
            th_fit = med_fit+th_keep*np.median(np.abs(im_sig__ff-med_fit))
            self.th_fits.append(th_fit)
            self.med_fits.append(med_fit)
            Xh = get_local_max(im_sig__ff,th_fit,delta=2,delta_fit=3,dbscan=True,return_centers=False,mins=None)
            self.Xh_RNAs.append(Xh)
            self.ims_signal_final_norm.append(im_sig__ff)
            dic_pts_cell[icol] = {}

            ### iterate through cells
            for icell in tqdm(np.arange(len(slices_cells))):
                slice_cell = slices_cells[icell]
                if slice_cell is not None:
                    inside_cell = imseg__[slice_cell]==(icell+1) #pixels inside cell
                    volume = np.sum(inside_cell) #cell volume
                    starts = [sl.start for sl in slice_cell]
                    #ends = [sl.end for sl in slice_cell]
                    pfits_ = []
                    if len(Xh):
                        Xh_cell = Xh.copy()
                        Xh_cell[:,:3]-=starts
                        keep_ = np.all((Xh_cell[:,:3]>=0)&(Xh_cell[:,:3]<np.array(inside_cell.shape)[np.newaxis]),axis=-1)
                        Xh_cell = Xh_cell[keep_]
                        if len(Xh_cell)>0:
                            X_cell = Xh_cell[:,:3].astype(int) 

                            ### select the top nmax points in the cell in order of brightness and fit those with gaussians
                            is_inside = inside_cell[X_cell[:,0],X_cell[:,1],X_cell[:,2]]
                            Xh_cell = Xh_cell[is_inside]
                            #h_cell = Xh_cell[:,-1]
                            #XT=Xh_cell[np.argsort(h_cell)[::-1]]#[:nmax]

                            #centers_zxy = XT[:,:-1]
                            #pfits = ft.fast_fit_big_image(im_sig__f_,centers_zxy,radius_fit = 4,better_fit = False,avoid_neigbors=False,verbose=False)
                            pfits_ = Xh_cell[:,[3,0,1,2]]
                            if len(pfits_)>0:
                                pfits_[:,1:4]=pfits_[:,1:4]+starts+[t_ if t_>0 else 0 for t_ in Tref] ### replace in original image coordinates
                    dic_pts_cell[icol][icell+1] = {'fits':pfits_,'volume':volume,'slice':slice_cell}
                    #dic_pts_cell[icol][icell] = pfits
                    if False:
                        plt.figure()
                        plt.imshow(np.max(im_sig__f_,0))
                        plt.plot(pfits[:,3],pfits[:,2],'r.')

        self.dic_pts_cell_RNA=dic_pts_cell
        
    def re_th_RNA(self,icols=[0,1,2],th_s = [1.4,1.4,1.4],save=None):
        self.dic_drift,self.dic_pts_cell_RNA,self.th_fits,self.med_fits = pickle.load(open(self.dic_pts_cell_fl,'rb'))
        for icol,th_ in zip(icols,th_s):
            th_col = (self.th_fits[icol]-self.med_fits[icol])*th_+self.med_fits[icol]

            dic_pts_cell_RNA_ = self.dic_pts_cell_RNA[icol]
            for icell in dic_pts_cell_RNA_:
                fits = dic_pts_cell_RNA_[icell]['fits']
                if len(fits)>0:
                    self.dic_pts_cell_RNA[icol][icell]['fits'] = fits[fits[:,0]>th_col]
        if save is not None:
            pickle.dump([self.dic_drift,self.dic_pts_cell_RNA,self.th_fits,self.med_fits],
                        open(self.dic_pts_cell_fl.replace('.pkl',save+'.pkl'),'wb'))
    def save_fits_RNA(self,icols=None,plt_val=True):
        if plt_val:
            if icols is None:
                icols =  range(self.ncol-1)
            for icol in icols:
                dic_cell_ = self.dic_pts_cell_RNA[icol]

                fig = plt.figure(figsize=(40,40))
                for icell in dic_cell_:
                    X = dic_cell_[icell]['fits']
                    if len(X):
                        h,z,x,y = X.T
                        tz_,tx_,ty_ = [t_ if t_>0 else 0 for t_ in self.Tref]
                        plt.scatter(y-ty_,x-tx_,facecolors="None", edgecolors='b',s=100)

                    
                outpix = self.outpix
                #tx_,ty_ = [t_ if t_>0 else 0 for t_ in self.Tref][1:]
                ty_,tx_ = [t_ if t_>0 else 0 for t_ in -self.Tref][1:]
                for k in range(len(outpix)):
                    plt.plot(outpix[k][:,0]-tx_, outpix[k][:,1]-ty_, color='r')
                
                plt.imshow(np.max(self.ims_signal_final_norm[icol],0),vmin=0,vmax=2*self.th_fits[icol],cmap='gray')
                fig.savefig(self.base_save+'_signal-col'+str(icol)+'.png')

                fig = plt.figure(figsize=(40,40))
                for k in range(len(outpix)):
                    plt.plot(outpix[k][:,0]-tx_, outpix[k][:,1]-ty_, color='r')
                plt.imshow(np.max(self.ims_signal_final_norm[icol],0),vmin=0,vmax=2*self.th_fits[icol],cmap='gray')
                fig.savefig(self.base_save+'_signal-col'+str(icol)+'__simple.png')

                plt.close('all')
        pickle.dump([self.dic_drift,self.dic_pts_cell_RNA,self.th_fits,self.med_fits],open(self.dic_pts_cell_fl,'wb'))
        pickle.dump(self.Xh_RNAs,
                    open(self.dic_pts_cell_fl.replace('.pkl','_Xh_RNAs.pkl'),'wb'))