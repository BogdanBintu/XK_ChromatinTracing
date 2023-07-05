#Bogdan Bintu 7/2018
#This is intended to deal with the mosaics created either by Steve (part of storm-control)
#or generically by files having x,y and ims
from __future__ import print_function

from PIL import Image
import pickle
import numpy as np
import glob
import os
import matplotlib.pylab as plt

def rotate(image, angle, center=None, scale=1.0):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]
    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)
    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    # return the rotated image
    return rotated
def norm_sets(ims,norm_dapi=True,perc =80,nl = 30):
    normns = [1 for im_ in ims ]
    if norm_dapi and len(ims)>nl:
        normns = [np.percentile(im_,perc) for im_ in ims]        
        nll = int(nl/2)
        norms_ = list(normns[:nll][::-1])+list(normns)+list(normns[-nll:][::-1])
        norms_ = [np.median(norms_[il:il+nl]) for il in range(len(norms_)-nl)]
        normns = [e/norms_[0] for e in norms_]
    return normns
    
def get_new_name(dax_fl):
    dax_fl_ = dax_fl
    if not os.path.exists(dax_fl_):
        if dax_fl_.split('.')[-1]=='dax':
            dax_fl_ = dax_fl_.replace('.dax','.dax.zst')
        if dax_fl_.split('.')[-1]=='zst':
            dax_fl_ = dax_fl_.replace('.zst','')
    return dax_fl_
    

from tqdm import tqdm_notebook as tqdm
import IOTools as io
import cv2
def load_im(dax_fl,custom_frms=None,dims=(2048,2048),cast=True):
    dax_fl_ = dax_fl
    if dax_fl_.split('.')[-1]=='npy':
        return np.load(dax_fl_)#np.swapaxes(np.load(dax_fl_),-1,-2)
    dax_fl_ = get_new_name(dax_fl)
            
    
    ### patch missed frames
    master_folder = os.path.dirname(dax_fl)
    bucket_fl =master_folder+os.sep+'buckets.npy'
            
    if dax_fl_.split('.')[-1]=='dax':
        if custom_frms is not None and not os.path.exists(bucket_fl):
            daxReader = io.DaxReader(dax_fl_)
            im = np.array([daxReader.loadAFrame(frm) for frm in custom_frms])
        else:
            im = io.DaxReader(dax_fl).loadMap()
            #im = np.swapaxes(np.frombuffer(open(dax_fl_,'rb').read(),dtype=np.uint16).reshape([-1]+list(dims)),1,2)
    else:
        import blosc
        #im = np.swapaxes(np.frombuffer(blosc.decompress(open(dax_fl_,'rb').read()),dtype=np.uint16).reshape([-1]+list(dims)),1,2)
        im = np.frombuffer(blosc.decompress(open(dax_fl_,'rb').read()),dtype=np.uint16).reshape([-1]+list(dims))
        if custom_frms is not None and not os.path.exists(bucket_fl):
            im = np.array([im[frm] for frm in custom_frms if frm<len(im)])
    
    
    
    ### patch missed frames
    if os.path.exists(bucket_fl):
        buckets = np.load(bucket_fl)
        buckets = [list(b) for b in buckets]
        ires = 10
        for iim in range(len(im)):
            cors = [np.corrcoef(im[iim,::ires,::ires].ravel(),im[b[-1],::ires,::ires].ravel())[0,1] for b in buckets]
            best = np.argmax(cors)
            if iim>buckets[best][-1]:
                buckets[best]+=[iim]
        #buckets = [b[1:]for b in buckets]
        len_ = np.min([len(b) for b in buckets])
        im = im[[b[l] for l in range(len_) for b in buckets]]
        if custom_frms is not None:
            im = im[[frn for frm in custom_frms if frm<len(im)]]
    return im
    
def save_tile_image_and_label_imagesets(ims,xys,save_file,resc=2,custom_frms=None,pix_size_=0.108,
                        max_impose=True,verbose=False,im_cor__=None,
                              rotation_angle=0,add_txt=True,norm_dapi=False,invertX=False,invertY=False):

        import tifffile
        save_pars_all=[]
        pix_size=pix_size_*resc
        fls_iter = [str(i)for i in range(len(ims))]
        #return ims
        if im_cor__ is None:
            #im_cor = np.median(ims,axis=0)
            #ims = [im_/im_cor*np.median(im_cor) for im_ in ims]
            
            ims = np.array(ims,dtype=np.float32)
            ims_sort = np.sort(ims,0)
            delta = 10
            ielems = (np.arange(delta,100-delta,delta)*len(ims)/100).astype(int)
            ims_sort_ = ims_sort[ielems]
            vec = np.median(ims_sort_.reshape(ims_sort_.shape[0],-1),-1)
            ima = ims_sort_[0]#np.median(ims_sort_,0)
            imb = (ims_sort_[-1]-ima)/(vec[-1]-vec[0])#[:,np.newaxis,np.newaxis])[-1]*vec[-1]
            im_cor = imb
            ims = [(im_-ima)/imb+np.min(ima/imb) for im_ in ims]
        else:
            im_cor = im_cor__[1:-1:resc,1:-1:resc].copy()
            ims = [im_/im_cor*np.median(im_cor) for im_ in ims]
        xys_=np.array(xys,dtype=int)
        xys_=xys_-np.expand_dims(np.min(xys_,axis=0), axis=0)

        if len(ims[0].shape)>2:
            sx,sy,sz = ims[0].shape
            dim_base = list(np.max(xys_,axis=0)+np.array([sx+1,sy+1]))
            dim_base+=[sz]
        else:
            sx,sy = ims[0].shape
            dim_base = list(np.max(xys_,axis=0)+np.array([sx+1,sy+1]))
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        im_base = np.zeros(dim_base,dtype=float)
        im_compare = np.zeros(dim_base,dtype=float)
        infos_filename = '.'.join(save_file.split('.')[:-1])+'.infos'
        fid = open(infos_filename,'w')
        

        normns = norm_sets(ims,norm_dapi=norm_dapi,perc =80,nl = 30)
        
        
        for i_,(im,(x,y),norm) in enumerate(zip(ims[:],xys_[:],normns)):
            Ix = int((1-2*invertX))
            Iy = int((1-2*invertY))
            im_ = np.swapaxes(im[::Ix,::Iy,...],0,1)
            im_cor_ = np.swapaxes(im_cor[::Ix,::Iy,...],0,1)#[::-1,::-1]
            im_cor_ = im_cor_.astype(np.float32)
            im_cor_ -= np.percentile(im_cor_,1)
            
            im_cor_[im_cor_<0]=0
            
            #im_ = im_[::-1,::-1]
            im_ = im_/norm
            if rotation_angle!=0:
                im_ = rotate(im_, rotation_angle, center=None, scale=1.0)
                im_cor_ = rotate(im_cor_, rotation_angle, center=None, scale=1.0)
            if max_impose:
                #im_base[x:sx+x,y:sy+y,...]=np.max([im_base[x:sx+x,y:sy+y,...],im_],axis=0)
                #im_ind = im_compare[x:sx+x,y:sy+y,...]>im_cor_
                #im_compare[x:sx+x,y:sy+y,...] = np.max([im_compare[x:sx+x,y:sy+y,...],im_cor_],axis=0)
                im_base[x:sx+x,y:sy+y,...]=(im_compare[x:sx+x,y:sy+y,...]*im_base[x:sx+x,y:sy+y,...]+im_cor_*im_)/(im_compare[x:sx+x,y:sy+y,...]+im_cor_)
                im_compare[x:sx+x,y:sy+y,...]=(im_compare[x:sx+x,y:sy+y,...]*im_compare[x:sx+x,y:sy+y,...]+im_cor_*im_cor_)/(im_compare[x:sx+x,y:sy+y,...]+im_cor_)
                
                im_compare[x:sx+x,y:sy+y,...][np.isnan(im_compare[x:sx+x,y:sy+y,...])]=0
                bad_inf = np.isnan(im_base[x:sx+x,y:sy+y,...])|np.isinf(im_base[x:sx+x,y:sy+y,...])
                im_base[x:sx+x,y:sy+y,...][bad_inf]=np.nanmedian(im_base[x:sx+x,y:sy+y,...])
                
                #im_sel[im_ind]#np.mean([im_base[x:sx+x,y:sy+y,...],im_],axis=0)
            else:
                im_base[x:sx+x,y:sy+y,...]=im_
             
            save_pars = [fls_iter[i_],x,sx+x,y,sy+y,resc]
            save_pars_all+=[save_pars]
            fid.write("\t".join(map(str,save_pars))+'\n')
        
        fid.close()
        if add_txt:
            for fl,x,xM,y,yM,resc in save_pars_all:
                txt = os.path.basename(fl).split('_')[-1].split('.')[0]
                im_base[x:xM,y:yM] = cv2.putText(im_base[x:xM,y:yM], txt, (100,100), cv2.FONT_HERSHEY_SIMPLEX ,  1, 2**16-1, 2, cv2.LINE_AA)
        tifffile.imsave(save_file,np.clip(im_base,0,2**16-1).astype(np.uint16))
def save_tile_image_and_label(fls_iter,save_file,resc=2,custom_frms=None,pix_size_=0.162,
                        max_impose=True,verbose=False,im_cor__=None,
                              rotation_angle=0,add_txt=True,norm_dapi=False,invertX=False,invertY=False):

        import tifffile
        ims,xys=[],[]
        save_pars_all=[]
        pix_size=pix_size_*resc
        fls_iter_ = fls_iter
        if verbose: fls_iter_ = tqdm(fls_iter)
        for dax in fls_iter_:
            
            xml_fl = os.path.dirname(dax)+os.sep+os.path.basename(dax).split('_fr')[0].split('.dax')[0]+'.xml'
            pos = np.array([ln.split('>')[1].split('<')[0].split(',') for ln in open(xml_fl,'r') if 'stage_position' in ln][0],dtype=np.float32)
            dapi_im = load_im(dax,custom_frms=custom_frms)
            
            dapi_im_small  = np.array(dapi_im)[:,1:-1:resc,1:-1:resc]
            
            #Illumination correction:
            dapi_im_small = np.max(dapi_im_small,0)
            if '.zst' in dax: dapi_im_small = dapi_im_small.T
            #Consider adding illumination correction and better stitching
            xys.append(pos/pix_size)
            ims.append(dapi_im_small)
        #return ims
        if im_cor__ is None:
            #im_cor = np.median(ims,axis=0)
            #ims = [im_/im_cor*np.median(im_cor) for im_ in ims]
            
            ims = np.array(ims,dtype=np.float32)
            ims_sort = np.sort(ims,0)
            delta = 10
            ielems = (np.arange(delta,100-delta,delta)*len(ims)/100).astype(int)
            ims_sort_ = ims_sort[ielems]
            vec = np.median(ims_sort_.reshape(ims_sort_.shape[0],-1),-1)
            ima = ims_sort_[0]#np.median(ims_sort_,0)
            imb = (ims_sort_[-1]-ima)/(vec[-1]-vec[0])#[:,np.newaxis,np.newaxis])[-1]*vec[-1]
            im_cor = imb
            ims = [(im_-ima)/imb+np.min(ima/imb) for im_ in ims]
        else:
            im_cor = im_cor__[1:-1:resc,1:-1:resc].copy()
            ims = [im_/im_cor*np.median(im_cor) for im_ in ims]
        xys_=np.array(xys,dtype=int)
        xys_=xys_-np.expand_dims(np.min(xys_,axis=0), axis=0)

        if len(ims[0].shape)>2:
            sx,sy,sz = ims[0].shape
            dim_base = list(np.max(xys_,axis=0)+np.array([sx+1,sy+1]))
            dim_base+=[sz]
        else:
            sx,sy = ims[0].shape
            dim_base = list(np.max(xys_,axis=0)+np.array([sx+1,sy+1]))
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        im_base = np.zeros(dim_base,dtype=float)
        im_compare = np.zeros(dim_base,dtype=float)
        infos_filename = '.'.join(save_file.split('.')[:-1])+'.infos'
        fid = open(infos_filename,'w')
        

        normns = norm_sets(ims,norm_dapi=norm_dapi,perc =80,nl = 30)
        
        
        for i_,(im,(x,y),norm) in enumerate(zip(ims[:],xys_[:],normns)):
            Ix = int((1-2*invertX))
            Iy = int((1-2*invertY))
            im_ = np.swapaxes(im[::Ix,::Iy,...],0,1)
            im_cor_ = np.swapaxes(im_cor[::Ix,::Iy,...],0,1)#[::-1,::-1]
            im_cor_ = im_cor_.astype(np.float32)
            im_cor_ -= np.percentile(im_cor_,1)
            
            im_cor_[im_cor_<0]=0
            
            #im_ = im_[::-1,::-1]
            im_ = im_/norm
            if rotation_angle!=0:
                im_ = rotate(im_, rotation_angle, center=None, scale=1.0)
                im_cor_ = rotate(im_cor_, rotation_angle, center=None, scale=1.0)
            if max_impose:
                #im_base[x:sx+x,y:sy+y,...]=np.max([im_base[x:sx+x,y:sy+y,...],im_],axis=0)
                #im_ind = im_compare[x:sx+x,y:sy+y,...]>im_cor_
                #im_compare[x:sx+x,y:sy+y,...] = np.max([im_compare[x:sx+x,y:sy+y,...],im_cor_],axis=0)
                im_base[x:sx+x,y:sy+y,...]=(im_compare[x:sx+x,y:sy+y,...]*im_base[x:sx+x,y:sy+y,...]+im_cor_*im_)/(im_compare[x:sx+x,y:sy+y,...]+im_cor_)
                im_compare[x:sx+x,y:sy+y,...]=(im_compare[x:sx+x,y:sy+y,...]*im_compare[x:sx+x,y:sy+y,...]+im_cor_*im_cor_)/(im_compare[x:sx+x,y:sy+y,...]+im_cor_)
                
                im_compare[x:sx+x,y:sy+y,...][np.isnan(im_compare[x:sx+x,y:sy+y,...])]=0
                bad_inf = np.isnan(im_base[x:sx+x,y:sy+y,...])|np.isinf(im_base[x:sx+x,y:sy+y,...])
                im_base[x:sx+x,y:sy+y,...][bad_inf]=np.nanmedian(im_base[x:sx+x,y:sy+y,...])
                
                #im_sel[im_ind]#np.mean([im_base[x:sx+x,y:sy+y,...],im_],axis=0)
            else:
                im_base[x:sx+x,y:sy+y,...]=im_
             
            save_pars = [fls_iter[i_],x,sx+x,y,sy+y,resc]
            save_pars_all+=[save_pars]
            fid.write("\t".join(map(str,save_pars))+'\n')
        
        fid.close()
        if add_txt:
            for fl,x,xM,y,yM,resc in save_pars_all:
                txt = os.path.basename(fl).split('_')[-1].split('.')[0]
                im_base[x:xM,y:yM] = cv2.putText(im_base[x:xM,y:yM], txt, (100,100), cv2.FONT_HERSHEY_SIMPLEX ,  1, 2**16-1, 2, cv2.LINE_AA)
        tifffile.imsave(save_file,np.clip(im_base,0,2**16-1).astype(np.uint16))

def linear_flat_correction(ims,fl=None,reshape=True,resample=4,vec=[0.1,0.15,0.25,0.5]):
    #correct image as (im-bM[1])/bM[0]
    #ims=np.array(ims)
    if reshape:
        ims_pix = np.reshape(ims,[ims.shape[0]*ims.shape[1],ims.shape[2],ims.shape[3]])
    else:
        ims_pix = np.array(ims[::resample])
    ims_pix_sort = np.sort(ims_pix[::resample],axis=0)
    ims_perc = np.array([ims_pix_sort[int(frac*len(ims_pix_sort))] for frac in vec])
    i1,i2=np.array(np.array(ims_perc.shape)[1:]/2,dtype=int)
    x = ims_perc[:,i1,i2]
    X = np.array([x,np.ones(len(x))]).T
    y=ims_perc
    a = np.linalg.inv(np.dot(X.T,X))
    cM = np.swapaxes(np.dot(X.T,np.swapaxes(y,0,-2)),-2,1)
    bM = np.swapaxes(np.dot(a,np.swapaxes(cM,0,-2)),-2,1)
    if fl is not None:
        folder = os.path.dirname(fl)
        if not os.path.exists(folder):
            os.makedirs(folder)
        pickle.dump(bM,open(fl,'wb'))
    return bM    
def compose_mosaic(ims,ims_c,xs_um,ys_um,fl=None,um_per_pix=0.153,rot = -2,plt_val=False,tag='',monitor=False):
    dtype = np.float32
    im_ = ims[0]
    szs = im_.shape
    sx,sy = szs[-2],szs[-1]
    theta=-np.deg2rad(rot)
    xs_um_ = np.array(xs_um)*np.cos(theta)-np.array(ys_um)*np.sin(theta)
    ys_um_ = np.array(ys_um)*np.cos(theta)+np.array(xs_um)*np.sin(theta)
    xs_pix = np.array(xs_um_)/um_per_pix
    xs_pix = np.array(xs_pix-np.min(xs_pix),dtype=int)
    ys_pix = np.array(ys_um_)/um_per_pix
    ys_pix = np.array(ys_pix-np.min(ys_pix),dtype=int)
    sx_big = np.max(xs_pix)+sx+1
    sy_big = np.max(ys_pix)+sy+1
    dim = [sx_big,sy_big]
    if len(szs)==3:
        dim = [szs[0],sx_big,sy_big]
    if fl is None:
        im_big = np.zeros(dim,dtype = dtype)
    else:
        import h5py
        f = h5py.File(fl, "w")
        f.create_dataset("mosaic", dim, dtype=dtype)
        im_big = f["mosaic"]

    for i,(im_,x_,y_) in enumerate(zip(ims,xs_pix,ys_pix)):
        if monitor:
            print(str(i)+'/'+str(len(ims)))
        if ims_c is not None:
            if len(ims_c)==2:
                im_coef,im_inters = np.array(ims_c,dtype = 'float32')
                im__=(np.array(im_,dtype = 'float32')-im_inters)/im_coef
            else:
                ims_c_ = np.array(ims_c,dtype = 'float32')
                im__=np.array(im_,dtype = 'float32')/ims_c_*np.median(ims_c_)
        else:
            im__=np.array(im_,dtype = 'float32')
        if tag=='dax':
            im__ = im__[...,::1,::-1]
        if tag=='merfish3':
            im__ = np.swapaxes(im__[...,::-1,::1],-2,-1)
        #im__[im__>(2**16-1)]=2**16-1
        im__ = np.array(im__,dtype = dtype)
        im_big[...,x_:x_+sx,y_:y_+sy]=im__
        #im_big[x_:x_+sx,y_:y_+sy]=im__
    if plt_val:
        plt.figure()
        plt.imshow(im_big[::,::],cmap='gray',interpolation='nearest')
        plt.show()
    if fl is not None:
        f.close()
        return None
    return im_big
    
class Mosaic(object):
    def load(self,folder,nmax = None):
        filenames = np.array(glob.glob(folder+os.sep+'*.stv'))
        ind_ = np.argsort([int(fl.split('_')[-1].split('.')[0]) for fl in filenames])
        if nmax is None:
            nmax=len(filenames)
        filenames=filenames[ind_][:nmax]

        print("Started loading images")
        try:
            self.dics=[pickle.load(open(filename,'r')) for filename in filenames]
        except:
            self.dics=[pickle.load(open(filename,'rb')) for filename in filenames]
        print("Finished loading images")
        dics = self.dics
        self.obj = dics[0]['objective_name']
        self.pos_x = np.array([dic['x_um'] for dic in dics],dtype=float)
        self.pos_y = np.array([dic['y_um'] for dic in dics],dtype=float)
        self.ims = [dic["data"] for dic in self.dics]
    def get_correction(self,ims):
        return np.median(ims,0)
    def get_ims(self):
        self.ims = [dic["data"] for dic in self.dics]
    def get_mosaic(self,ims=None,xs_um=None,ys_um=None,rot=-2,um_per_pixel_dic={'10x':0.93,'60x':0.153,'20x':0.46},med=False):
        um_per_pix = um_per_pixel_dic[self.obj]
        if ims is None:
            self.get_ims()
            ims = self.ims
            xs_um,ys_um=self.pos_x,self.pos_y
        if med:
            ims_c = np.median(ims,0)
        else:
            #ims_c = linear_flat_correction(ims,fl=None,reshape=False,resample=1,vec=[0.1,0.15,0.25,0.5])
            ims_c = linear_flat_correction(ims,fl=None,reshape=False,resample=1,vec=[0.1,0.15,0.25,0.5,0.65,0.75,0.9])
        self.im = compose_mosaic(ims,ims_c,xs_um,ys_um,um_per_pix=um_per_pix,rot = rot,plt_val=False,tag='steve')
    def save(self,filename,im_min=None,im_max=None):
        im=np.array(self.im,dtype=float)
        if im_min==None:
            im_min=np.min(im)
        if im_max==None:
            im_max=np.max(im)
        im=(im-im_min)/(im_max-im_min)
        im[im<0]=0
        im[im>1]=1
        result = Image.fromarray((im * 255).astype(np.uint8))
        result.save(filename)
    def mask(self,filename,sz=[22,22],plot_val=False):
        self.sz=sz
        image = Image.open(filename)
        xsz,ysz=image.size
        image_small=image.resize([xsz/sz[0],ysz/sz[1]],Image.NEAREST)
        print("Number of snaps:" + str(np.sum(np.array(image_small)>0)))#*3./60/60
        x,y=np.where(np.array(image_small)>0)
        cities = frozenset(Point(x[i],y[i]) for i in range(len(x)))
        cities_sort=alter_tour(greedy_tsp(cities))
        x2=[val.real for val in cities_sort]
        y2=[val.imag for val in cities_sort]
        if plot_val:
            plt.figure(figsize=(20,20))
            plt.imshow(np.array(image_small)>0)
            plt.plot(y2,x2)
            plt.plot(y2,x2,'.')
    def savePos(self,filename,pix_sz=0.16):
        x = self.starts[0]+(np.array(self.x))/pix_sz*self.sz[0]
        y = self.starts[1]+(np.array(self.y))/pix_sz*self.sz[1]
        fid=open(filename,'w')
        for i in range(len(x)):
            fid.write(str(x[i])+', '+str(y[i])+'\r\n')
        fid.close()

##TSP
# Cities are represented as Points, which are represented as complex numbers
Point = complex
def X(point): 
    "The x coordinate of a point."
    return point.real
def Y(point): 
    "The y coordinate of a point."
    return point.imag

def distance(A, B): 
    "The distance between two points."
    return abs(A - B)
def nn_tsp(cities):
    """Start the tour at the first city; at each step extend the tour 
    by moving from the previous city to its nearest neighbor 
    that has not yet been visited."""
    start = first(cities)
    tour = [start]
    unvisited = set(cities - {start})
    while unvisited:
        C = nearest_neighbor(tour[-1], unvisited)
        tour.append(C)
        unvisited.remove(C)
    return tour
def first(collection):
    "Start iterating over collection, and return the first element."
    return next(iter(collection))
def nearest_neighbor(A, cities):
    "Find the city in cities that is nearest to city A."
    return min(cities, key=lambda c: distance(c, A))
def greedy_tsp(cities):
    """Go through edges, shortest first. Use edge to join segments if possible."""
    edges = shortest_edges_first(cities) # A list of (A, B) pairs
    endpoints = {c: [c] for c in cities} # A dict of {endpoint: segment}
    for (A, B) in edges:
        if A in endpoints and B in endpoints and endpoints[A] != endpoints[B]:
            new_segment = join_endpoints(endpoints, A, B)
            if len(new_segment) == len(cities):
                return new_segment
def shortest_edges_first(cities):
    "Return all edges between distinct cities, sorted shortest first."
    edges = [(A, B) for A in cities for B in cities 
                    if id(A) < id(B)]
    return sorted(edges, key=lambda edge: distance(*edge))
def join_endpoints(endpoints, A, B):
    "Join B's segment onto the end of A's and return the segment. Maintain endpoints dict."
    Asegment, Bsegment = endpoints[A], endpoints[B]
    if Asegment[-1] is not A: Asegment.reverse()
    if Bsegment[0] is not B: Bsegment.reverse()
    Asegment.extend(Bsegment)
    del endpoints[A], endpoints[B]
    endpoints[Asegment[0]] = endpoints[Asegment[-1]] = Asegment
    return Asegment

def alter_tour(tour):
    "Try to alter tour for the better by reversing segments."
    original_length = tour_length(tour)
    for (start, end) in all_segments(len(tour)):
        reverse_segment_if_better(tour, start, end)
    # If we made an improvement, then try again; else stop and return tour.
    if tour_length(tour) < original_length:
        return alter_tour(tour)
    return tour

def all_segments(N):
    "Return (start, end) pairs of indexes that form segments of tour of length N."
    return [(start, start + length)
            for length in range(N, 2-1, -1)
            for start in range(N - length + 1)]
def tour_length(tour):
    "The total of distances between each pair of consecutive cities in the tour."
    return sum(distance(tour[i], tour[i-1]) 
               for i in range(len(tour)))
def reverse_segment_if_better(tour, i, j):
    "If reversing tour[i:j] would make the tour shorter, then do it." 
    # Given tour [...A-B...C-D...], consider reversing B...C to get [...A-C...B-D...]
    A, B, C, D = tour[i-1], tour[i], tour[j-1], tour[j % len(tour)]
    # Are old edges (AB + CD) longer than new ones (AC + BD)? If so, reverse segment.
    if distance(A, B) + distance(C, D) > distance(A, C) + distance(B, D):
        tour[i:j] = reversed(tour[i:j])
##

def get_tsp(xy,num_iter=500,first_pos=-1,plt_val=True):
    cities = frozenset(Point(x_,y_) for x_,y_ in xy)
    #be greedy first
    cities_sort=greedy_tsp(cities)
    #alter
    for i in range(num_iter):
        cities_sort=alter_tour(cities_sort)
    x=[val.real for val in cities_sort]
    y=[val.imag for val in cities_sort]
    itinerary = np.array([x,y]).T
    # new file to save

    last_pos = np.where(np.all(np.array(itinerary)==[xy[-1]],-1))[0][0]
    itinerary = np.roll(itinerary,-last_pos,axis=0)
    if plt_val:
        plt.plot(itinerary[:,0], itinerary[:,1],'o-')
        plt.show()
    return itinerary