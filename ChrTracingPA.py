import numpy as np
import glob,os,sys
import cv2
import matplotlib.pyplot as plt
import tifffile

from tqdm.notebook import tqdm
import pickle
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
def make_trace_unique(XT,RT,hT,th_gd=100):
    from scipy.spatial.distance import cdist
    d = cdist(XT,XT)
    gd = cdist(RT[:,np.newaxis],RT[:,np.newaxis])
    W = w(d,gd)
    W[gd==0]=np.nan
    W[gd>th_gd]=np.nan
    WT = np.nanmean(W,axis=-1)## this is the average weight considering the nodes with genomic distance <th_gd
    uRs = np.unique(RT)
    WT[np.isnan(WT)]=0
    inds = [np.where(RT==R_)[0][np.argmax(WT[RT==R_])] for R_ in uRs]
    return XT[inds],RT[inds],hT[inds]
def plot_trace(X,XT,RT,hT,size_pt = 0.25,min_sz=0,viewer=None,name=None):
    import napari
    from matplotlib import cm as cmap
    size=hT*size_pt+min_sz
    addbase = False
    if viewer is None:
        addbase = True
        viewer = napari.Viewer()
    cols = cmap.rainbow(RT/np.max(RT))
    txt = list((RT-3).astype(str))
    text = {
        'string': txt,
        'size': 10,
        'color': 'white'}
    if addbase:
        viewer.add_points(X*[0.5,1,1],size=0.05,face_color=[1,1,1],edge_color=[0,0,0,0],edge_width=0)#,text=text)
    viewer.add_points(XT*[0.5,1,1],size=size,face_color=cols,edge_color=[0,0,0,0],edge_width=0,name=name)#,text=text)
    return viewer



    
    
def get_trace(X,R,h,seed=None,visited_iX=None,traces =[],ddist = 0.75,dgen=20,plt_val=False,exclude=True,tqdm_=None):

    if visited_iX is None:
        visited_iX = np.unique([iX for tr in traces for iX in tr])
    
    keepf = np.setdiff1d(np.arange(len(X)),visited_iX)

    iXtemp = [np.argmax(h[keepf])]    
    if not exclude:
        iXtemp = [keepf[np.argmax(h[keepf])]]
        keepf = np.arange(len(X))
    if seed is not None:
        iXtemp = list(seed)
    
    iXtemp0 = list(iXtemp)
    X_ = X[keepf]
    R_ = R[keepf]
    tree = KDTree(X_)
    Xtemp = X_[iXtemp]
    Rtemp = R_[iXtemp]
    while True:
        res = tree.query_ball_point(Xtemp,ddist)
        iXs = np.unique([r for rs in res for r in rs]+iXtemp)
        
        Wd = get_weight(X_[iXs],Xtemp,R_[iXs],Rtemp,th_gd=dgen)
        good= ~np.isnan(Wd)
        iXs = iXs[good]
        Rcand = R_[iXs]
        Wd = Wd[good]
        dic_best={R_[iXtemp0[0]]:[iXtemp0[0],np.inf]}
        for iX__,R__,W__ in zip(iXs,Rcand,Wd):
            if R__ not in dic_best: 
                dic_best[R__]=[iX__,W__]
            else: 
                #if dic_best[R__][0]!=iXtemp0[0]:
                if W__>dic_best[R__][-1]: 
                    dic_best[R__]=[iX__,W__]
        iX_keep = [dic_best[R__][0] for R__ in dic_best]
        iXtemp_prev = np.unique(iXtemp)
        
        if len(iXtemp_prev)==len(iX_keep):
            break
        iXtemp = list(iX_keep)
        Xtemp = X_[iXtemp]
        Rtemp = R_[iXtemp]
        #print(len(iXtemp))
    
    trace = keepf[iXtemp]
    if tqdm_ is not None:
        tqdm_.update(len(visited_iX)-tqdm_.n)
    if plt_val:
        plt.figure()
        plt.plot(X[:,1],X[:,2],'o',alpha=0.01)
        plt.plot(X[trace,1],Xtemp[trace,2],'o',alpha=0.1)
        plt.axis('equal')
        plt.show()
    return trace

def get_initial_traces(X,R,h,ddist=0.5,dgen=50):
    tqdm_ = tqdm(total=len(X))
    traces = []
    while True:
        trace = get_trace(X,R,h,traces=traces,ddist = ddist,dgen=dgen,plt_val=False,
                          exclude=True,tqdm_=tqdm_)
        traces+=[trace]
        used_iX = np.unique([iX for tr in traces for iX in tr])

        if len(used_iX)==len(X):
            break
    return traces
def enforce_unique(keep_traces_,X,R,th_dg=50):
    """Given a list of traces (index in X/R) and the points X and genomic indexes R 
    this ensures that each point is only in one trace"""
    uiX = np.unique([iX for tr in keep_traces_ for iX in tr])
    isInTrace = np.array([np.in1d(uiX,tr) for tr in keep_traces_])
    badiX = np.where(np.sum(isInTrace,0)>1)[0]
    #print(len(badiX))
    deg_traces = [np.where(isInTrace[:,iX_])[0] for iX_ in badiX]
    iXbad = uiX[badiX]
    keep_traces__ = [np.array(tr) for tr in keep_traces_]
    for iX,itrs in zip(iXbad,deg_traces):
        scores = np.array([get_weight(X[[iX]],X[keep_traces__[itr]],R[[iX]],R[keep_traces__[itr]],th_gd=th_dg)[0] for itr in itrs])
        scores[np.isnan(scores)]=-np.inf
        btr = np.argmax(scores)
        for itr in itrs:
            if itr!=itrs[btr]:
                keep_traces__[itr] = np.setdiff1d(keep_traces__[itr],[iX])
    return keep_traces__    
def get_fisher(SC_calc,SC,exclude_inf=True):
    if exclude_inf:
        SC_ = np.sort(SC[~(np.isinf(SC)|np.isnan(SC))])[:,np.newaxis]
    else:
        SC_ = np.sort(SC[~np.isnan(SC)])[:,np.newaxis]
    if len(SC_)==0:
        return -np.inf
    _,ipts = KDTree(SC_).query(SC_calc[:,np.newaxis])
    return np.log(ipts+1)-np.log(len(SC_))
def get_weight(X,XT,R,RT,th_gd=20,dth = 1,return_good=False):
    from scipy.spatial.distance import cdist
    d = cdist(X,XT)
    
    gd = cdist(R[:,np.newaxis],RT[:,np.newaxis])
    W = w(d,gd)
    W[gd==0]=np.nan
    W[gd>th_gd]=np.nan
    WT = np.nanmean(W,axis=-1)## this is the average weight considering the nodes with genomic distance <th_gd
    is_good = np.any((d<dth)&(gd<th_gd)&(gd>0),axis=-1)
    if return_good:
        return WT,is_good
    return WT
def refine_trace(X,R,h,XT,RT,hT,th_gd=100,per_keep=5,dth = 0.3,WThT=None,use_brightness=False):
    WT = get_weight(XT,XT,RT,RT,th_gd=th_gd,dth = dth)
    W,is_good = get_weight(X,XT,R,RT,th_gd=th_gd,dth = dth,return_good=True)
    if use_brightness:
        if WThT is None:
            WThT = WT,hT
        WTs,hTs = WThT
        SC = np.array([get_fisher(W,WTs),get_fisher(h,hTs)]).T
        SCT = np.array([get_fisher(WT,WTs),get_fisher(hT,hTs)]).T#get_fisher(WT,WT)+get_fisher(hT,hT)
        SCTs = np.array([get_fisher(WTs,WTs),get_fisher(hTs,hTs)]).T
    else:
        if WThT is None:
            WThT = WT
        WTs = WThT
        SC = W[:,np.newaxis]
        SCT = WT[:,np.newaxis]
        SCTs = WTs[:,np.newaxis]
    min_ = np.percentile(SCTs,per_keep,axis=0)
    uRs = np.unique(R)
    inds = []
    for R_ in uRs:
        ind_ = np.where(R==R_)[0]
        SC_ = SC[ind_]
        is_good_ = is_good[ind_]
        SC_[np.isnan(SC_)]=-np.inf
        SC_[~is_good_]=-np.inf
        imax_ = np.argmax(np.sum(SC_,axis=-1))
        if np.all(SC_[imax_]>min_) and is_good_[imax_]:
            inds.append(ind_[imax_])

    return X[inds],R[inds],h[inds],WThT
def w(x,gd=1,s1=0.085,normed=True): 
    sigmasq = 0.025**2
    k = (s1*s1-2*sigmasq)/1
    ssq = 2*sigmasq+k*gd
    xsq = x*x
    w_= 4*np.pi*xsq/(2*np.pi*ssq)**(3/2)*np.exp(-xsq/2/ssq)
    return w_
def get_rough_traces(Xs_D,Rs_,hs_,cells_,icell=0,dth=1.25,th_fr = 0.4,gdmax=20000):
    
    X = Xs_D[cells_==icell]
    R = Rs_[cells_==icell]
    h = hs_[cells_==icell]
    #napari.view_points(X,size=0.1)
    tree = KDTree(X)
    res = tree.query_ball_tree(tree,dth)
    Ws = []
    edges = []
    for iR,rs in enumerate(res):
        edges_ = np.array([(iR,r_) for r_ in rs])
        d = np.linalg.norm(X[rs]-X[iR],axis=-1)
        gd = np.abs(R[rs]-R[iR])
        keep = (gd>=1)&(gd<=gdmax)
        Ws.extend(w(d,gd)[keep])
        edges.extend(edges_[keep])
    Ws = np.array(Ws)
    edges= np.array(edges)


    neighbours = {iX:[iX] for iX in range(len(X))}
    order = np.argsort(Ws)[::-1]
    for iX1,iX2 in tqdm(edges[order]):
        neigh_ = neighbours[iX1]+ neighbours[iX2]
        iRs_,cts_ = np.unique([R[iX] for iX in neigh_],return_counts=True)
        if np.percentile(cts_,50)==1:
            for iX in neigh_:
                neighbours[iX]=neigh_

    traces = {}            
    max_trace=0
    for iR in neighbours:
        if iR not in traces:
            max_trace+=1
            traces[iR]=max_trace
        for iRR in neighbours[iR]:
            traces[iRR]=traces[iR]


    trcs = np.array([traces[iX] for iX in range(len(traces))])
    utrcs,cts_ = np.unique(trcs,return_counts=True)
    utrcs = utrcs[np.argsort(cts_)[::-1]]
    uRs = np.unique(Rs_)
    frs = np.array([len(np.unique(R[trcs==utr]))/len(uRs) for utr in utrcs])
    
    
    
    plt.plot(frs,'o-')
    plt.plot([0, len(frs)],[th_fr,th_fr],'k-')
    plt.xlabel('Traces')
    plt.ylabel("Detection efficiecy")
    keep = frs>th_fr
    keep_tr = utrcs[keep]
    print("Detection efficiency:",np.median(frs[keep]))
    print("Detected traces:",np.sum(keep))
    return trcs,keep_tr
def get_points_cell(self,icell_=0,bad_points=50,std_th=0.15,pix = [0.25,0.10833,0.10833]):
    #self = chr_
    dic_drifts,dic_pts_cells = self.dic_drifts,self.dic_pts_cells
    fls_fov = self.fls_fov  
    #if volume_th is not None:
    self.cells =  get_cells(dic_pts_cells,volume_th =0)
    cells = self.cells 
    icell = cells[icell_]
    Xs,hs,Rs,icols = get_X(dic_drifts,dic_pts_cells,fls_fov,icell,pix=pix,nelems = 10000)


    ### Get the average maximum brightness per color
    dic_h = {}
    for icol in np.unique(icols):
        keep = (icols==icol)
        Rkeep = np.unique(Rs[keep])
        dic_h[icol] = np.median([np.max(hs[Rs==R,0])for R in Rkeep])
    dic_Rcol = {R:icol for R,icol in zip(Rs,icols)}
    self.dic_Rcol = dic_Rcol
    ### normalize brigtheses
    hsnorm = hs[:,0].copy()
    bknorm = hs[:,1].copy()
    is_good = []
    for iR in np.unique(Rs):

        hsnorm[Rs==iR] = hsnorm[Rs==iR]/dic_h[dic_Rcol[iR]]
        bknorm[Rs==iR] = bknorm[Rs==iR]/dic_h[dic_Rcol[iR]]
        hs_ = hsnorm[Rs==iR]
        min_ = np.median(np.sort(hs_)[:bad_points])
        #std_ = np.std(np.sort(hs_)[:50])*20
        std_ = std_th
        keep = hs_>(min_+std_)
        is_good.extend(keep)

    ### find minimum
    #min_ = np.median([h_ for iR in np.unique(Rs) for h_ in np.sort(hsnorm[Rs==iR])[:50]])
    is_good = np.array(is_good)
    hsnorm_ = np.clip(hsnorm-min_,0,1)


    Xs_ = Xs[is_good]
    Rs_ = Rs[is_good]
    hs_ = hsnorm[is_good]
    iRs = np.unique(Rs_)
    return Xs,Rs,hsnorm,Xs_,Rs_,hs_
    
def compute_hybe_drift(Xs_CC,Hybe_,ncompare = 20,th_dist=0.5,npoint=10):
    """
    Given points Xs_CC and indexes Hybe_ this compares points in hybe and hybe+i (with all i<ncompare) and 
    based on the nearest neighbors < th_dist will computed the consensus drift.
    Apply as:
    icols_ = (Rs_-1)%3
    Hybe_ = (Rs_-1)//3
    drift_hybe = compute_hybe_drift(Xs_CC,Hybe_,ncompare = 20)
    Xs_D = Xs_CC.copy()
    Xs_D-= np.array([drift_hybe[hybe] for hybe in Hybe_])
    """
    Hybes = np.unique(Hybe_)
    nH = len(Hybes)
    #for i in range(len())
    a = [np.zeros(nH)]
    a[0][nH//2]=1
    b = [[0,0,0]]
    count=1
    for iH in range(nH):
        for jH in range(iH):
            if np.abs(iH-jH)<=ncompare:
                X1 = Xs_CC[Hybe_==iH]#[1::2]
                X2 = Xs_CC[Hybe_==jH]#[1::2]

                tree = KDTree(X1)
                dists,inds = tree.query(X2)
                keep = dists<th_dist
                X1_= X1[inds[keep]]
                X2_= X2[keep]


                if len(X1_)>npoint:
                    b_ = np.median((X1_-X2_),axis=0)
                    b.append(b_)
                    arow = np.zeros(nH)
                    arow[iH],arow[jH]=1,-1
                    a.append(arow)
                    count+=1
    a=np.array(a)
    b=np.array(b)
    res = np.linalg.lstsq(a,b)[0]
    drift_hybe = {hybe:res[iH]for iH,hybe in enumerate(Hybes)}
    return drift_hybe
def get_NN_distances(Xs_,Rs_,deltaR=1,th_dist=1):
    """
    Given a set of points <Xs_> and indixes of rounds <Rs_> this returns the distances of nearest neighbors from R and R+deltaR.
    It only keeps distances smaller than th_dist.
    """
    
    all_dists = []
    iRs = np.unique(Rs_)
    for iR in iRs[:]:
        iR1,iR2=iR,iR+deltaR
        X1 = Xs_[Rs_==iR1]
        X2 = Xs_[Rs_==iR2]
        #if dic_Rcol.get(iR,0)==dic_Rcol.get(iR+1,0):
        if True:#((iR1-1)//3)==((iR2-1)//3):
            tree = KDTree(X1)
            dists,inds = tree.query(X2)
            all_dists.extend(dists[dists<th_dist])
    return all_dists
def compute_color_drift_per_cell(Xs_,Rs_,cells_,dic_Rcol,th_dist=0.5):
    """
    For each cell this looks for nearest neighbours across consecutive genomic regions
    It saves a dictionary dic_col_drift[(cell,color)] with the drift that needs to be added to the positions.
    Apply as:
    
    Xs_CC = Xs_.copy()
    XDC = np.array([dic_col_drift[(cell,dic_Rcol[iR])]for cell,iR in zip(cells_,Rs_)])
    Xs_CC+=XDC
    
    """
    dic_col_drift = {}
    for cell in np.unique(cells_):
        dic_pair = {}
        keep = cells_==cell
        Xs_T = Xs_[keep]
        Rs_T = Rs_[keep]
        for iR in np.unique(Rs_T):
            X1 = Xs_T[Rs_T==iR]
            X2 = Xs_T[Rs_T==(iR+1)]
            key = (dic_Rcol.get(iR,0),dic_Rcol.get(iR+1,0))
            tree = KDTree(X1)
            dists,inds = tree.query(X2)
            keep = dists<th_dist
            X1_= X1[inds[keep]]
            X2_= X2[keep]
            if key not in dic_pair: dic_pair[key]=[]
            dic_pair[key] += [np.median(X1_-X2_,axis=0)]

        dic_col_drift[(cell,0)]=np.array([0,0,0])
        dic_col_drift[(cell,1)]=np.nanmedian(dic_pair[(0,1)],axis=0)
        dic_col_drift[(cell,2)]=np.nanmedian(dic_pair[(0,1)],axis=0)+np.nanmedian(dic_pair[(1,2)],axis=0)
        #dic_col_drift,-np.nanmedian(dic_pair[(2,0)],axis=0)
    return dic_col_drift
def get_X(dic_drifts,dic_pts_cells,fls_fov,icell,pix=[0.200,0.1083,0.1083],nelems = 4,dic_zdist=None):
    ncol = len(dic_pts_cells[0])
    Xs,Rs,hs,icols = [],[],[],[]
    R_ = 0
    H_ = 0 
    for dic_drift,dic_pts_cell in zip(dic_drifts,dic_pts_cells):
        for icol in range(ncol):
            R_+=1
            if icell in dic_pts_cell[icol]:
                fits = dic_pts_cell[icol][icell]['fits']
            else:
                fits = []
            if len(fits)>0:
                X_ = fits[:nelems,1:4]
                h_ = fits[:nelems,[0,4]]
                if dic_zdist is not None:
                    zs,zf = dic_zdist[H_]
                    X_[:,0] = np.interp(X_[:,0],zs,zf)
                X_ = X_-dic_drift['Ds'][0]#drift correct
                X_ = X_*pix #convert to um
                Xs.extend(X_)
                hs.extend(h_)
                Rs.extend([R_]*len(X_))
                icols.extend([icol]*len(X_))
        H_+=1
    return np.array(Xs),np.array(hs),np.array(Rs),np.array(icols)
def get_cells(dic_pts_cells,volume_th =200000):
    cells = list(dic_pts_cells[0][0].keys())
    return np.array([icell for icell in cells if dic_pts_cells[0][0][icell]['volume']>volume_th])
def determine_number_of_chromosomes(Xs,hs,Rs,radius_chr = 1.25,enhanced_radius=1.25,nchr_=5,fr_th = 0.5,plt_val=True):
    from scipy.spatial.distance import pdist,squareform
    nRs = len(np.unique(Rs))
    mat = squareform(pdist(Xs)) #distance matrix
    mat_connection = np.exp(-mat**2/2/(radius_chr/3)**2) #gaussian connectivity matrix

    keep_index = np.arange(len(mat))
    ibests = []
    for iiter in range(nchr_):
        mat_connection_ = mat_connection[keep_index][:,keep_index]
        mat_ = mat[keep_index][:,keep_index]
        ibest = np.argmax([np.mean(row) for row in mat_connection_])
        ibests.append(keep_index[ibest])
        keep_index = keep_index[np.where(mat_[ibest]>radius_chr)[0]]

    ibests = np.array(ibests)
    #print(np.sum(mat[ifirst]<radius_chr),np.sum(mat[isecond]<radius_chr))
    cls = np.argmin(mat[ibests,:],0)
    nfs = []
    for icl,ibest in enumerate(ibests):
        keep_elems = np.where((mat[ibest]<radius_chr*1.25)&(cls==icl))[0] #which elements are closest to center chr and closeset tot that
        Rs_keep = np.unique(Rs[keep_elems])## regions we are covering
        #print(Rs_keep)
        nfr_ = len(Rs_keep)/nRs
        nfs.append(nfr_)
        if plt_val: print(nfr_)
    nfs = np.array(nfs)
    ibests = ibests[nfs>fr_th]
    if plt_val: 
        if len(ibests)>0:
            cls = np.argmin(mat[ibests,:],0)
            plt.figure()
            #plt.title("Cell "+str(icell))
            iz = 1
            plt.plot(Xs[:,iz],Xs[:,2],'.',color='gray')
            print("Final_ths:")
            for ic in range(len(ibests)):
                ibest = ibests[ic]
                keep = (cls==ic)&(mat[ibests[ic]]<radius_chr*1.25)
                Rs_keep = np.unique(Rs[keep])
                nfr_ = len(Rs_keep)/nRs
                print(nfr_)
                plt.plot(Xs[keep,iz],Xs[keep,2],'o',alpha=1)
                plt.plot(Xs[ibest,iz],Xs[ibest,2],'ko')
            #plt.plot(Xs[isecond,1],Xs[isecond,2],'ko')
            plt.axis('equal')
            plt.show()
    return Xs[ibests]


### EM functions

### Usefull functions

def nan_moving_average(a,n=3):
    a_ = np.array(a)
    if n>0: a_ = np.concatenate([a[-n:],a,a[:n]])
    ret = np.nancumsum(a_,axis=0, dtype=float)
    ret_nan = ~np.isnan(a_)
    ret_nan = np.cumsum(ret_nan,axis=0, dtype=float)
    n_=2*n+1
    ret[n_:] = ret[n_:] - ret[:-n_]
    ret_nan[n_:] = ret_nan[n_:] - ret_nan[:-n_]
    ret_ = ret[n_ - 1:] / ret_nan[n_ - 1:]
    return ret_
def moving_average(a,n=3):
    a_ = np.array(a)
    if n>0: a_ = np.concatenate([a[-n:],a,a[:n]])
    ret = np.cumsum(a_,axis=0, dtype=float)
    n_=2*n+1
    ret[n_:] = ret[n_:] - ret[:-n_]
    ret_ = ret[n_ - 1:] / n_
    return ret_
def cum_val(vals,target):
    """returns the fraction of elements with value < taget. assumes vals is sorted"""
    niter_max = 10
    niter = 0
    m,M = 0,len(vals)-1
    while True:
        mid = int((m+M)/2)
        if vals[mid]<target:
            m = mid
        else:
            M = mid
        niter+=1
        if (M-m)<2:
            break
    return mid/float(len(vals))
def flatten(l):
    return [item for sublist in l for item in sublist]

def get_Ddists_Dhs(zxys_f,hs_f,nint=5):
    h = np.ravel(hs_f)#[np.ravel(cols_f)=='750']
    h = h[(~np.isnan(h))&(~np.isinf(h))&(h>0)]
    h = np.sort(h)
    dists = []
    distsC = []
    for zxys_T in zxys_f:
        difs = zxys_T-nan_moving_average(zxys_T,nint)#np.nanmedian(zxys_T,0)
        difsC = zxys_T-np.nanmedian(zxys_T,axis=0)
        dists.extend(np.linalg.norm(difs,axis=-1))
        distsC.extend(np.linalg.norm(difsC,axis=-1))
    dists = np.array(dists)
    dists = dists[(~np.isnan(dists))&(dists!=0)]
    dists = np.sort(dists)
    
    distsC = np.array(distsC)
    distsC = distsC[(~np.isnan(distsC))&(distsC!=0)]
    distsC = np.sort(distsC)
    return h,dists,distsC
def get_maxh_estimate(pfits_cands_,Rs_u = np.arange(175)+1):
    """
    Assumes pfits_cands_ is of the form Nx5 where 1:3 - z,x,y 4-h and 5-R
    """
    zxys_T = []
    hs_T=[]
    hs_bk_T=[]
    if len(pfits_cands_)>0:
        Rs = pfits_cands_[:,-1]
        for R_ in Rs_u:
            pfits = pfits_cands_[Rs==R_]
            if len(pfits)==0:
                zxys_T.append([np.nan]*3)
                hs_T.append(np.nan)
                hs_bk_T.append(np.nan)
                continue
            hs = pfits[:,3]
            hs_bk = pfits[:,4]
            
            
            
            zxys = pfits[:,:3]
            imax = np.argmax(hs)
            hs_T.append(hs[imax])
            hs_bk_T.append(hs_bk[imax])
            zxys_T.append(zxys[imax])
    return zxys_T,hs_T,hs_bk_T

def get_statistical_estimate(pfits_cands_,Dhs,Ddists,DdistsC,zxys_T=None,nint=5,use_local=True,use_center=True,
                             Rs_u = np.arange(175)+1):
    if zxys_T is None:
        zxys_T,hs_T,hs_bk_T = get_maxh_estimate(pfits_cands_,Rs_u=Rs_u)
    zxys_mv = nan_moving_average(zxys_T,nint)
    zxysC = np.nanmean(zxys_T,axis=0)
    zxys_T_ = []
    hs_T=[]
    scores_T = []
    all_scores=[]
    for R_ in Rs_u:#range(len(pfits_cands_)):
        Rs = pfits_cands_[:,-1]
        pfits = pfits_cands_[Rs==R_]
        if len(pfits)==0:
            zxys_T_.append([np.nan]*3)
            hs_T.append(np.nan)
            scores_T.append(np.nan)
            continue
        hs = pfits[:,3]
        zxys_ = pfits[:,:3]
        u_i = R_-1
        dists = np.linalg.norm(zxys_-zxys_mv[u_i],axis=-1)
        distsC = np.linalg.norm(zxys_-zxysC,axis=-1)
        if use_local and use_center:
            scores = [(1-cum_val(DdistsC,dC_))*(1-cum_val(Ddists,d_))*(cum_val(Dhs,h_)) for dC_,d_,h_ in zip(distsC,dists,hs)]
        if not use_local and use_center:
            scores = [(1-cum_val(DdistsC,dC_))*(cum_val(Dhs,h_)) for dC_,d_,h_ in zip(distsC,dists,hs)]
        if use_local and not use_center:
            scores = [(1-cum_val(Ddists,d_))*(cum_val(Dhs,h_)) for dC_,d_,h_ in zip(distsC,dists,hs)]
        if not use_local and not use_center:
            scores = [cum_val(Dhs,h_) for dC_,d_,h_ in zip(distsC,dists,hs)]
        iscore = np.argmax(scores)
        all_scores.append(scores)
        scores_T.append(scores[iscore])
        zxys_T_.append(zxys_[iscore])
        hs_T.append(hs[iscore])
    zxys_T_ = np.array(zxys_T_)
    hs_T =np.array(hs_T)
    return zxys_T_,hs_T,scores_T,all_scores
def get_fov(fl):
        return int(os.path.basename(fl).split('--')[0].split('_')[-1])
        
def get_hybe(fl):
    try:
        return int(os.path.basename(fl).split('--')[1].split('R')[0].split('_')[0][1:])
    except:
        return -1
def get_last_readout(fl,mistake=True):
    try:
        return int(os.path.basename(fl).split('--')[1].split('R')[1].split('_')[0].split(',')[0])
    except:
        return -1
def unique_fl_set(fl_set):
    """If given a file set fl_set this returns a unique ordered fl_set keeping the highest hybe"""
    dic_reorder = {}
    for fl in fl_set:
        hi = get_hybe(fl)
        ri = get_last_readout(fl)
        if ri>-1:
            if ri not in dic_reorder:
                dic_reorder[ri] = (hi,fl)
            else:
                if dic_reorder[ri][0]<hi:
                    dic_reorder[ri] = (hi,fl)
    ris =  np.sort(list(dic_reorder.keys()))
    return [dic_reorder[ri][-1]for ri in ris]


class chromatin_postfits():
    def __init__(self,save_folder=r'\\BBFISH1\Raw_data_1\Glass_MERFISH\CGBB_1_25_2022_Analysis_v4',nHs=None):
        self.save_folder = save_folder
        self.fls_dics = glob.glob(save_folder+os.sep+'*H*R*-dic_pts_cell.pkl')
        fls_dics = np.array(self.fls_dics)
        fovs_ = np.array([get_fov(fl) for fl in fls_dics])
        fovs,ncts = np.unique(fovs_,return_counts=True)
        dic_fls = {}
        for fov in fovs:
            dic_fls[fov]=fls_dics[fovs_==fov]
            
        self.dic_fls =  {elem:unique_fl_set(dic_fls[elem]) for elem in dic_fls}
        #self.dic_fls = dic_fls####
        
        #self.nHs = np.max([len(dic_fls[fov]) for fov in  dic_fls])
        if nHs is None:
            self.nHs = np.max([len(dic_fls[fov]) for fov in  dic_fls])
        else:
            self.nHs = nHs
        self.completed_fovs = [fov for fov in  dic_fls if len(dic_fls[fov])>=self.nHs]
        
        
        print("Detected fovs:",len(fovs),list(fovs))
        print("Detected complete fovs:",len(self.completed_fovs),self.completed_fovs )
        print("Detected number of hybes:",list(np.unique([len(self.dic_fls[ifov]) for ifov in self.dic_fls])))
        
    def load_fov(self,ifov,volume_th=200000):
        self.fov = ifov
        fls_dics = self.dic_fls[ifov]

        #fls_dics = np.array(fls_dics)[np.argsort([int(os.path.basename(fl).split('--H')[-1].split('_')[0])for fl in fls_dics])]
        #fls_fov_ = np.array(fls_dics)
        #iRs = np.array([int(os.path.basename(fl).split('_R')[-1].split('--')[0].split(',')[0]) for fl in fls_fov_])
        #iHs = np.array([int(os.path.basename(fl).split('_R')[0].split('--H')[-1]) for fl in fls_fov_])
        #iRsu,ctsRs = np.unique(iRs,return_counts=True)
        #duplicateIRs = iRsu[ctsRs>1]
        #fls_dics = [fls_fov_[iRs==iR][np.argmax(iHs[iRs==iR])] for iR in iRsu]
        self.fls_fov = fls_dics
        dic_drifts = []
        dic_pts_cells = []
        #print(fls_dics)
        for fl in tqdm(fls_dics):
            dic_drift,dic_pts_cell = pickle.load(open(fl,'rb'))
            dic_pts_cells.append(dic_pts_cell)
            fl_  = fl.replace('dic_pts_cell.pkl','new_drift.pkl')
            if os.path.exists(fl_):
                dic_drift = pickle.load(open(fl_,'rb'))
            dic_drifts.append(dic_drift)

        self.dic_pts_cells = dic_pts_cells
        self.dic_drifts = dic_drifts
        self.cells = get_cells(self.dic_pts_cells,volume_th =volume_th)
        self.volume_th = volume_th

        print("Found cells: "+str(len(self.cells)))     
    def load_fov_old(self,ifov,volume_th=200000):
        self.fov = ifov
        fls_dics = self.dic_fls[ifov]

        fls_dics = np.array(fls_dics)[np.argsort([int(os.path.basename(fl).split('--H')[-1].split('_')[0])for fl in fls_dics])]
        fls_fov_ = np.array(fls_dics)
        iRs = np.array([int(os.path.basename(fl).split('_R')[-1].split('--')[0].split(',')[0]) for fl in fls_fov_])
        iHs = np.array([int(os.path.basename(fl).split('_R')[0].split('--H')[-1]) for fl in fls_fov_])
        iRsu,ctsRs = np.unique(iRs,return_counts=True)
        #duplicateIRs = iRsu[ctsRs>1]
        fls_dics = [fls_fov_[iRs==iR][np.argmax(iHs[iRs==iR])] for iR in iRsu]
        self.fls_fov = fls_dics
        dic_drifts = []
        dic_pts_cells = []
        #print(fls_dics)
        for fl in tqdm(fls_dics):
            dic_drift,dic_pts_cell = pickle.load(open(fl,'rb'))
            dic_pts_cells.append(dic_pts_cell)
            fl_  = fl.replace('dic_pts_cell.pkl','new_drift.pkl')
            if os.path.exists(fl_):
                dic_drift = pickle.load(open(fl_,'rb'))
            dic_drifts.append(dic_drift)

        self.dic_pts_cells = dic_pts_cells
        self.dic_drifts = dic_drifts
        self.cells = get_cells(self.dic_pts_cells,volume_th =volume_th)
        self.volume_th = volume_th

        print("Found cells: "+str(len(self.cells))) 
    def check_a_cell(self,icell_,nchr_=5,volume_th=None,pix=[0.200,0.1083,0.1083],
                     radius_chr = 1.25,enhanced_radius=1.25,fr_th=0.5,plt_val = False):
        dic_drifts,dic_pts_cells = self.dic_drifts,self.dic_pts_cells
        fls_fov = self.fls_fov  
        #if volume_th is not None:
        self.cells = get_cells(dic_pts_cells,volume_th =volume_th)
        cells = self.cells 
        icell = cells[icell_]
        Xs,hs,Rs,icols = get_X(dic_drifts,dic_pts_cells,fls_fov,icell,pix=pix,nelems = nchr_)

        print(len(Rs),len(icols))
        X_chrs = determine_number_of_chromosomes(Xs,hs,Rs,radius_chr = radius_chr,
                                                 enhanced_radius=enhanced_radius,
                                                 nchr_=nchr_,fr_th = fr_th,plt_val=plt_val)
        return X_chrs


    def get_X_cands(self,nchr_=5,volume_th=None,pix=[0.200,0.1083,0.1083],
                     radius_chr = 1.25,enhanced_radius=1.25,radius_cand =2,fr_th=0.5,nelems=50,plt_val = False,dic_zdist=None):
        self.pix = pix
        
        if volume_th is None: volume_th = self.volume_th
        cells = self.cells
        fls_fov,dic_drifts,dic_pts_cells = self.fls_fov,self.dic_drifts,self.dic_pts_cells  

        X_cands = []
        icell_cands = []
        for icell in tqdm(cells):
            Xs,hs,Rs,icols = get_X(dic_drifts,dic_pts_cells,fls_fov,icell,pix=pix,nelems = nchr_,dic_zdist=dic_zdist)
            X_chrs = determine_number_of_chromosomes(Xs,hs,Rs,
                                        nchr_=nchr_,radius_chr = radius_chr,enhanced_radius=enhanced_radius,fr_th=fr_th
                                                     ,plt_val=False)
            if len(X_chrs)>0:

                Xs,hs,Rs,icols = get_X(dic_drifts,dic_pts_cells,fls_fov,icell,pix=pix,nelems = nelems,dic_zdist=dic_zdist)

                mat = cdist(X_chrs,Xs)
                nchr = len(X_chrs)
                best_asign = np.argmin(mat,axis=0)

                for ichr in range(nchr):
                    keep = (best_asign==ichr)&(mat[ichr]<radius_cand)
                    X_cands_ = np.array([Xs[keep,0],Xs[keep,1],Xs[keep,2],hs[keep,0],hs[keep,1],icols[keep],Rs[keep]]).T
                    X_cands.append(X_cands_)
                    icell_cands.append(icell)

        self.X_cands =X_cands
        self.icell_cands=icell_cands


        print("Detected number of chromosomes:" + str(len(self.icell_cands)))
        ploidy,ncells = np.unique(np.unique(self.icell_cands,return_counts=True)[-1],return_counts=True)
        for pl,nc in zip(ploidy,ncells):
            print("Number of cells with "+str(pl) +" chromosomes: "+str(nc))
    def initialize_with_max_brightness(self,nkeep = 8000,Rs_u = np.arange(177)+1):
        ### Initialize with maximum brightness########
        X_cands = self.X_cands
        self.Rs_u = Rs_u
        zxys_f,hs_f,hs_bk_f  = [],[],[]

        for pfits_cands_ in tqdm(X_cands[:nkeep]):
            zxys_T,hs_T,hs_bk_T = get_maxh_estimate(pfits_cands_,Rs_u=Rs_u)
            zxys_f.append(zxys_T)
            hs_f.append(hs_T)
            hs_bk_f.append(hs_bk_T)
        #hs_f = np.array(hs_f)# -np.array(hs_bk_f)

        self.zxys_f,self.hs_f,self.hs_bk_f = zxys_f,hs_f,hs_bk_f 

    def normalize_color_brightnesses(self):
        zxys_f,hs_f,hs_bk_f  = self.zxys_f,self.hs_f,self.hs_bk_f
        ### get dic_col
        X_cands = self.X_cands
        dic_col = {}
        for X in X_cands:
            Rs = X[:,-1]
            icols = X[:,-2]
            for R,icol in zip(Rs,icols):
                if R in self.Rs_u:
                    dic_col[R] = icol
        self.dic_col=dic_col

        cols = np.unique(list(dic_col.values()))

        hmed = np.nanmedian(np.array(hs_f),axis=0)
        Hths = np.array([np.nanmedian(hmed[[list(self.Rs_u).index(R) for R in dic_col if dic_col[R]==icol]]) 
                         for icol in cols])
        X_cands_ = [X.copy() for X in X_cands]
        for X in X_cands_:
            Rs = X[:,-1]
            icols = X[:,-2].astype(int)
            X[:,3]=X[:,3]/Hths[icols]
        self.X_cands_ = X_cands_
    def plot_std_col(self):
        hs_f = self.hs_f
        hs_bk_f = self.hs_bk_f
        iRs = np.arange(np.array(hs_f).shape[-1]//3)
        for icol in range(3):
            plt.figure()
            plt.title(str(icol))
            plt.plot(np.nanmedian(np.array(hs_f)[::2],axis=0)[icol::3],'o-')
            plt.plot(np.nanmedian(np.array(hs_f)[1::2],axis=0)[icol::3],'o-')
            plt.plot(np.nanmedian(np.array(hs_bk_f)[::2],axis=0)[icol::3],'o-')
            plt.plot(np.nanmedian(np.array(hs_bk_f)[1::2],axis=0)[icol::3],'o-')
            y = np.nanmedian(np.array(hs_f),axis=0)[icol::3]
            x = np.arange(len(y))
            for iR_,x_,y_ in zip(iRs,x,y):
                plt.text(x_,y_,str(iR_+1))

    def run_EM(self,nkeep = 8000,niter = 4,Rs_u = np.arange(175)+1):
        self.Rs_u = Rs_u
        #nkeep - Number of chromsomes to keep if want to check a subset of data 
        # iter = 4 #number of EM steps
        X_cands_ = self.X_cands_
        ### Initialize with maximum brightness########
        from tqdm import tqdm_notebook as tqdm
        zxys_f,hs_f,hs_bk_f  = [],[],[]

        for pfits_cands_ in tqdm(X_cands_[:nkeep]):
            zxys_T,hs_T,hs_bk_T = get_maxh_estimate(pfits_cands_,Rs_u = Rs_u)
            zxys_f.append(zxys_T)
            hs_f.append(hs_T)
            hs_bk_f.append(hs_bk_T)
        #hs_f = np.array(hs_f)# -np.array(hs_bk_f)

        ### Run to converge #########
        def refine_set(pfits_cands,zxys_f,hs_f,use_local=True,use_center=True,resample=1):
            Dhs,Ddists,DdistsC = get_Ddists_Dhs(zxys_f[::resample],hs_f[::resample],nint=5)
            zxys_f2,hs_f2,cols_f2,scores_f2,all_scores_f2  = [],[],[],[],[]
            i_ = 0
            for pfits_cands_ in tqdm(pfits_cands):
                    zxys_T,hs_T,scores_T,all_scores = get_statistical_estimate(pfits_cands_,Dhs,Ddists,DdistsC,
                                             zxys_T=zxys_f[i_],nint=5,use_local=use_local,use_center=use_center,Rs_u = Rs_u)
                    zxys_f2.append(zxys_T)
                    hs_f2.append(hs_T)
                    scores_f2.append(scores_T)
                    all_scores_f2.append(all_scores)
                    i_+=1
            return zxys_f2,hs_f2,scores_f2,all_scores_f2

        saved_zxys_f=[zxys_f[:nkeep]]
        save_hs_f=[hs_f[:nkeep]]

        for num_ref in range(niter):
            use_local = True#num_ref>=niter/2
            print('EM iteration number: ',num_ref+1)

            zxys_f,hs_f,scores_f,all_scores_f = refine_set(X_cands_[:nkeep],zxys_f[:nkeep],hs_f[:nkeep],use_local=use_local)
            saved_zxys_f.append(zxys_f)
            save_hs_f.append(hs_f)

            #check convergence
            dif = np.array(saved_zxys_f[-1])-np.array(saved_zxys_f[-2])
            nan =  np.all(np.isnan(dif),axis=-1)
            same = nan|np.all(dif==0,axis=-1)
            print("fraction the same:",np.sum(same)/float(np.prod(same.shape)))
            print("fraction nan:",np.sum(nan)/float(np.prod(nan.shape)))
        self.zxys_f,self.hs_f,self.scores_f = zxys_f,hs_f,scores_f
        self.all_scores_f = all_scores_f

    def get_scores_and_threshold(self,th_score = -6):
        scores_f,all_scores_f = self.scores_f,self.all_scores_f

        scores_all_ = [sc_ for scs in all_scores_f for sc in scs for sc_ in np.sort(sc)[:-1]]
        scores_good_ = [sc_ for scs in scores_f for sc_ in scs]
        scores_all_ = np.array(scores_all_)
        scores_all_ = scores_all_[~np.isnan(scores_all_)]
        scores_good_ = np.array(scores_good_)
        scores_good__ = scores_good_[~np.isnan(scores_good_)]

        plt.figure()
        plt.ylabel('Probability density')
        plt.xlabel('Log-score')
        plt.hist(np.log(scores_good__),density=True,bins=100,alpha=0.5,label='good spots');
        plt.hist(np.log(scores_all_),density=True,bins=100,alpha=0.5,label='background spots');
        plt.legend()



        plt.figure()
        plt.plot(np.mean(np.log(scores_f)>th_score,axis=0),'o-')
        plt.ylabel('Detection efficiency')
        plt.xlabel('Region')
        plt.figure()
        det_ef = np.mean(np.log(scores_f)>th_score,axis=1)
        plt.plot(det_ef)
        plt.title("Median detection efficiency: "+str(np.round(np.median(det_ef),2)))
        plt.ylabel('Detection efficiency')
        plt.xlabel('Chromosome')


    def plot_matrix(self,th_score=-5,lazy_color_correction = True):
        self.th_score = th_score
        Xf = np.array(self.zxys_f)
        bad = np.log(self.scores_f)<th_score
        Xf[bad] = np.nan
        if lazy_color_correction:
            ncol=3
            cm = np.nanmean(Xf[:,:,:],axis=1)[:,np.newaxis]
            for icol in range(ncol):
                Xf[:,icol::ncol,:]-=np.nanmedian(Xf[:,icol::ncol,:],axis=1)[:,np.newaxis]+cm

        from scipy.spatial.distance import pdist,squareform
        mats = np.array([squareform(pdist(X_)) for X_ in Xf])

        plt.figure(figsize=(10,10))
        keep = np.arange(mats.shape[1])#[icol::3]
        plt.imshow(np.nanmedian(mats[:,keep][:,:,keep],0),vmax=1,vmin=0.2,cmap='seismic_r')

        if False:
            from scipy.spatial.distance import pdist,squareform
            mats = np.array([squareform(pdist(X_)) for X_ in Xf])
            for icol in range(3):
                plt.figure()
                keep = np.arange(mats.shape[1])[icol::3]
                plt.imshow(np.nanmedian(mats[:,keep][:,:,keep],0),vmax=1,vmin=0.2,cmap='seismic_r')