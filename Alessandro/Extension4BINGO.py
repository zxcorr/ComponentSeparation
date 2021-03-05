import numpy as np
import time,os

def remove_mean(map_= None):
	len_nu_ch = np.shape(map_)[0]
	map_ = np.array([[map_[i] - np.mean(map_[i],axis=0)] for i in range(len_nu_ch)])
	return map_[:,0,:]

def getmaps(type_= None, observed_without_mean=True, apply_mask = False, add_noise=True, sigmaE=None, path = "/home/marins/Documents/Programmation/BINGO/Component Separation/maps",
name_21cm="(0)Cube_21_Smooth_L10.fits", name_fg =  "(0)Cube_5PSM_L10_RS.fits", name_mask = "Mask_Bin.fits", name_noise = None, name_prior = None, name_pure=None):
	import os
	if type_=="MeerKAT":
		import h5py
		path2file = os.path.join(path,'sim_10MHz.hd5')
		file = h5py.File(path2file,'r')
		# Preparing the observed map (sum of all components + noise):
		components = list(file.keys())
		components.remove('frequencies')
		# and save the channel frequencies
		len_nu_ch = len(np.array(file['frequencies']))
		
		npix = np.shape(file[file.keys()[0]])[1]
		obs_maps = np.zeros((len_nu_ch,npix))
		
		for c in components:
			if c=="cosmological_signal":
				maps = {"21cm": np.array(file[c])}
				obs_maps += np.array(file[c])
			else:
				maps[c]   = np.array(file[c])  
				obs_maps += np.array(file[c])
				try:
					maps["foregrounds"] += np.array(file[c])         
				except:
					maps["foregrounds"] = np.array(file[c])
		
		
	elif type_=="BINGO":
		import astropy.io.fits as fits
		files      = [name_21cm, name_fg]#, "psm_ame_rot_mK.fits", "psm_cmb_rot_mK.fits", "psm_free_rot_mK.fits", "psm_frps_rot_mK.fits", "psm_synch_rot_mK.fits"]
		names      = ["21cm", "foregrounds"]#, "AME", "CMB", "FF", "FRPS", "SYNCH"]
		components = ["21cm"]#, "AME", "CMB", "FF", "FRPS", "SYNCH"]
		paths      = []
		for ifile in files:
			paths.append(os.path.join(path,ifile))
		
		for i,ipath in enumerate(paths):
			with fits.open(ipath) as h:
				if i==0:
					maps       = {names[i]:h[0].data}
					nbins,npix = maps[names[i]].shape
				else:
					maps[names[i]]=h[0].data
		maps.update(foregrounds=maps["foregrounds"] - maps["21cm"])
		maps["observed"] = maps["foregrounds"] + maps["21cm"]

	elif type_=="GNILC":
		import astropy.io.fits as fits
		gnilc_maps = "reconstructed_maps_No_M1_L100_M.fits"
		pathmask = os.path.join(path,gnilc_maps)
		with fits.open(pathmask) as h:
			maps = h[0].data
		del pathmask
		return maps

	elif type_=="noise":
		import astropy.io.fits as fits
		path   = os.path.join(path,name_noise)
		with fits.open(path) as h:
			maps = h[0].data
		return maps

	elif type_=="prior":
		import astropy.io.fits as fits
		path   = os.path.join(path,name_prior)
		with fits.open(path) as h:
			maps = h[0].data
		return maps

	elif type_=="pure":
		import astropy.io.fits as fits
		path   = os.path.join(path,name_pure)
		with fits.open(path) as h:
			maps = h[0].data
		return maps
		
	else:
		raise NameError("There is no any {} maps".format(type_))
	
	if add_noise:
		if type(name_noise) == str:
			pathwn = os.path.join(path,name_noise)
			with fits.open(pathwn) as h:
				noise = h[0].data
				maps.update(observed=maps["observed"]+noise)
				#obs_maps += WN
				maps["noise"] = noise
		elif name_noise==None:
			WN = noise(maps, sigmaE=sigmaE)
			maps["observed"]+= noise
			maps["noise"] = noise
		else: 
			raise NameError
		del noise
		
	if apply_mask:
		pathmask = os.path.join(path,name_mask)
		with fits.open(pathmask) as h:
			mask = h[0].data
		for i,c in enumerate(maps.keys()):
			maps[c] = maps[c]*mask
		maps["mask"] = mask			
		del pathmask
		
	if observed_without_mean:
		if apply_mask:
			maps.update(observed=remove_mean(maps["observed"]))
		else:
			maps.update(observed=remove_mean(maps["observed"]))
		
	return maps


def Include_scalar_coefs(w,c,J):
    nbins,npix = c.shape

    for i in range(nbins):
        X = w[i].reshape(-1,J).T
        X = np.vstack((X,c[i]))
        X = X.T.flatten()
        if i==0:
            Xw_ = np.array(X)
        else:
            Xw_ = np.vstack((Xw_,X))
    return Xw_

def reshape_coefs(Xw=None, J=None, ndiv=1, idiv=0, use_scale_coefs=True): #ndiv = number of the divisions #idiv = i-division
    if use_scale_coefs:
        J=J+1
    nbins, npix = Xw.shape
    
    if (idiv<ndiv)*((nbins - np.absolute(np.fix(nbins)))>0.):
        raise ValueError #TypeError("idiv+1>=ndiv. Number of the division is taller then of divisions.")
    else:
        npart = int(J/ndiv)
    
    npix  = npix/J

    for i in range(nbins):
        X = Xw[i].reshape(int(npix),int(J)).T
        X = X[idiv*npart:(idiv+1)*npart,:].T.flatten()
        if i==0:
            Xw_ = np.array(X)
        else:
            Xw_ = np.vstack((Xw_,X))
    return Xw_		

def Filter(Ae=None, FG=False, without_covx=True, X=None):
	if without_covx:
		W   = np.linalg.inv(np.dot(Ae.T,Ae))
		W   = np.dot(W,Ae.T) #Filter
	else:
		AC  = np.dot(Ae.T,np.linalg.inv(np.cov(X)))
		W   = np.linalg.inv(np.dot(AC,Ae))
		W   = np.dot(W,AC) #Filter		#Gauss-Markov estimator
	if FG:
		return np.dot(Ae,W) #foreground filter
	else:
		return W

def Reconstruction_maps(X=None,Ae=None, without_covx=True):
    W    = Filter(Ae,True, without_covx, X)
    X_fg = np.dot(W,X)
    X_21 = X - X_fg
    return {"21cm":X_21,"foregrounds":X_fg}

def Residual_maps(X=None,Ae=None,without_covx=True):
    X    = Reconstruction_maps(X,Ae,without_covx)
    W_fg = Filter(Ae,True, without_covx, X)
    R_21 = X["21cm"]        - np.dot(W_fg,X["21cm"])  
    R_fg = X["foregrounds"] - np.dot(W_fg,X["foregrounds"])  
    return {"21cm":R_21, "foregrounds":R_fg}


def w2GMCAmaps(coefs, maps, J=None, div = 1, n_s=3, mints=0.1, nmax=100, L=0, AInit=None, ColFixed=None, whitening=False, epsi=1e-3,verbose=False, use_scale_coefs=True):
    import gmca4im_lib2 as g4i
    print("Starting Component Analysis...")
    time0 = time.time()
    if int(div)==1:
        A,S = g4i.run_GMCA(coefs, AInit, n_s, mints, nmax, L, ColFixed, whitening, epsi, verbose)
        del S, coefs
        print("Building reconstructed maps...")
        m   = Reconstruction_maps(maps,A)
        print("Building residuals maps...")
        r   = Residual_maps(maps,A)
        m_rec_21 = m["21cm"]
        m_rec_fg = m["foregrounds"]
        r_rec_21 = r["21cm"]
        r_rec_fg = r["foregrounds"]
        Am       = A
    else:
        for i in range(int(div)):
            w_ = reshape_coefs(Xw=coefs, J=J, ndiv=div, idiv=i, use_scale_coefs=use_scale_coefs)
            A,S = g4i.run_GMCA(w_, AInit, n_s, mints, nmax, L, ColFixed, whitening, epsi, verbose)
            del S,w_
            print("Building reconstructed maps...")
            m   = Reconstruction_maps(maps,A)
            print("Building residuals maps...")
            r   = Residual_maps(maps,A)
            if i==0:
                nbins,npix = m["21cm"].shape
                m_rec_21 = np.zeros((nbins,npix))
                m_rec_fg = np.zeros((nbins,npix))
                r_rec_21 = np.zeros((nbins,npix))
                r_rec_fg = np.zeros((nbins,npix))
            m_rec_21 += m["21cm"]
            m_rec_fg += m["foregrounds"]
            r_rec_21 += r["21cm"]
            r_rec_fg += r["foregrounds"]
            if i==0:
                Am = A
            else:
                Am = np.vstack((Am,A))
    time0 = time.time()-time0			
    print("Finished in: {:.2f} min".format(time0/60))
    return {"reconstruction":{"21cm":m_rec_21, "foregrounds":m_rec_fg},"residual":{"21cm":r_rec_21, "foregrounds":r_rec_fg}, "mixmatrix":Am}


def maps2GMCAmaps(maps, J=3, div = 3, n_s=3, mints=0.1, nmax=100, L=0, AInit=None, ColFixed=None, whitening=False, epsi=1e-3, use_scale_coefs=True, WT="starlet"):#maps=observed maps #WT = starlet, axisym, directional
	import gmca4im_lib2 as g4i
	if WT=="starlet":
		w, c = g4i.wavelet_transform(maps,J)
	elif WT=="axisym":
		raise NameError("Not implemented yet: {0}".format(WT))
	elif WT=="directional":
		raise NameError("Not implemented yet: {0}".format(WT))
	else:
		raise NameError("Not implemented: {0}".format(WT))
	if use_scale_coefs:
		X_wt = Include_scalar_coefs(w,c,J)
	else:
		X_wt = w
	return w2GMCAmaps(coefs=X_wt, maps=maps, J=J, div = div, n_s= n_s, mints=mints, nmax=nmax, L=L, AInit=AInit, ColFixed=ColFixed, whitening=whitening, epsi=epsi, use_scale_coefs=use_scale_coefs)
    
def Xw_coefs(w,c,ich,J):
    w_star=w[ich].reshape(-1,J)
    c_star=c[ich]
    len_ = len(c_star)
    Xw = np.empty((len_,J+1))
    for i in range(J+1):
        if i==J:
            Xw[:,i]=c_star
        else:
            Xw[:,i]=w_star[:,i]
    return Xw

def maps_wavs(w,c,J):
    bins = np.shape(w)[0]
    for i in range(bins):
        if i == 0:
            Xw= {str(i):Xw_coefs(w,c,i,J)}
        else:
            Xw[str(i)]=Xw_coefs(w,c,i,J)
    return Xw

def white_noise(maps=None, sigmaE=None):
	nbin,npix = maps["21cm"].shape
	WN = np.zeros((nbin,npix))
	
	if sigmaE==None:
		sigmaE = np.std(maps["21cm"],axis=1)
		for i in range(nbin):
			WN[i,:] += np.random.normal(loc=0,scale=sigmaE[i],size=npix)	
	else:
		WN+=sigmaE
	return WN

def noisedebiasing(Cls_ = None, seed_used = None, dir_hi = None, dir_prior = None, dir_noise = None , dir_pure = None, dir_projpure = None, dir_projnoise = None, dir_projprior = None, type_ = None): #Cls=dictionary with all of components #type_=filipe or mathieu
    if type_=="filipe":
        if not type(Cls_)==dict:
            raise Exception("Cls is not a dictionary!")

        L0     = "L{}".format(seed_used)
        Cls_L0 = Cls_[dir_hi][L0]

        for k in Cls_.keys():
            del Cls_[k][L0]

        for i, Li in enumerate(Cls_[dir_hi].keys()):
            if i==0:
                S        = Cls_[dir_hi][Li]/Cls_[dir_prior][Li]
                Cls_noise = Cls_[dir_noise][Li]
            else:
                S         += Cls_[dir_hi][Li]/Cls_[dir_prior][Li]
                Cls_noise += Cls_[dir_noise][Li]

        S         = S/(len(Cls_[dir_hi].keys()))
        Cls_noise = Cls_noise/(len(Cls_[dir_hi].keys()))
        Cls_ndb   = (Cls_L0/S) - Cls_noise    
        return {"Cls_debias":Cls_ndb,"S":S,"Cls_noise":Cls_noise}
    elif type_=="mathieu":
        if not type(Cls_)==dict:
            raise Exception("Cls is not a dictionary!")

        L0     = "L{}".format(seed_used)
        Cls_L0 = Cls_[dir_hi][L0]

        for k in Cls_.keys():
            del Cls_[k][L0]

        for i, Li in enumerate(Cls_[dir_hi].keys()):
            if i==0:
                S             = Cls_[dir_projpure][Li]/Cls_[dir_pure][Li]
                Cls_projnoise = Cls_[dir_projnoise][Li]
            else:
                S             += Cls_[dir_projpure][Li]/Cls_[dir_pure][Li]
                Cls_projnoise += Cls_[dir_projnoise][Li]

        S             = S/(len(Cls_[dir_hi].keys()))
        Cls_projnoise = Cls_projnoise/(len(Cls_[dir_hi].keys()))
        Cls_ndb       = (Cls_L0 - Cls_projnoise)/S
        return {"Cls_debias":Cls_ndb,"S":S,"Cls_noise":Cls_projnoise}        

def savedata(Cl_, filename=None, path=None, iseed=None, header= "Cl pix p/ column, bin p/ row"):
    nu,npix = Cl_.shape
    filename = "".join((filename, "_" + iseed ,".txt"))
    pathname = os.path.join(path,filename)
    np.savetxt(pathname, Cl_.T, fmt=["%e"]*nu, delimiter=" ", header=header)

#mrec can be a dictionary when used with 21cm, foregrounds and mixmatrix; or a matrix when white_noise
def saveouts(mrec=None, A=None, pathout=None, iseed=None, J=3, div=3+1,header= None, plot_="reconstruction", subdirs=["21cm","foregrounds","mixmatrix"], without_covx=True):
	import healpy as hp
	subdirs = np.asarray(subdirs)
	ind     = np.union1d(np.where(subdirs=="21cm")[0],np.where(subdirs=="foregrounds")[0])
	if len(ind)>0:
		nu,npix = np.shape(mrec[plot_][subdirs[ind][0]])
		nside   = hp.npix2nside(npix)
		L       =  3*nside
		for type_ in subdirs[ind]:
			path = os.path.join(pathout,type_)
			cl   = np.zeros((nu,L))
			for inu in range(nu):
				cl[inu] += hp.anafast(mrec[plot_][type_][inu], lmax=L-1)
			savedata(Cl_=cl, filename="cl", path=path, iseed=iseed)
			del cl
	if "mixmatrix" in subdirs:
		path = os.path.join(pathout,"mixmatrix")
		nu   = int(mrec["mixmatrix"].shape[0]/(div))
		for i in range(div):
			A = mrec["mixmatrix"][i*nu:(i+1)*nu,:]
			savedata(Cl_= A.T, filename="A"+str(i), path=path, iseed=iseed, header="")
	if "noise" in subdirs:
		path    = os.path.join(pathout,"noise")
		nu,npix = np.shape(mrec) #Here, mrec==cube of maps and not a python dictionary
		nside   = hp.npix2nside(npix)
		L       =  3*nside
		cl      = np.zeros((nu,L))
		for inu in range(nu):
			cl[inu] += hp.anafast(mrec[inu], lmax=L-1)
		savedata(Cl_=cl, filename="cl", path=path, iseed=iseed)		
		del cl
	if "prior" in subdirs:
		path    = os.path.join(pathout,"prior")
		nu,npix = np.shape(mrec) #Here, mrec==cube of maps and not a python dictionary
		nside   = hp.npix2nside(npix)
		L       =  3*nside
		cl      = np.zeros((nu,L))
		for inu in range(nu):
			cl[inu] += hp.anafast(mrec[inu], lmax=L-1)
		savedata(Cl_=cl, filename="cl", path=path, iseed=iseed)
		del cl		
	if "pure" in subdirs:
		path    = os.path.join(pathout,"pure")
		nu,npix = np.shape(mrec) #Here, mrec==cube of maps and not a python dictionary
		nside   = hp.npix2nside(npix)
		L       =  3*nside
		cl      = np.zeros((nu,L))
		for inu in range(nu):
			cl[inu] += hp.anafast(mrec[inu], lmax=L-1)
		savedata(Cl_=cl, filename="cl", path=path, iseed=iseed)
		del cl		
	if "projpure" in subdirs:
		path    = os.path.join(pathout,"projpure")
		nu,npix = np.shape(mrec) #Here, mrec==cube of maps and not a python dictionary
		nside   = hp.npix2nside(npix)
		L       =  3*nside
		cl      = np.zeros((nu,L))		
		R21     = Residual_maps(X=mrec,Ae=A,without_covx=without_covx)["foregrounds"] #A = A[Li] #here, I used "foregrounds" because it is Wfg*X (X=Xpure)
		for inu in range(nu):
			cl[inu] += hp.anafast(R21[inu], lmax=L-1)
		savedata(Cl_=cl, filename="cl", path=path, iseed=iseed)
		del cl				
	if "projnoise" in subdirs:
		path    = os.path.join(pathout,"projnoise")
		nu,npix = np.shape(mrec) #Here, mrec==cube of maps and not a python dictionary
		nside   = hp.npix2nside(npix)
		L       =  3*nside
		cl      = np.zeros((nu,L))		
		R21     = Residual_maps(X=mrec,Ae=A,without_covx=without_covx)["foregrounds"] #A = A[Li] #here, I used "foregrounds" because it is Wfg*X (X=Xnoise)
		for inu in range(nu):
			cl[inu] += hp.anafast(R21[inu], lmax=L-1)
		savedata(Cl_=cl, filename="cl", path=path, iseed=iseed)
		del cl						
	if "projprior" in subdirs:
		path    = os.path.join(pathout,"projprior")
		nu,npix = np.shape(mrec) #Here, mrec==cube of maps and not a python dictionary
		nside   = hp.npix2nside(npix)
		L       =  3*nside
		cl      = np.zeros((nu,L))		
		R21     = Residual_maps(X=mrec,Ae=A,without_covx=without_covx)["21cm"] #A = A[Li] #here, I used "foregrounds" because it is Wfg*X (X=Xnoise)
		for inu in range(nu):
			cl[inu] += hp.anafast(R21[inu], lmax=L-1)
		savedata(Cl_=cl, filename="cl", path=path, iseed=iseed)
		del cl								

def checkdir(pathout=None, subdirs=["21cm","foregrounds","mixmatrix","prior","noise","pure","projpure","projnoise","projprior"], return_=True):
    import shutil
    if not os.path.isdir(pathout):
        os.makedirs(pathout)
    else:
        shutil.rmtree(pathout)
        os.makedirs(pathout)
    for subdir in subdirs:
        os.makedirs(os.path.join(pathout,subdir))
    if return_:
        return subdirs
    else:
        return None

def clsbinned(cls=None,del_l=10,l0=0): #cl= matrix with the rows being maps and columns spatial positions #del_l=size of the binned #l0=start multipole 
    l_     = np.arange(len(cls)) #l=[0,1,2,...,lmax]
    lmin   = max(l0,min(l_)) 
    n_dell = int((max(l_)-lmin)/del_l) #number of multipole bins
    lmax   = n_dell*del_l + lmin - 1
    lnew_  = np.arange(lmin,lmax+1,1) #new vector of multipoles
    clnew = np.empty((n_dell))
    lnew   = []
    for bin_l in range(n_dell):
        l_ini    = int(bin_l*del_l)
        l_ini    = int(lnew_[l_ini])
        l_bin    = np.arange(l_ini,int(l_ini + del_l))
        weight_l = 2*l_bin + np.ones(len(l_bin))
        clnew[bin_l] = np.dot(weight_l,cls[l_bin])/np.sum(weight_l)
        lnew.append(l_ini)
    lnew=np.asarray(lnew)
    return lnew,clnew
    
def loadcls(pathcls=None,dirs=["21cm","foregrounds","prior","noise","pure","projpure","projnoise"]):
	for i,dir_ in enumerate(dirs):
		path  = os.path.join(pathcls,dir_)
		names = os.listdir(path)
		for j,iname in enumerate(names):
			num = iname.split("_")[-1].split(".")[0]
			cl = np.loadtxt(os.path.join(path,iname)).T
			if j==0:
				cls = {num:cl}
			else:
				cls[num] = cl
		if i==0:
			Cls={dir_:cls}
		else:
			Cls[dir_]=cls
	return Cls
	
def loadmixmatrix(pathA=None, mixmatrixdir ="mixmatrix"):
    path  = os.path.join(pathA,mixmatrixdir)
    names = os.listdir(path)
    for j,iname in enumerate(names):
        num = iname.split("_")[-1].split(".")[0]
        Ai = np.loadtxt(os.path.join(path,iname))
        if j==0:
            A = {num:Ai}
        else:
            A[num] = Ai
    return A

def index_cls_binned(l_,lbin_):
    inds = []
    for il in lbin_:
        i = np.where(l_==il)[0][0]
        inds.append(i)
    return np.asarray(inds)
