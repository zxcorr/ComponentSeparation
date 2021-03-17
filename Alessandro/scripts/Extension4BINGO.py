import numpy as np
import pandas as pd
import time,os

def remove_mean(map_= None):
	len_nu_ch = np.shape(map_)[0]
	map_ = np.array([[map_[i] - np.mean(map_[i],axis=0)] for i in range(len_nu_ch)])
	return map_[:,0,:]

def getmaps(params_maps=None, params_path=None):
	import os
	if  params_maps.getdata=="observed":
		import astropy.io.fits as fits
		with fits.open(os.path.join(params_path.dir_observed,params_path.name_observed)) as h:
			maps = h[0].data
		return maps

	elif params_maps.getdata=="GNILC":
		import astropy.io.fits as fits
		#params_maps.name_gnilc ="reconstructed_maps_No_M1_L100_M.fits"
		with fits.open(os.path.join(params_path.dir_gnilc,params_path.name_gnilc)) as h:
			maps = h[0].data
		return maps

	elif params_maps.getdata=="noise":
		import astropy.io.fits as fits
		with fits.open(os.path.join(params_path.dir_noise,params_path.name_noise)) as h:
			maps = h[0].data
		return maps

	elif params_maps.getdata=="prior":
		import astropy.io.fits as fits
		with fits.open(os.path.join(params_path.dir_prior,params_path.name_prior)) as h:
			maps = h[0].data
		return maps

	elif params_maps.getdata=="pure":
		import astropy.io.fits as fits
		with fits.open(os.path.join(params_path.dir_pure,params_path.name_pure)) as h:
			maps = h[0].data
		return maps
	
	elif params_maps.getdata=="foregrounds":
		import astropy.io.fits as fits
		with fits.open(os.path.join(params_path.dir_fg,params_path.name_fg)) as h:
			maps = h[0].data
		return maps
		
	else:
		raise NameError("There is no any {} maps".format(params_maps.survey))

def adaptation_maps(maps=None, params_maps=None, params_path=None):		
	import astropy.io.fits as fits
	if params_maps.apply_mask:
		with fits.open(os.path.join(params_path.dir_mask,params_path.name_mask)) as h:
			mask = h[0].data
		maps = maps*mask
	if params_maps.without_mean:
		if params_maps.apply_mask:
			maps = remove_mean(maps)
		else:
			maps = remove_mean(maps)
	return maps


def include_scalar_coefs(w,c,J):
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

def filterW(Ae=None, FG=False, without_covx=True, X=None):
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

def reconstruction_maps(X=None,Ae=None, without_covx=True):
    W    = filterW(Ae,True, without_covx, X)
    X_fg = np.dot(W,X)
    X_21 = X - X_fg
    return {"21cm":X_21,"foregrounds":X_fg}

def residual_maps(X=None,Ae=None,without_covx=True):
    X    = reconstruction_maps(X,Ae,without_covx)
    W_fg = filterW(Ae,True, without_covx, X)
    R_21 = X["21cm"]        - np.dot(W_fg,X["21cm"])  
    R_fg = X["foregrounds"] - np.dot(W_fg,X["foregrounds"])  
    return {"21cm":R_21, "foregrounds":R_fg}


def maps2GMCAmaps(X, params_WT, params_CS):
	import gmca4im_lib2 as g4i
	if params_WT.wtransform=="starlet":
		w, c = g4i.wavelet_transform(X,params_WT.J)
	elif params_WT.wtransform=="axisym":
		w, c = coefs_S2LETransform(X,params_WT)
	elif params_WT.wtransform=="directional":
		raise NameError("Not implemented yet: {0}".format(params_WT.wtransform))
	else:
		raise NameError("Not implemented: {0}".format(params_WT.wtransform))
	if params_WT.use_c:
		X_wt = include_scalar_coefs(w,c,params_WT.J)
	else:
		X_wt = w
	#return w,c
	return w2GMCAmaps(X_wt, X, params_WT, params_CS)

def w2GMCAmaps(X_wt, X, params_WT, params_CS):   
    import gmca4im_lib2 as g4i
    print("Starting Component Analysis...")
    time0 = time.time()
    if int(params_CS.division)==1:
        A,S = g4i.run_GMCA(X_wt, params_CS.A_ini, params_CS.ns, params_CS.mints, params_CS.nmax, params_CS.L0, params_CS.ColFixed, params_CS.whitening, params_CS.epsi, params_CS.verbose)
        del S, X_wt
        print("Building reconstructed maps...")
        m   = reconstruction_maps(X,A)
        print("Building residuals maps...")
        r   = residual_maps(X,A)
        m_rec_21 = m["21cm"]
        m_rec_fg = m["foregrounds"]
        r_rec_21 = r["21cm"]
        r_rec_fg = r["foregrounds"]
        Am       = A
    else:
        for i in range(int(params_CS.division)):
            print(param_CS.division, str(i))
            w_  = reshape_coefs(Xw=X_wt, J=params_WT.J, ndiv=params_CS.division, idiv=i, use_scale_coefs=params_WT.use_c)
            A,S = g4i.run_GMCA(w_, params_CS.A_ini, params_CS.ns, params_CS.mints, params_CS.nmax, params_CS.L0, params_CS.ColFixed, params_CS.whitening, params_CS.epsi, params_CS.verbose)
            del S,w_
            print("Building reconstructed maps...")
            m   = reconstruction_maps(X,A)
            print("Building residuals maps...")
            r   = residual_maps(X,A)
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

def white_noise(maps21cm=None, sigmaE=None):
	nbin,npix = maps21cm.shape
	WN = np.zeros((nbin,npix))
	
	if sigmaE==None:
		sigmaE = np.std(maps21cm,axis=1)
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
				S         = Cls_[dir_hi][Li]/Cls_[dir_prior][Li]
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

def saveouts(mrec=None, A=None, header= None, params_path=None, params_maps=None, params_WT=None, params_CS=None, subdirs=["21cm","foregrounds","mixmatrix"]):
    import healpy as hp
    subdirs = np.asarray(subdirs)
    ind     = np.union1d(np.where(subdirs=="21cm")[0],np.where(subdirs=="foregrounds")[0])
    if len(ind)>0:
        nu,npix = np.shape(mrec[params_maps.cl_type_save][subdirs[ind][0]])
        nside   = hp.npix2nside(npix)
        L       =  3*nside
        for type_ in subdirs[ind]:
            path = os.path.join(params_path.pathout,type_)
            cl   = np.zeros((nu,L))
            for inu in range(nu):
                cl[inu] += hp.anafast(mrec[params_maps.cl_type_save][type_][inu], lmax=L-1)
            savedata(Cl_=cl, filename="cl", path=path, iseed=params_maps.iseed)
            del cl
    if "mixmatrix" in subdirs:
        path = os.path.join(params_path.pathout,"mixmatrix")
        nu   = int(mrec["mixmatrix"].shape[0]/(params_CS.division))
        for i in range(params_CS.division):
            A = mrec["mixmatrix"][i*nu:(i+1)*nu,:]
            savedata(Cl_= A.T, filename="A"+str(i), path=path, iseed=params_maps.iseed, header="")
    if "noise" in subdirs:
        path    = os.path.join(params_path.pathout,"noise")
        nu,npix = np.shape(mrec) #Here, mrec==cube of maps and not a python dictionary
        nside   = hp.npix2nside(npix)
        L       =  3*nside
        cl      = np.zeros((nu,L))
        for inu in range(nu):
            cl[inu] += hp.anafast(mrec[inu], lmax=L-1)
        savedata(Cl_=cl, filename="cl", path=path, iseed=params_maps.iseed)
        del cl
    if "prior" in subdirs:
        path    = os.path.join(params_path.pathout,"prior")
        nu,npix = np.shape(mrec) #Here, mrec==cube of maps and not a python dictionary
        nside   = hp.npix2nside(npix)
        L       =  3*nside
        cl      = np.zeros((nu,L))
        for inu in range(nu):
            cl[inu] += hp.anafast(mrec[inu], lmax=L-1)
        savedata(Cl_=cl, filename="cl", path=path, iseed=params_maps.iseed)
        del cl
    if "pure" in subdirs:
        path    = os.path.join(params_path.pathout,"pure")
        nu,npix = np.shape(mrec) #Here, mrec==cube of maps and not a python dictionary
        nside   = hp.npix2nside(npix)
        L       =  3*nside
        cl      = np.zeros((nu,L))
        for inu in range(nu):
            cl[inu] += hp.anafast(mrec[inu], lmax=L-1)
        savedata(Cl_=cl, filename="cl", path=path, iseed=params_maps.iseed)
        del cl
    if "projpure" in subdirs:
        path    = os.path.join(params_path.pathout,"projpure")
        nu,npix = np.shape(mrec) #Here, mrec==cube of maps and not a python dictionary
        nside   = hp.npix2nside(npix)
        L       =  3*nside
        cl      = np.zeros((nu,L))
        R21     = residual_maps(X=mrec,Ae=A,without_covx=params_CS.without_covx)["21cm"] #A = A[Li] #here, I used "foregrounds" because it is Wfg*X (X=Xpure)
        for inu in range(nu):
            cl[inu] += hp.anafast(R21[inu], lmax=L-1)
        savedata(Cl_=cl, filename="cl", path=path, iseed=params_maps.iseed)
        del cl
    if "projnoise" in subdirs:
        path    = os.path.join(params_path.pathout,"projnoise")
        nu,npix = np.shape(mrec) #Here, mrec==cube of maps and not a python dictionary
        nside   = hp.npix2nside(npix)
        L       =  3*nside
        cl      = np.zeros((nu,L))
        R21     = residual_maps(X=mrec,Ae=A,without_covx=params_CS.without_covx)["21cm"] #A = A[Li] #here, I used "foregrounds" because it is Wfg*X (X=Xnoise)
        for inu in range(nu):
            cl[inu] += hp.anafast(R21[inu], lmax=L-1)
        savedata(Cl_=cl, filename="cl", path=path, iseed=params_maps.iseed)
        del cl
    if "projprior" in subdirs:
        path    = os.path.join(params_path.pathout,"projprior")
        nu,npix = np.shape(mrec) #Here, mrec==cube of maps and not a python dictionary
        nside   = hp.npix2nside(npix)
        L       =  3*nside
        cl      = np.zeros((nu,L))
        R21     = residual_maps(X=mrec,Ae=A,without_covx=params_CS.without_covx)["21cm"] #A = A[Li] #here, I used "foregrounds" because it is Wfg*X (X=Xprior)
        for inu in range(nu):
            cl[inu] += hp.anafast(R21[inu], lmax=L-1)
        savedata(Cl_=cl, filename="cl", path=path, iseed=params_maps.iseed)
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
    
def loadtheoricalcls(pathcls=None, type_="auto"):
	clt = np.loadtxt(pathcls)
	if type_=="auto":
		for j in np.arange(int((-1 + np.sqrt(1 - 8*(1-clt.shape[1])))/2.)):  #ncols = nch**2/2 + nch/2 + 1
			if j==0:
				col = 1
				cols=[col]
			else:
				col+=30 - (j-1)
				cols.append(col)
		cols = np.asarray(cols)        #autocorrelation from theorical data
		#cols+=1
		return clt[:,0],clt[:,cols].T
	else:
		return clt[:,0],clt.T
		
	

def index_cls_binned(l_,lbin_):
    inds = []
    for il in lbin_:
        i = np.where(l_==il)[0][0]
        inds.append(i)
    return np.asarray(inds)


def S2LETransform(X,params_WT): #return 1 cube
	import healpy as hp
	import pys2let as pys2	
	nbin,npix = X.shape
	nside     = int(hp.npix2nside(npix))
	if params_WT.L==None:
		params_WT.L=3*nside
	Jmax      = pys2.pys2let_j_max(params_WT.B, params_WT.L, params_WT.Jmin)
	J         = Jmax - params_WT.Jmin + 2
	params_WT["Jmax"]  = Jmax
	params_WT["J"]     = J
	if params_WT.wtransform=="axisym":
		Xwt = np.zeros((nbin, J*npix))
		for i in range(nbin):
			Xi = X[i]
			X_lm = hp.map2alm(Xi, lmax=params_WT.L - 1)  # Its alms
			wlm,clm = pys2.analysis_axisym_lm_wav(X_lm, params_WT.B, params_WT.L, params_WT.Jmin)
			c = hp.alm2map(clm, nside=nside, lmax=params_WT.L - 1, verbose=False) #coeficiente escalar
			w = np.empty([npix, Jmax-params_WT.Jmin+1]) #coeficiente wavelet
			for j in range(Jmax-params_WT.Jmin+1):
				flm = wlm[:, j].ravel()
				w[:, j] = hp.alm2map(flm, nside=nside, lmax=params_WT.L - 1, verbose=False)
			Xwt[i,:] = np.vstack((w.T, c)).T.flatten()
		params_WT["J"] = J - 1
		return Xwt
	else:
		raise Exception("There is no {} transform".format(params_WT.wtransform))    

def coefs_S2LETransform(X,params_WT):#return J+1 cubes
	import time
	import healpy as hp
	import pys2let as pys2
	import sys
	timei = time.time()
	print("Starting {} wavelet transform ...".format(params_WT.wtransform))
	nbin,npix = X.shape
	nside     = int(hp.npix2nside(npix))
	if params_WT.L==None:
		params_WT.L=3*nside
	Jmax      = pys2.pys2let_j_max(params_WT.B, params_WT.L, params_WT.Jmin)
	params_WT["Jmax"]  = Jmax
	params_WT["J"]  = Jmax - params_WT.Jmin + 1
	c_wt = []
	w_wt = []
	if params_WT.wtransform=="axisym":
		Xwt = np.zeros((nbin, (params_WT.J+1)*npix))
		for i in range(nbin):
			Xi = X[i]
			X_lm = hp.map2alm(Xi, lmax=params_WT.L - 1)  # Its alms
			wlm,clm = pys2.analysis_axisym_lm_wav(X_lm, params_WT.B, params_WT.L, params_WT.Jmin)
			c = hp.alm2map(clm, nside=nside, lmax=params_WT.L - 1, verbose=False) #coeficiente escalar
			w = np.empty([npix, params_WT.J]) #coeficiente wavelet
			for j in range(params_WT.J):
				flm = wlm[:, j].ravel()
				w[:, j] += hp.alm2map(flm, nside=nside, lmax=params_WT.L - 1, verbose=False)	
			c_wt.append(c)
			w_wt.append(w.flatten())
		time0 = time.time()-timei
		print("Finished {0} wavelet transform in: {1:.2f} min".format(params_WT.wtransform,time0/60))
		return np.asarray(w_wt),np.asarray(c_wt)

def error_cl(l_, del_l, fsky, cl_):
    sigma = 2./((2*l_+1)*fsky)
    return np.sqrt(sigma)*cl_


def PSNR(Xi=None,Yi=None): #(p)eak (s)ignal-to-(n)oise (r)atio between the original image X and reconstructed image Y. X,Y = maps (it is not a cube)
	maXi = np.max(Xi)
	miXi = np.min(Xi)
	N   = Xi.size
	var = np.var(Xi-Yi)
	return 20*np.log10((N*(maXi - miXi))/(var))

def SNR(Xobs=None,Xtrue=None): #(s)inal-(n)oise ratio  Xobs = Xtrue + R ---> Xtrue = AS  | Xobs - Xtrue = R
	snr = np.linalg.norm(Xtrue)/np.linalg.norm(Xtrue-Xobs)
	return 10*np.log10(snr)


def extracting_seed_from_string(string=None):
    try:
        num = int(string.split("L")[1].split("M")[0])
    except:
        try:
            num = int(string.split("_")[-1].split("M")[0])
        except:
            num = int(string.split("L")[1].split("_")[0])
    return num
    
def extracting_seed_from_filenames(vectornames=None):
    vectornames = np.asarray(vectornames)
    ind = []
    for n in vectornames:
        ind.append(extracting_seed_from_string(n))
    return np.asarray(ind)
