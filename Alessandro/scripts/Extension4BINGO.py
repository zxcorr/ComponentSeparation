import numpy as np
import pandas as pd
import time,os

'''
This code is a extension for the BINGO Telescope from 
 - GMCA (from Jean Luc Starck, Jerome Bobin and Isabella Carucci 's code ) 
 - (fast)ICA (with scikit-learn code) methods
 
It's a public code, but if you will use it I would like you to quote us:
 - Generalised Morphological Component Analysis applied to HI signals from Intensity Mapping - A. Marins, J. Leite et al
 - Different Component Separation Methods for BINGO Telescope - A. Marins, K. Fornazier et al 

It was create by 
 - Alessandro Marins (USP)
 - Julia Leite (USP)
 - Luiz Henrique Freitas Assis (USP)
 - Karin Fornazier (USP and INPE)
 - Filipe Abdalla (UCL, USP, INPE and RU)

USP  = Universidade de Sao Paulo (Brazil)
INPE = Instituto Nacional de Pesquisas Espaciais (Brazil)
UCL  = University College London (UK)
RU   = Rhodes University (South Africa)
 
I'd like to thank Isabella Carucci and Jerome Bobin for send us your GMCA code used on MeerKAT analysis

Necessary non-native python3 packages:
 - numpy >= 1.19.4
 - pandas >= 0.25.3
 - scikit-learn >= 0.22.2.post1
 - pys2let >= 2.2.1
 - healpy >= 1.13.0
 - astropy >= 4.1
 - gmca4im_lib2 (https://github.com/isab3lla/gmca4im)
 - PyWavelets >= 1.1.1 (Not implemented yet)

At the moment, this code contain:
 (1) Wavelet Transform:
     (1.1) Identity Wavelet Transform
     (1.2) Starlet on the Sphere (Isotropic Undecimated Wavelet Transform by Starck)
     (1.3) Axisymmetric Wavelet Transform on the Sphere (by McEwen)
 (2) Component Separation Method:
     (2.1) Generalized Morphological Component Analysis (GMCA) by Jerome Bobin and Isabella Carucci
     (2.2) Fast Independent Component Analysis (FastICA) by Scikit-Learn python package
     
Last update: April/03/2021

If you have any questions about that, please contact us:
 - alessandro.marins@usp.br
 - karin.fornazier@gmail.com
 - filipe.abdalla@gmail.com
 - julialeite.carvalho@usp.br
 - luiz.henrique.assis@usp.br
'''


########################################################################################################################
# Handling Maps
########################################################################################################################
def remove_mean(map_= None):
	len_nu_ch = np.shape(map_)[0]
	map_ = np.array([[map_[i] - np.mean(map_[i],axis=0)] for i in range(len_nu_ch)])
	return map_[:,0,:]

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

########################################################################################################################
# Component Separation models
########################################################################################################################
def building_dictmaps(w,c,params_WT, dic=None): #dic = dictionary    
    if (params_WT.use_c)*(dic!="identity"):
        J = int(w.shape[1]/params_WT.npix) + 1
        w = include_scalar_coefs(w,c,J-1)
    elif dic=="identity":
        J = 1
        w = w
    else:
        J = int(w.shape[1]/params_WT.npix) + 1
        w = w
    if not "J_types" in params_WT.keys():
        params_WT["J_types"] = pd.Series({dic:J})
        params_WT["Xwt"]     = pd.Series({dic:w})
    else:
        params_WT["J_types"][dic] = J
        params_WT.Xwt[dic]        = w    
    
    
def maps2CSmaps(X, params_WT, params_CS):
	params_WT["nbins"],params_WT["npix"] = X.shape
	##########
	# Starck --> Starlet (Isotropic Undecimated Wavelet Transform)
	if "starlet" in params_WT.wtransform:
		import gmca4im_lib2 as g4i
		w, c = g4i.wavelet_transform(X,params_WT.J)
		building_dictmaps(w,c,params_WT,"starlet")
	##########
	# McEwen --> Axisymmetric Wavelet Transform
	if "axisym" in params_WT.wtransform:
		w, c = coefs_S2LETransform(X,"axisym",params_WT)
		building_dictmaps(w,c,params_WT,"axisym")
	##########
	# McEwen --> Directional Wavelet Transform                
	if "directional" in params_WT.wtransform:
		raise NameError("Not implemented yet: {0}".format(params_WT.wtransform))
	##########
	# PyWavelets python packages
	if "pywavelets" in params_WT.wtransform:
		w,c = PyWavelets4CS(X,params_WT)
		building_dictmaps(w,c,params_WT,"pywavelets") # pywttype = kind of wavelet transf from PyWavelets
	##########
	# Curvelets
	if "curvelets" in params_WT.wtransform:
		raise NameError("Not implemented yet: {0}".format(params_WT.wtransform))
	##########
	# Countourlets
	if "countourlets" in params_WT.wtransform:
		raise NameError("Not implemented yet: {0}".format(params_WT.wtransform))    
	##########
	# Identity   
	if "identity" in params_WT.wtransform:
		building_dictmaps(X,None,params_WT,"identity")
	#####################################
	#dictionary(ies)
	if params_WT.wtransform.size>1:
		return w2CSmaps(overdictionary(params_WT), X, params_WT, params_CS)
	else:
		return w2CSmaps(params_WT.Xwt[0], X, params_WT, params_CS)
        

def w2CSmaps(X_wt, X, params_WT, params_CS):
	print("Component Analysis: Starting {} method ...".format(params_CS.method.upper()))
	time0 = time.time()
	if int(params_CS.division)==1:
		if params_CS.method.upper()=="GMCA":
			import gmca4im_lib2 as g4i
			A,S = g4i.run_GMCA(X_wt, params_CS.A_ini, params_CS.ns, params_CS.mints, params_CS.nmax, params_CS.L0, params_CS.ColFixed, params_CS.whitening, params_CS.epsi, params_CS.verbose)
		elif params_CS.method.upper()=="ICA" or params_CS.method.upper()=="FASTICA":
			from sklearn.decomposition import FastICA
			ica = FastICA(n_components = params_CS.ns, whiten = params_CS.whiten, fun = params_CS.fun, max_iter = params_CS.max_iter, tol = params_CS.tol)
			S = ica.fit_transform(X_wt.T) 
			A = ica.mixing_  
		else:
			raise NameError("Not implemented yet: {0} method".format(params_CS.method.upper()))			
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
			if params_CS.method.upper()=="GMCA":
				import gmca4im_lib2 as g4i
				A,S = g4i.run_GMCA(w_, params_CS.A_ini, params_CS.ns, params_CS.mints, params_CS.nmax, params_CS.L0, params_CS.ColFixed, params_CS.whitening, params_CS.epsi, params_CS.verbose)
			elif params_CS.method.upper()=="ICA" or params_CS.method.upper()=="FASTICA":
				from sklearn.decomposition import FastICA
				ica = FastICA(n_components = params_CS.n_s, whiten = params_CS.whiten, fun = params_CS.fun, max_iter = params_CS.max_iter, tol = params_CS.tol)
				S = ica.fit_transform(w_.T) 
				A = ica.mixing_  
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

########################################################################################################################
# Dictionaries/Riesz Basis: types
########################################################################################################################
def overdictionary(params_WT):
    Xwt=[]
    for i in range(params_WT.nbins):
        Xwt.append(np.vstack([params_WT.Xwt[name][i,:].reshape(-1,params_WT.J_types[name]).T for name in params_WT.wtransform]).T.flatten())
    del params_WT["Xwt"],params_WT["J_types"]
    return np.asarray(Xwt)
def verPyWavelets(params_WT):
## 
# Type of Wavelet Transforms ACCEPTED
##
#haar family: haar
#db family  : db1, db2, db3, db4, db5, db6, db7, db8, db9, db10, db11, db12, db13, db14, db15, db16, db17, db18, db19, db20, db21, 
#             db22, db23, db24, db25, db26, db27, db28, db29, db30, db31, db32, db33, db34, db35, db36, db37, db38
#sym family : sym2, sym3, sym4, sym5, sym6, sym7, sym8, sym9, sym10, sym11, sym12, sym13, sym14, sym15, sym16, sym17, sym18, sym19, sym20
#coif family: coif1, coif2, coif3, coif4, coif5, coif6, coif7, coif8, coif9, coif10, coif11, coif12, coif13, coif14, coif15, coif16, coif17
#bior family: bior1.1, bior1.3, bior1.5, bior2.2, bior2.4, bior2.6, bior2.8, bior3.1, bior3.3, bior3.5, bior3.7, bior3.9, bior4.4, bior5.5, bior6.8
#dmey family: dmey
	import pywt
	db  = np.array(["db"+str(i+1) for i in range(38)])
	sym  = np.array(["sym"+str(i+1) for i in range(20)])
	coif = np.array(["coif"+str(i+1) for i in range(17)])
	bior = np.array(["bior1.1", "bior1.3", "bior1.5", "bior2.2", "bior2.4", "bior2.6", "bior2.8", "bior3.1", "bior3.3", "bior3.5", "bior3.7", "bior3.9", "bior4.4", "bior5.5", "bior6.8"])
	haar = np.array(["haar"])
	dmey = np.array(["dmey"])
	
	if params_WT.pywttype in np.hstack([haar,dmey,db,sym,coif,bior]):
		return pywt.Wavelet(params_WT.pywttype)
	else:
		raise Exception("{} wavelet transform is not accepted".format(params_WT.pywttype.upper()))    
	
def PyWavelets4CS(X,params_WT):
	import pywt
	import time
	wav = verPyWavelets(params_WT)
	timei = time.time()
	print("Starting {} wavelet transform ...".format(params_WT.pywttype))	
	nbins, npix = X.shape
	c_wt = []
	w_wt = []
	for i in range(nbins):
		XWT = np.asarray(pywt.swt(X[i], wav, level=params_WT.Jpwt))
		c_wt.append(XWT[-1,0])            #last scale coef [c_(J-1)], J=0,1,..,J-1
		w_wt.append(XWT[:,1].T.flatten()) # 
	time0 = time.time()-timei
	print("Finished wavelet transform(s) in: {:.2f} min".format(time0/60))		
	return np.asarray(w_wt),np.asarray(c_wt)
		
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

def coefs_S2LETransform(X,type_,params_WT):#return J+1 cubes
	import time
	import healpy as hp
	import pys2let as pys2
	import sys
	timei = time.time()
	print("Starting {} wavelet transform ...".format(type_))
	nbin,npix = X.shape
	nside     = int(hp.npix2nside(npix))
	if params_WT.L==None:
		params_WT.L=3*nside
	Jmax      = pys2.pys2let_j_max(params_WT.B, params_WT.L, params_WT.Jmin)
	params_WT["Jmax"]  = Jmax
	params_WT["J"]  = Jmax - params_WT.Jmin + 1
	c_wt = []
	w_wt = []
	if type_=="axisym":
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
		print("Finished wavelet transform(s) in: {:.2f} min".format(time0/60))
		return np.asarray(w_wt),np.asarray(c_wt)
		
########################################################################################################################
# Coeficients from Dictionaries
########################################################################################################################		
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
    
########################################################################################################################
# Handling files
########################################################################################################################
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
    
def loadtheoricalcls(pathcls=None, type_="auto", masking_effect=False, pathFTmask=None):
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
		lt,clt = clt[:,0],clt[:,cols].T
	else:
		lt,clt = clt[:,0],clt.T
	
	if masking_effect:
		if type(pathFTmask)!=str:
			raise TypeError
		Rll = np.loadtxt(pathFTmask)[:,2]
		n   = int(np.sqrt(Rll.size))
		Rll = Rll.reshape(n,n)
		n   = min(Rll.shape[0],clt.shape[1])
		Rll = Rll[:n,:n]
		clt = clt[:,:n]		
		for i in range(np.shape(clt)[0]):
			clt[i,:] = np.dot(Rll,clt[i,:])
		return lt[:n],clt
	else:
		return lt,clt	
		
########################################################################################################################
# Reconstruction and Residual maps
########################################################################################################################
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
    
########################################################################################################################
# Noises
########################################################################################################################
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
########################################################################################################################
# Statistical Toys
########################################################################################################################
def PSNR(Xi=None,Yi=None): #(p)eak (s)ignal-to-(n)oise (r)atio between the original image X and reconstructed image Y. X,Y = maps (it is not a cube)
	maXi = np.max(Xi)
	miXi = np.min(Xi)
	N   = Xi.size
	var = np.var(Xi-Yi)
	return 20*np.log10((N*(maXi - miXi))/(var))

def SNR(Xobs=None,Xtrue=None): #(s)inal-(n)oise ratio  Xobs = Xtrue + R ---> Xtrue = AS  | Xobs - Xtrue = R
	snr = np.linalg.norm(Xtrue)/np.linalg.norm(Xtrue-Xobs)
	return 10*np.log10(snr)

########################################################################################################################
# Noise Debias
########################################################################################################################
def noisedebiasing(Cls_ = None, seed_used = None, dir_hi = None, dir_prior = None, dir_noise = None , dir_pure = None, dir_projpure = None, dir_projnoise = None, dir_projprior = None, type_ = None): #Cls=dictionary with all of components #type_=filipe or mathieu
	import copy
	cls_ = copy.deepcopy(Cls_)
	if type_=="filipe":
		if not type(cls_)==dict:
			raise Exception("Cls is not a dictionary!")
		L0     = "L{}".format(seed_used)
		cls_L0 = cls_[dir_hi][L0]
		for k in cls_.keys():
			del cls_[k][L0]
		for i, Li in enumerate(cls_[dir_hi].keys()):
			if i==0:
				S         = cls_[dir_hi][Li]/cls_[dir_prior][Li]
				cls_noise = cls_[dir_noise][Li]
			else:
				S         += cls_[dir_hi][Li]/cls_[dir_prior][Li]
				cls_noise += cls_[dir_noise][Li]
		S         = S/(len(cls_[dir_hi].keys()))
		cls_noise = cls_noise/(len(cls_[dir_hi].keys()))
		cls_ndb   = (cls_L0/S) - cls_noise
		del cls_, cls_L0
		return {"Cls_debias":cls_ndb,"S":S,"Cls_noise":cls_noise}
	elif type_=="mathieu":
		if not type(cls_)==dict:
			raise Exception("Cls is not a dictionary!")
		L0     = "L{}".format(seed_used)
		cls_L0 = cls_[dir_hi][L0]
		for k in cls_.keys():	
			del cls_[k][L0]
		for i, Li in enumerate(cls_[dir_hi].keys()):
			if i==0:
				S             = cls_[dir_projpure][Li]/cls_[dir_pure][Li]
				cls_projnoise = cls_[dir_projnoise][Li]
			else:
				S             += cls_[dir_projpure][Li]/cls_[dir_pure][Li]
				cls_projnoise += cls_[dir_projnoise][Li]
		S             = S/(len(cls_[dir_hi].keys()))
		cls_projnoise = cls_projnoise/(len(cls_[dir_hi].keys()))
		cls_ndb       = (cls_L0 - cls_projnoise)/S
		del cls_, cls_L0
		return {"Cls_debias":cls_ndb,"S":S,"Cls_noise":cls_projnoise}        

########################################################################################################################
# Error Functions
########################################################################################################################
def error_cl( cl_,l_,params_plot):
    sigma = 2./((2*l_+1)*params_plot.fsky*params_plot.del_l)
    return np.sqrt(sigma)*cl_

def diff_realisation_Cls_1map(Cls_=None,params_plot=None):
    import copy
    cls_ = copy.deepcopy(Cls_)
    del cls_[params_plot.L0]
    for i,li in enumerate(cls_.keys()):
        c_ = copy.deepcopy(cls_[li][params_plot.nch])
        l_ = np.arange(len(c_))
        if params_plot.del_l>1:
            l_,c_ = clsbinned(c_, del_l=params_plot.del_l, l0 = params_plot.l0)
        if i==0:
            cls_inu = c_
        else:
            cls_inu = np.vstack((cls_inu, c_))
    return l_, cls_inu

def varCls1bin(cls1bin=None):
    return np.array([np.var(cls1bin[:,i]) for i in range(cls1bin.shape[1])])
    
def errorCls(Cls_,params_plot, type_=["21cm","noise"]):#, error=["realisation","cosmic_variance"]):
    for i, t in enumerate(type_):
        l_, cls_ = diff_realisation_Cls_1map(Cls_[t], params_plot)
        if i==0:
            error_realisation     = varCls1bin(cls_)
            l_,c_ = clsbinned(Cls_[t][params_plot.L0][params_plot.nch], del_l=params_plot.del_l, l0 = params_plot.l0)
            error_cosmic_variance = error_cl(c_, l_, params_plot)
        else:
            error_realisation += varCls1bin(cls_)
            l_,c_ = clsbinned(Cls_[t][params_plot.L0][params_plot.nch], del_l=params_plot.del_l, l0 = params_plot.l0)
            error_cosmic_variance += error_cl(c_, l_, params_plot)
    return error_realisation + error_cosmic_variance
########################################################################################################################
# Realisation names/numbers
########################################################################################################################
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
########################################################################################################################
# Cls Binned
########################################################################################################################    
def index_cls_binned(l_,lbin_):
    inds = []
    for il in lbin_:
        i = np.where(l_==il)[0][0]
        inds.append(i)
    return np.asarray(inds)

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
