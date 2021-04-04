import os,sys,re
import numpy as np
import healpy as hp
import pandas as pd
import astropy.io.fits as fits
import Extension4BINGO as cs
import time
import argparse, json


###################################################################
# Check the python version and import configparser
###################################################################

if sys.version_info[0]==2:
	import ConfigParser
	config = ConfigParser.RawConfigParser()
elif sys.version_info[0]==3:
	import configparser
	config = configparser.ConfigParser()

###################################################################
# This part is for extracting information from parameters_cs.ini file
###################################################################
#timei       = time()
INI         = "parameters_cs.ini"
config.read(os.path.join(os.getcwd(),INI))

#General variables
method         = config.get(       "General","method")
wtransform     = config.get(       "General","wtransform")
maps_wout_mean = config.getboolean("General","maps_wout_mean")
apply_mask     = config.getboolean("General","apply_mask")
add_noise      = config.getboolean("General","add_noise")

try:
	wtransform  = json.loads(config.get(       "General","wtransform"))
	for i in range(len(wtransform)): 
		wtransform[i]=str(wtransform[i])
except:
	raise NameError
#Variables of the dictionary(ies)
J        = config.getint(    "Dictionary","J")
use_c    = config.getboolean("Dictionary","use_c")
L        = config.get(       "Dictionary","L")
J_min    = config.getint(    "Dictionary","J_min")
B        = config.getint(    "Dictionary","B")
N        = config.getint(    "Dictionary","N")
spin     = config.getint(    "Dictionary","spin")
upsample = config.getint(    "Dictionary","upsample")
Jpwt     = config.getint(    "Dictionary","Jpwt")
pywttype = config.get(       "Dictionary","pywttype")
try:
	L = int(L)
except:
	L = None

#Variables of the ComponentSeparation (At the moment, in this code are only GMCA and (fast)ICA methods of CS)
n_s          = config.getint(    "ComponentSeparation","n_s")
mints        = config.getfloat(  "ComponentSeparation","mints")
nmax         = config.getint(    "ComponentSeparation","nmax")
L0           = config.getint(    "ComponentSeparation","L0")
whitening    = config.getboolean("ComponentSeparation","whitening")
epsi         = config.getfloat(  "ComponentSeparation","epsi")
verbose      = config.getboolean("ComponentSeparation","verbose")
whiten       = config.getboolean("ComponentSeparation","whiten")
function     = config.get(       "ComponentSeparation","function")
max_iter     = config.getint(    "ComponentSeparation","max_iter")
tolerance    = config.getfloat(  "ComponentSeparation","tolerance")
div          = config.getint(    "ComponentSeparation","div")
without_covx = config.getboolean("ComponentSeparation","without_covx")
try:
	AInit  = np.asarray(json.loads(config.get( "ComponentSeparation", "AInit")))
except:
	AInit=None
try:
	ColFixed  = np.asarray(json.loads(config.get( "ComponentSeparation", "ColFixed")))
except:
	ColFixed=None

#Variable of the Realisation
seed_used  = config.getint("Realisation","seed_used")
#Paths
pathout       = config.get("Paths","pathout")
cl_type_save  = config.get("Paths","cl_type_save")
dir_observed  = config.get("Paths","dir_observed")
dir_mask      = config.get("Paths","dir_mask")
dir_prior     = config.get("Paths","dir_prior")
dir_noise     = config.get("Paths","dir_noise")
dir_pure      = config.get("Paths","dir_pure")
dir_projprior = config.get("Paths","dir_projprior")
dir_projnoise = config.get("Paths","dir_projnoise")
dir_projpure  = config.get("Paths","dir_projpure")	
#File(s)
name_mask  = config.get("Files","name_mask")

###############################################################################
# You can modify any options in the parameters.ini file by the command terminal
###############################################################################

parser = argparse.ArgumentParser(description='Modify by the command terminal parameters in parameters_cs.ini file')

parser.add_argument('--method'        , action = 'store', dest = 'method'        , default = method ,         help = 'If you can choice between GMCA and ICA methods.')
parser.add_argument('--wtransform'    , action = 'store', dest = 'wtransform'    , default = wtransform,      help = 'Type of wavelet transform(s).', type=str)
parser.add_argument('--maps_wout_mean', action = 'store', dest = 'maps_wout_mean', default = maps_wout_mean , help = 'Extracting mean value of each map.')
parser.add_argument('--apply_mask'    , action = 'store', dest = 'apply_mask'    , default = apply_mask ,     help = 'Applying mask to the maps.')
parser.add_argument('--add_noise'     , action = 'store', dest = 'add_noise'     , default = add_noise,       help = 'Add noise maps to the observed maps.')
parser.add_argument('--J'             , action = 'store', dest = 'J'             , default =J ,               help = 'Number of wavelet transform scales (it\'s just for starlet).')
parser.add_argument('--use_c'         , action = 'store', dest = 'use_c'         , default =use_c ,           help = 'Using scale wavelet tranform from wavelet maps.')
parser.add_argument('--L'             , action = 'store', dest = 'L'             , default =L ,               help = 'Band-limit value: L!=integer number, value which will use will be 3*nside (it\'s just for S2LET transfoms).')
parser.add_argument('--J_min'         , action = 'store', dest = 'J_min'         , default =J_min ,           help = 'Minimum scale (it\'s just for S2LET transfoms).')
parser.add_argument('--B'             , action = 'store', dest = 'B'             , default =B ,               help = 'B define range of scales: in j-scale it is [B^(j-1),B^(j+1)] (it\'s just for S2LET transfoms).')
parser.add_argument('--N'             , action = 'store', dest = 'N'             , default =N ,               help = 'Number of directions (for wavelet function with s!=0).')
parser.add_argument('--spin'          , action = 'store', dest = 'spin'          , default =spin ,            help = 'Spin of the signals. Set spin=0 for temperature.')
parser.add_argument('--upsample'      , action = 'store', dest = 'upsample'      , default =upsample ,        help = 'Set: 1 (means all scales at full resolution L) or 0 (means multiresolution wavelet transform).')
parser.add_argument('--Jpwt'          , action = 'store', dest = 'Jpwt'          , default =Jpwt ,            help = 'Number of wavelet transform scales (it\'s just for PyWavelet).')
parser.add_argument('--pywttype'      , action = 'store', dest = 'pywttype'      , default =pywttype ,        help = 'Type of PyWavelet Transform for Redundant algorithm (undecimated).')
parser.add_argument('--n_s'           , action = 'store', dest = 'n_s'           , default =n_s ,             help = 'Number of sources.')
parser.add_argument('--mints'         , action = 'store', dest = 'mints'         , default =mints ,           help = 'Minimun threshold.')
parser.add_argument('--nmax'          , action = 'store', dest = 'nmax'          , default =nmax ,            help = 'Number of iterations (usually 100 is safe).')
parser.add_argument('--L0'            , action = 'store', dest = 'L0'            , default =L0 ,              help = 'Switch between L0 norm (1) or L1 norm (0) to constrain in regression analysis.')
parser.add_argument('--whitening'     , action = 'store', dest = 'whitening'     , default =whitening ,       help = '(GMCA) Whiten the data. I.e., the cov matrix of the observed data will be identity.')
parser.add_argument('--epsi'          , action = 'store', dest = 'epsi'          , default =epsi ,            help = '(GMCA) Tolerance.')
parser.add_argument('--verbose'       , action = 'store', dest = 'verbose'       , default =verbose ,         help = '')
parser.add_argument('--whiten'        , action = 'store', dest = 'whiten'        , default =whiten ,          help = '(ICA) Whitening the data matrix.')
parser.add_argument('--function'      , action = 'store', dest = 'function'      , default =function ,        help = '(ICA) Function for negentropy analysis')
parser.add_argument('--max_iter'      , action = 'store', dest = 'max_iter'      , default =max_iter ,        help = '(ICA) Maximum number of iterations')
parser.add_argument('--tolerance'     , action = 'store', dest = 'tolerance'     , default =tolerance ,       help = '(ICA) Level of tolerance')
parser.add_argument('--div'           , action = 'store', dest = 'div'           , default =div ,             help = 'Partial GMCA analysis for div number of maps.')
parser.add_argument('--without_covx'  , action = 'store', dest = 'without_covx'  , default =without_covx ,    help = 'Using or not cov matrix ponderation in mixmatrix estimation.')
parser.add_argument('--seed_used'     , action = 'store', dest = 'seed_used'     , default =seed_used ,       help = 'Realisation used.')
parser.add_argument('--pathout'       , action = 'store', dest = 'pathout'       , default =pathout ,         help = 'Path where will write the cls outputs.')
parser.add_argument('--cl_type_save'  , action = 'store', dest = 'cl_type_save'  , default =cl_type_save ,    help = 'cls from _ reconstrution _ or _ residuous _ type.')
parser.add_argument('--dir_observed'  , action = 'store', dest = 'dir_observed'  , default =dir_observed ,    help = 'Path of the observed directory')
parser.add_argument('--dir_mask'      , action = 'store', dest = 'dir_mask'      , default =dir_mask ,        help = 'Path of the mask directory')
parser.add_argument('--dir_prior'     , action = 'store', dest = 'dir_prior'     , default =dir_prior ,       help = 'Path of the prior directory')
parser.add_argument('--dir_noise'     , action = 'store', dest = 'dir_noise'     , default =dir_noise ,       help = 'Path of the noise directory')
parser.add_argument('--dir_pure'      , action = 'store', dest = 'dir_pure'      , default =dir_pure ,        help = 'Path of the pure directory')
parser.add_argument('--dir_projprior' , action = 'store', dest = 'dir_projprior' , default =dir_projprior ,   help = '')
parser.add_argument('--dir_projnoise' , action = 'store', dest = 'dir_projnoise' , default =dir_projnoise ,   help = '')
parser.add_argument('--dir_projpure'  , action = 'store', dest = 'dir_projpure'  , default =dir_projpure ,    help = '')
parser.add_argument('--name_mask'     , action = 'store', dest = 'name_mask'     , default =name_mask ,       help = 'Mask name')
#parser.add_argument('--AInit', action = 'store', dest = 'AInit', default =AInit , help = '!=None if you guess for the mix matrix')
#parser.add_argument('--ColFixed', action = 'store', dest = 'ColFixed', default =ColFixed , help = 'You can impose a column of the mixing matrix')

###############################################################################
# Variables
###############################################################################
arguments = parser.parse_args()
method         = str(arguments.method)
wtransform     = np.asarray(arguments.wtransform.split(",")) if type(arguments.wtransform)==str else np.asarray(arguments.wtransform)
maps_wout_mean = bool(arguments.maps_wout_mean)
apply_mask     = bool(arguments.apply_mask) 
add_noise      = bool(arguments.add_noise)
J        = int(arguments.J)
use_c    = bool(arguments.use_c)
try:
	L = int(arguments.L)
except:
	L = None
J_min    = int(arguments.J_min)
B        = int(arguments.B)
N        = int(arguments.N)
spin     = int(arguments.spin)
upsample = int(arguments.upsample)
Jpwt     = int(arguments.Jpwt)
pywttype = str(arguments.pywttype)
n_s      = int(arguments.n_s)
mints    = int(arguments.mints)
nmax     = int(arguments.nmax)
L0       = int(arguments.L0)
#AInit  = list(arguments.AInit)
#ColFixed = list(arguments.ColFixed)
whitening = bool(arguments.whitening)
epsi      = float(arguments.epsi)
verbose   = bool(arguments.verbose)
div       = int(arguments.div)
whiten    = bool(arguments.whiten)
fun       = str(arguments.function)
max_iter  = int(arguments.max_iter)
tol       = float(arguments.tolerance)

without_covx = bool(arguments.without_covx)
seed_used    = int(arguments.seed_used)
pathout      = str(arguments.pathout)
cl_type_save = str(arguments.cl_type_save)
dir_observed = str(arguments.dir_observed)
dir_mask     = str(arguments.dir_mask)
dir_pure     = str(arguments.dir_pure)
dir_prior    = str(arguments.dir_prior)
dir_noise    = str(arguments.dir_noise)
dir_projpure  = str(arguments.dir_projpure)
dir_projprior = str(arguments.dir_projprior)
dir_projnoise = str(arguments.dir_projnoise)
name_mask     = str(arguments.name_mask)

####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################

params_maps = pd.Series({"without_mean":maps_wout_mean, "apply_mask":apply_mask, "add_noise":add_noise, "cl_type_save":cl_type_save})
params_CS   = pd.Series({"method":method,
                         "A_ini":AInit, "ColFixed":ColFixed, "whitening":whitening, "epsi":epsi, "verbose":verbose, "ns":n_s, "mints":mints,"nmax":nmax, "L0":L0, "division":div, "without_covx":without_covx, "seed_used":seed_used,
                         "whiten":whiten, "fun":fun, "max_iter":max_iter, "tol":tol})
params_WT   = pd.Series({"wtransform":np.asarray(wtransform), "use_c":use_c, "J":J, 
                         "L":L, "Jmin":J_min, "B": B, "N":N, "spin":spin, "upsample":upsample,
                         "Jpwt":Jpwt, "pywttype":pywttype.lower()})
params_path = pd.Series({"pathout":pathout, "dir_observed":dir_observed, "dir_mask":dir_mask, "dir_noise":dir_noise, "dir_prior":dir_prior,"dir_pure":dir_pure, "name_mask":name_mask})

####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
print("####################################################################")
print("# This code was developed by the BINGO project using GMCA adapted  #")
print("####################################################################\n")
print("Starting to run Noise Debias Analysis from the GMCA as Component Separation method.\n\n")
print("####################################################################")
print("The parameters that will be used are:\n")
for p in [params_maps,params_CS,params_WT,params_path] :
	for iname in p.keys():
		if type(p[iname])==int:
			s = "int"
		elif type(p[iname])==float:
			s="float"
		elif type(p[iname])==type(None):
			s="None"
		elif type(p[iname])==bool:
			s="bool"
		elif type(p[iname])==str:
			s="str"
		else:
			s="other"
		print("{} ({}): {}".format(iname,s,p[iname]))
print("####################################################################\n\n")
#####################################################################################################################################################################
#####################################################################################################################################################################
#####################################################################################################################################################################
### Observed
print("Building cls of the observed maps")
###
names = np.asarray(os.listdir(params_path.dir_observed))
nseed = cs.extracting_seed_from_filenames(vectornames=names)
timei = time.time()
params_maps["getdata"] = "observed"
subdirs = cs.checkdir(params_path.pathout, subdirs=["21cm","foregrounds","mixmatrix"])
for i,iname in enumerate(names):
    time0 = time.time()
    params_path["name_observed"] = iname
    X = cs.getmaps(params_maps, params_path)
    X = cs.adaptation_maps(X, params_maps, params_path)
    X = cs.maps2CSmaps(X, params_WT, params_CS)
    params_maps["iseed"]="L"+str(nseed[i])
    cs.saveouts(mrec=X, params_path=params_path, params_maps=params_maps, params_WT=params_WT, params_CS=params_CS, subdirs=subdirs)
    del X
    time0   = time.time()-time0
    print("Loop time {0}: {1:.2f} min\n".format(i,time0/60))
time0 = time.time()-timei
print("Completed in: {:.2f} min".format(time0/60))

### Noise
print("\n\nBuilding cls of the noise maps")
###
names = np.asarray(os.listdir(params_path.dir_noise))
nseed = cs.extracting_seed_from_filenames(vectornames=names)
timei   = time.time()
params_maps["getdata"] = "noise"
subdirs = ["noise"]
if not os.path.isdir(os.path.join(params_path.pathout,subdirs[0])):
    os.makedirs(os.path.join(params_path.pathout,subdirs[0]))
for i,iname in enumerate(names):
    time0 = time.time()
    params_path["name_noise"] = iname
    X = cs.getmaps(params_maps, params_path)
    X = cs.adaptation_maps(X, params_maps, params_path)
    params_maps["iseed"]="L"+str(nseed[i])
    cs.saveouts(mrec=X, params_path=params_path, params_maps=params_maps, params_WT=params_WT, params_CS=params_CS, subdirs=subdirs)    
    time0 = time.time()-time0
    print("[L{0}] Loop time: {1:.2f} min".format(nseed[i],time0/60))
time0 = time.time()-timei
print("Completed in: {:.2f} min".format(time0/60))

###Prior
print("\nBuilding cls of the prior maps")
###
names = np.asarray(os.listdir(params_path.dir_prior))
#nseed = np.asarray([iname.split("_")[-1].split(".")[0][:3] for iname in names])#selecting L-number from string name
nseed = cs.extracting_seed_from_filenames(vectornames=names)
timei   = time.time()
params_maps["getdata"] = "prior"
subdirs = ["prior"]
if not os.path.isdir(os.path.join(params_path.pathout,subdirs[0])):
    os.makedirs(os.path.join(params_path.pathout,subdirs[0]))
for i,iname in enumerate(names):
    time0 = time.time()
    params_path["name_prior"] = iname
    X = cs.getmaps(params_maps, params_path)
    X = cs.adaptation_maps(X, params_maps, params_path)
    params_maps["iseed"]="L"+str(nseed[i])
    cs.saveouts(mrec=X, params_path=params_path, params_maps=params_maps, params_WT=params_WT, params_CS=params_CS, subdirs=subdirs)    
    time0 = time.time()-time0
    print("[L{0}] Loop time: {1:.2f} min".format(nseed[i],time0/60))
time0 = time.time()-timei
print("Completed in: {:.2f} min\n".format(time0/60))

###Pure (21cm)
print("\nBuilding cls of the pure (21cm) maps")
###
names = np.asarray(os.listdir(params_path.dir_pure))
nseed = cs.extracting_seed_from_filenames(vectornames=names)
timei   = time.time()
params_maps["getdata"] = "pure"
subdirs = ["pure"]
if not os.path.isdir(os.path.join(params_path.pathout,subdirs[0])):
    os.makedirs(os.path.join(params_path.pathout,subdirs[0]))
for i,iname in enumerate(names):
    time0 = time.time()
    params_path["name_pure"] = iname
    X = cs.getmaps(params_maps, params_path)
    X = cs.adaptation_maps(X, params_maps, params_path)
    params_maps["iseed"]="L"+str(nseed[i])
    cs.saveouts(mrec=X, params_path=params_path, params_maps=params_maps, params_WT=params_WT, params_CS=params_CS, subdirs=subdirs)    
    time0 = time.time()-time0
    print("[L{0}] Loop time: {1:.2f} min".format(nseed[i],time0/60))
time0 = time.time()-timei
print("Completed in: {:.2f} min".format(time0/60))

### projnoise
print("\nBuilding cls of the noise residual maps")
###
names = np.asarray(os.listdir(params_path.dir_noise))
nseed = cs.extracting_seed_from_filenames(vectornames=names)
timei   = time.time()
params_maps["getdata"] = "noise"
subdirs = ["projnoise"]
L0      = "L{}".format(params_CS.seed_used)
A       = cs.loadmixmatrix(params_path.pathout,"mixmatrix")
if not os.path.isdir(os.path.join(params_path.pathout,subdirs[0])):
    os.makedirs(os.path.join(params_path.pathout,subdirs[0]))
for i,iname in enumerate(names):
    time0 = time.time()
    params_path["name_noise"] = iname
    X = cs.getmaps(params_maps, params_path)
    X = cs.adaptation_maps(X, params_maps, params_path)
    params_maps["iseed"]="L"+str(nseed[i])
    cs.saveouts(mrec=X, A=A[L0], params_path=params_path, params_maps=params_maps, params_WT=params_WT, params_CS=params_CS, subdirs=subdirs)    
    time0 = time.time()-time0
    print("[L{0}] Loop time: {1:.2f} min".format(nseed[i],time0/60))
time0 = time.time()-timei
print("Completed in: {:.2f} min".format(time0/60))


###projprior
print("\nBuilding cls of the prior residual maps")
###
names   = np.asarray(os.listdir(params_path.dir_prior))
nseed = cs.extracting_seed_from_filenames(vectornames=names)
timei   = time.time()
params_maps["getdata"] = "prior"
subdirs = ["projprior"]
L0      = "L{}".format(params_CS.seed_used)
A       = cs.loadmixmatrix(params_path.pathout,"mixmatrix")
if not os.path.isdir(os.path.join(params_path.pathout,subdirs[0])):
    os.makedirs(os.path.join(params_path.pathout,subdirs[0]))
for i,iname in enumerate(names):
    time0 = time.time()
    params_path["name_prior"] = iname
    X = cs.getmaps(params_maps, params_path)
    X = cs.adaptation_maps(X, params_maps, params_path)
    params_maps["iseed"]="L"+str(nseed[i])
    cs.saveouts(mrec=X, A=A[L0], params_path=params_path, params_maps=params_maps, params_WT=params_WT, params_CS=params_CS, subdirs=subdirs)    
    time0 = time.time()-time0
    print("[L{0}] Loop time: {1:.2f} min".format(nseed[i],time0/60))
time0 = time.time()-timei
print("Completed in: {:.2f} min".format(time0/60))


### projpure
print("\nBuilding cls of the pure residual maps")
###
names = np.asarray(os.listdir(params_path.dir_pure))
nseed = cs.extracting_seed_from_filenames(vectornames=names)
timei   = time.time()
params_maps["getdata"] = "pure"
subdirs = ["projpure"]
L0      = "L{}".format(params_CS.seed_used)
A       = cs.loadmixmatrix(params_path.pathout,"mixmatrix")
if not os.path.isdir(os.path.join(params_path.pathout,subdirs[0])):
    os.makedirs(os.path.join(params_path.pathout,subdirs[0]))
for i,iname in enumerate(names):
    time0 = time.time()
    params_path["name_pure"] = iname
    X = cs.getmaps(params_maps, params_path)
    X = cs.adaptation_maps(X, params_maps, params_path)
    params_maps["iseed"]="L"+str(nseed[i])
    cs.saveouts(mrec=X, A=A[L0], params_path=params_path, params_maps=params_maps, params_WT=params_WT, params_CS=params_CS, subdirs=subdirs)    
    time0 = time.time()-time0
    print("[L{0}] Loop time: {1:.2f} min".format(nseed[i],time0/60))
time0 = time.time()-timei
print("Completed in: {:.2f} min".format(time0/60))

import subprocess
subprocess.Popen('cp parameters_cs.ini'+str(pathout), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
