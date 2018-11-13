import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import os
from uncertainties import ufloat, unumpy
from uncertainties.umath import *  # sin(), etc
from scipy.ndimage.filters import gaussian_filter
from StringIO import StringIO

sns.set()

sns.set_context("paper")
# sns.set(font_scale=1.2) # Bigger than normal fonts
sns.set_style("ticks", { 'axes.grid': True})
plt.rcParams['legend.loc'] = 'best'

cwd = os.getcwd()
###########

def rhoCT(Ex,T,E0):
	E = Ex-E0
	return exp(E/T)/T

from decimal import Decimal
print '%.2E' % Decimal(rhoCT(6.5,T=0.425,E0=-0.456))
# print rhoCT(6.5,T=0.44,E0=-0.456)



###########
# get calibration from counting.dat
def getCalibrationFromCounting(filename):
    f = open(filename)
    lines = f.readlines()
    #19: float a0 =  -0.7800;
    #20: float a1 =   0.1300;
    cal = np.genfromtxt(StringIO(lines[18]),dtype=object, delimiter="=")
    if cal[0]!="float a0 ":
        raise ValueError("Could not read calibration")
    a0 = float(cal[1][:-1])
    cal = np.genfromtxt(StringIO(lines[19]),dtype=object, delimiter="=")
    if cal[0]!="float a1 ":
        raise ValueError("Could not read calibration")
    a1 = float(cal[1][:-1])
    return a0, a1

# transform strength.nrm to f
def convertStrength(filename,a0_strength, a1_strength):
    data_ocl = np.loadtxt(filename)
    data_ocl_copy = data_ocl
    n_datapoints = int(len(data_ocl_copy)/2)
    data_ocl = np.ndarray((n_datapoints,3),dtype=float)
    for i in range(n_datapoints):
        if ( data_ocl_copy[i] > 0 ): 
            data_ocl[i,0] = a0_strength + (a1_strength*i)
        else:
            data_ocl[i,0] = 0
        data_ocl[i,1] = data_ocl_copy[i]
        data_ocl[i,2] = data_ocl_copy[i+n_datapoints]
    data_ocl = data_ocl[np.all(data_ocl,axis=1)] # delete "0" elments
    return data_ocl

# Extrapolation of the transmission coefficient
def getTransExt(myfile, a0_strength, a1_strength, Emin, Emax):
	transextfile = open(myfile)
	transextlines = transextfile.readlines()
	transextfile.close()
	Ntrans = len(transextlines)
	trans = np.zeros((Ntrans,2))
	for i in range(Ntrans):
		trans[i,0] = a0_strength + i*a1_strength
		trans[i,1] = float(transextlines[i].split()[0])/(2.*np.pi*trans[i,0]**3.)
	trans = trans[np.all((trans[:,0]>Emin,trans[:,0]<Emax), axis = 0),:]
	return trans

# OCL_singleJ_strength = convertStrength(cwd+"/Jint_singleJ/strength.nrm")
# OCL_singleJ_trans = getTransExt("/Jint_singleJ/transext.nrm", Emin=0.1, Emax=8.)
# OCL_singleJ = {'strength': OCL_singleJ_strength, 
# 			   'trans': OCL_singleJ_trans,
# 			    'label':"J_pop=3 (less statistics)"}

def ReadFiles(folder, label):
    a0_strength, a1_strength = getCalibrationFromCounting(folder+"/counting.cpp")
    print "Reading {0}\nwith calibration a0={1:.3e}, a1 ={2:.3e}".format(folder, a0_strength, a1_strength)
    strength = convertStrength(folder+"/strength.nrm",a0_strength, a1_strength)
    trans = getTransExt(folder+"/transext.nrm", a0_strength, a1_strength, Emin=0.1, Emax=8.)
    nld = convertStrength(folder+"/rhopaw.cnt",a0_strength, a1_strength)
    data = {'strength': strength, 
    		 'trans': trans,
    	     'nld': nld,
    		 'label':label}
    return data

OCL_Potel_rhotot = ReadFiles(cwd+"/Jint_Greg_parity_rhotot","Potel_rhotot")
OCL_Potel = ReadFiles(cwd+"/Jint_Greg","Potel_r30")
OCL_Potelr10 = ReadFiles(cwd+"/Jint_Greg_r10_stripNLD","Potel_r10")
OCL_EB05 = ReadFiles(cwd+"/Jint_EB06","EB05")		   

# Make x-axis array to plot from
# Earray = np.linspace(0,20,800)
###############################
# Initialize figure
# plt.figure()
fig= plt.figure("NLD")
ax = fig.add_subplot(211)
ax.set_yscale("log", nonposy='clip')

ax.tick_params("x", top="off")
ax.tick_params("y", right="off")

def plotData(data, dicEntry, fmt='v-'):
	try:
		plot = plt.errorbar(data[dicEntry][:,0], data[dicEntry][:,1], yerr=data[dicEntry][:,2], markersize=4, linewidth=1.5, fmt=fmt, label=data["label"])
	except (IndexError):
		# plot = plt.errorbar(data[dicEntry][:,0], data[dicEntry][:,1], markersize=4, linewidth=1.5, fmt=fmt, label=data["label"])
		plot = plt.errorbar(data[dicEntry][:,0], data[dicEntry][:,1], markersize=4, linewidth=1.5, fmt=fmt)

	return plot

NLD_true_disc = np.loadtxt("misc/NLD_exp_disc.dat")
NLD_true_cont = np.loadtxt("misc/NLD_exp_cont.dat")
# apply same binwidth to continuum states
binwidth_goal = NLD_true_disc[1,0]-NLD_true_disc[0,0]
binwidth_cont = NLD_true_cont[1,0]-NLD_true_cont[0,0]
Emax = NLD_true_cont[-1,0]
nbins = int(np.ceil(Emax/binwidth_goal))
Emax_adjusted = binwidth_goal*nbins # Trick to get an integer number of bins
bins = np.linspace(0,Emax_adjusted,nbins+1)
hist, edges = np.histogram(NLD_true_cont[:,0],bins=bins,weights=NLD_true_cont[:,1]*binwidth_cont)
NLD_true = np.zeros((nbins,2))
NLD_true[:nbins,0] = bins[:nbins]
NLD_true[:,1] = hist/binwidth_goal
NLD_true[:len(NLD_true_disc),1] += NLD_true_disc[:,1]

OCL_Potel_nld = plotData(OCL_Potelr10, dicEntry="nld", fmt="v--")
OCL_Potel_nld = plotData(OCL_Potel, dicEntry="nld", fmt="v--")
OCL_Potel_rhotot_nld = plotData(OCL_Potel_rhotot, dicEntry="nld")
OCL_EB05_nld = plotData(OCL_EB05, dicEntry="nld")
plt.step(np.append(-binwidth_goal,NLD_true[:-1,0])+binwidth_goal/2.,np.append(0,NLD_true[:-1,1]), "k", where="pre",label="input NLD, binned")

handles, labels = ax.get_legend_handles_labels()
lgd1= ax.legend(handles, labels)

plt.xlim(0,7)
plt.xlabel(r'$E_x$ [MeV]',fontsize="medium")
plt.ylabel(r'levels/MeV',fontsize="medium")

#############################
# fig= plt.figure("Ratios NLD")
ax2 = fig.add_subplot(212, sharex=ax)
# ax.set_yscale("log", nonposy='clip')

ax2.tick_params("x", top="off")
ax2.tick_params("y", right="off")

def calcRatio(set1, set2, attribute):
	set1un = unumpy.uarray(set1[attribute][:,1],std_devs=set1[attribute][:,2])
	set2un = unumpy.uarray(set2[attribute][:,1],std_devs=set2[attribute][:,2])
	return set1un/set2un

def calcRatioTrue(dic1, true, attribute):
	un1 = unumpy.uarray(dic1[attribute][:,1],std_devs=dic1[attribute][:,2])
	true_interpolate = np.interp(dic1[attribute][:,0],true[:,0],true[:,1])
	return un1/true_interpolate

ratio = calcRatioTrue(OCL_Potel,NLD_true, "nld")
ratio_plot = plt.errorbar(OCL_Potel["nld"][:,0], unumpy.nominal_values(ratio), yerr=unumpy.std_devs(ratio), markersize=4, linewidth=1.5, fmt='v--', color="grey", label=OCL_Potel["label"])
ratio = calcRatioTrue(OCL_Potel_rhotot,NLD_true, "nld")
ratio_plot = plt.errorbar(OCL_Potel_rhotot["nld"][:,0], unumpy.nominal_values(ratio), yerr=unumpy.std_devs(ratio), markersize=4, linewidth=1.5, fmt='.-', color="black", label=OCL_Potel_rhotot["label"])
ratio = calcRatioTrue(OCL_EB05,NLD_true, "nld")
ratio_plot = plt.errorbar(OCL_EB05["nld"][:,0], unumpy.nominal_values(ratio), yerr=unumpy.std_devs(ratio), markersize=4, linewidth=1.5, fmt='v-', color="green", label=OCL_EB05["label"])

# # gaussian smoothing of ratios
# # approximated (!) variance reduction by gaussian filter; https://dsp.stackexchange.com/questions/26859/how-will-a-gaussian-blur-affect-image-variance
# sigma_smooth = 1.
# uncReduction = 2*sqrt(np.pi)*sigma_smooth
# ratio = calcRatioTrue(OCL_Potel,NLD_true, "nld")
# ratio_plot = plt.errorbar(OCL_Potel["nld"][:,0], gaussian_filter(unumpy.nominal_values(ratio),sigma=sigma_smooth), yerr=unumpy.std_devs(ratio)/uncReduction, markersize=4, linewidth=1.5, fmt='v--', color="grey", label=OCL_Potel["label"]+", smoothed")
# ratio = calcRatioTrue(OCL_Potel_rhotot,NLD_true, "nld")
# ratio_plot = plt.errorbar(OCL_Potel_rhotot["nld"][:,0], gaussian_filter(unumpy.nominal_values(ratio),sigma=sigma_smooth), yerr=unumpy.std_devs(ratio)/uncReduction, markersize=4, linewidth=1.5, fmt='.-', color="black", label=OCL_Potel_rhotot["label"]+", smoothed")
# ratio = calcRatioTrue(OCL_EB05,NLD_true, "nld")
# ratio_plot = plt.errorbar(OCL_EB05["nld"][:,0], gaussian_filter(unumpy.nominal_values(ratio),sigma=sigma_smooth), yerr=unumpy.std_devs(ratio)/uncReduction, markersize=4, linewidth=1.5, fmt='v-', color="green", label=OCL_EB05["label"]+", smoothed")


ax2.axhline(1, color='r')

handles, labels = ax2.get_legend_handles_labels()
lgd1= ax2.legend(handles, labels)

ax2.set_ylim(0,2)

plt.xlabel(r'$E_x$ [MeV]',fontsize="medium")
plt.ylabel(r'ratio to input',fontsize="medium")

# Fine-tune figure; make subplots close to each other and hide x ticks for
# all but bottom plot.
fig.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
plt.savefig("nld_RAINIER.pdf")
###############################

fig= plt.figure("gSF")
ax = fig.add_subplot(211)
ax.set_yscale("log", nonposy='clip')

ax.tick_params("x", top="off")
ax.tick_params("y", right="off")

# Plot data points with error bars

# GDR data
# experimental1

#test
# OCL_singleJ_sf = plt.errorbar(OCL_singleJ["strength"][:,0], OCL_singleJ["strength"][:,1], yerr=OCL_singleJ["strength"][:,2], markersize=4, linewidth=1.5, fmt='v-', color="purple", label=OCL_singleJ["label"])
# # OCL_singleJ_tr = ax.plot(OCL_singleJ["trans"][:,0], OCL_singleJ["trans"][:,1], '--',markersize=4, linewidth=1.5, color="purple")
# # OCL_singleJ.update({"plt":OCL_singleJ_sf, "plt_trans":OCL_singleJ_tr})

# OCL_Potel_sf = plt.errorbar(OCL_Potel["strength"][:,0], OCL_Potel["strength"][:,1], yerr=OCL_Potel["strength"][:,2], markersize=4, linewidth=1.5, fmt='v-', color="green", label=OCL_Potel["label"])
# OCL_Potel_tr = ax.plot(OCL_Potel["trans"][:,0], OCL_Potel["trans"][:,1], '--',markersize=4, linewidth=1.5, color="green")
# OCL_Potel.update({"plt":OCL_Potel_sf, "plt_trans":OCL_Potel_tr})

# OCL_EB05_sf = plt.errorbar(OCL_EB05["strength"][:,0], OCL_EB05["strength"][:,1], yerr=OCL_EB05["strength"][:,2], markersize=4, linewidth=1.5, fmt='v-', color="black", label=OCL_EB05["label"])
# OCL_EB05_tr = ax.plot(OCL_EB05["trans"][:,0], OCL_EB05["trans"][:,1], '--',markersize=4, linewidth=1.5, color="black")
# OCL_EB05.update({"plt":OCL_EB05_sf, "plt_trans":OCL_EB05_tr})

OCL_Potelr10_plot = plotData(OCL_Potelr10, dicEntry="strength")
OCL_Potel_plot = plotData(OCL_Potel, dicEntry="strength")
OCL_Potel_rhotot_plot = plotData(OCL_Potel_rhotot, dicEntry="strength", fmt="--")
OCL_EB05_plot = plotData(OCL_EB05, dicEntry="strength")

plt.gca().set_prop_cycle(None) # reset color cycle

OCL_Potelr10_plot = plotData(OCL_Potelr10, dicEntry="trans", fmt="--")
OCL_Potel_plot = plotData(OCL_Potel, dicEntry="trans", fmt="--")
OCL_Potel_rhotot_plot = plotData(OCL_Potel_rhotot, dicEntry="trans", fmt="--")
OCL_EB05_plot = plotData(OCL_EB05, dicEntry="trans", fmt="--")


gSF_true_all = np.loadtxt("misc/GSFTable_py.dat")
gSF_true_tot = gSF_true_all[:,1] + gSF_true_all[:,2]
gSF_true = np.array(zip(gSF_true_all[:,0],gSF_true_tot))
gSF_true_plot = plt.plot(gSF_true[:140,0],gSF_true[:140,1], "k-", label="gSF_true")

handles, labels = ax.get_legend_handles_labels()
lgd1= ax.legend(handles, labels)
# lgd2 = ax.legend([KopecyE1], ['data'])
# plt.gca().add_artist(lgd1)
# ax.legend([(shade_r100,OCL_r100), (shade_r008,OCL_r008), (shade_r008tadj,OCL_r008), gur79, mor93, ber86, KopecyE1, KopecyM1, OCL_fermi_r100, OCL_fermi_r008,OCL_Potel,OCL_EB05],
#                 ["Oslo (r=1) incl. upper/lower band", "Oslo (r=0.08) incl. upper/lower band","Oslo (r=0.08); + 0.02 uncertainty in T_red; incl. upper/lower band" ,"Gurevich 1976", "Moraes 1993",  "Berman 1986/Evaluated", "Kopecky 2017 (E1)", "Kopecky 2017 (M1)", "OCL, FG, r=1", "OCL, FG, r=0.08",
#                 "OCL_Potel","OCL_EB05"],
# 	            loc=4)
# ax.legend(handler_map={OCL_r100:HandlerLine2D(numpoints=2)})

plt.xlabel(r'$E_\gamma$ [MeV]',fontsize="medium")
plt.ylabel(r"$\gamma$SF [MeV$^{-3}$]",fontsize="medium")


#############################
# fig= plt.figure("Ratios gSF")
ax2 = fig.add_subplot(212, sharex = ax)
# ax.set_yscale("log", nonposy='clip')

ax2.tick_params("x", top="off")
ax2.tick_params("y", right="off")

ratio_gSF_true = calcRatioTrue(OCL_Potelr10,gSF_true, "strength")
ratio_plot = plt.errorbar(OCL_Potelr10["strength"][:,0], unumpy.nominal_values(ratio_gSF_true), yerr=unumpy.std_devs(ratio_gSF_true), markersize=4, linewidth=1.5, fmt='.-', color="C3", label=OCL_Potelr10["label"])
ratio_gSF_true = calcRatioTrue(OCL_Potel,gSF_true, "strength")
ratio_plot = plt.errorbar(OCL_Potel["strength"][:,0], unumpy.nominal_values(ratio_gSF_true), yerr=unumpy.std_devs(ratio_gSF_true), markersize=4, linewidth=1.5, fmt='.-', color="black", label=OCL_Potel["label"])
ratio_gSF_true = calcRatioTrue(OCL_EB05,gSF_true, "strength")
ratio_plot = plt.errorbar(OCL_EB05["strength"][:,0], unumpy.nominal_values(ratio_gSF_true), yerr=unumpy.std_devs(ratio_gSF_true), markersize=4, linewidth=1.5, fmt='v-', color="grey", label=OCL_EB05["label"])

ax2.axhline(1, color='r')

handles, labels = ax2.get_legend_handles_labels()
lgd1= ax2.legend(handles, labels)

ax2.set_ylim(0,3)
plt.xlim((0,7))

plt.xlabel(r'$E_\gamma$ [MeV]',fontsize="medium")
plt.ylabel(r'ratio to input',fontsize="medium")

# Fine-tune figure; make subplots close to each other and hide x ticks for
# all but bottom plot.
fig.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
plt.savefig("gsf_RAINIER.pdf")
###############################

plt.show()
