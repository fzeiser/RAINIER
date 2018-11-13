import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import os
from uncertainties import ufloat, unumpy
from uncertainties.umath import *  # sin(), etc
from scipy.ndimage.filters import gaussian_filter
import io
from utilities import *
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate

sns.set()

# sns.set_context("paper")
# sns.set_context("talk")

# sns.set(font_scale=1.2) # Bigger than normal fonts
# sns.set(font_scale=1.2)
sns.set(rc={'figure.figsize':(5.5,6.5)})
sns.set_style("ticks", { 'axes.grid': True})
plt.rcParams['legend.loc'] = 'best'

cwd = os.getcwd()
###########

def rhoCT(Ex,T,E0):
	E = Ex-E0
	return np.exp(E/T)/T

from decimal import Decimal
print(('%.2E' % Decimal(rhoCT(6.543,T=0.425,E0=-0.456))))
# print rhoCT(6.5,T=0.44,E0=-0.456)



###########
# get calibration from counting.dat
def getCalibrationFromCounting(filename):
    f = open(filename)
    lines = f.readlines()
    #19: float a0 =  -0.7800;
    #20: float a1 =   0.1300;
    cal = np.genfromtxt(io.BytesIO(lines[18].encode()),dtype=object, delimiter="=")
    if cal[0]!=b"float a0 ":
        raise ValueError("Could not read calibration")
    a0 = float(cal[1][:-1])
    cal = np.genfromtxt(io.BytesIO(lines[19].encode()),dtype=object, delimiter="=")
    if cal[0]!=b"float a1 ":
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
    print(("Reading {0}\nwith calibration a0={1:.3e}, a1 ={2:.3e}".format(folder, a0_strength, a1_strength)))
    strength = convertStrength(folder+"/strength.nrm",a0_strength, a1_strength)
    trans = getTransExt(folder+"/transext.nrm", a0_strength, a1_strength, Emin=0.1, Emax=8.)
    nld = convertStrength(folder+"/rhopaw.cnt",a0_strength, a1_strength)
    data = {'strength': strength, 
    		 'trans': trans,
    	     'nld': nld,
    		 'label':label}
    return data

# normal runs
# # OCL_Potel_rhotot = ReadFiles(cwd+"/Jint_Greg_mama_RIPL_absolut/1Gen_rhotot",r"$g_{pop} \ll g_{int}$, r=1.0")
# OCL_Potel_rhotot = ReadFiles(cwd+"/Jint_Greg_mama_RIPL_absolut/folded_rhotot",r"$g_{pop} \ll g_{int}$, r=1.0")
OCL_Potel_rhotot = ReadFiles(cwd+"/Jint_Greg_mama_RIPL_gsf01/folded_rhotot",r"$g_{pop} \ll g_{int}$, r=1.0")
# OCL_Potel = ReadFiles(cwd+"/Jint_Greg_mama_RIPL_allE1/folded_rhotot",r"$g_{pop} \ll g_{int}$, r=0.3")
OCL_Potel = ReadFiles(cwd+"/Jint_Greg_mama_RIPL_absolut/folded_r30",r"$g_{pop} \ll g_{int}$, r=0.3")
OCL_Potelr10 = ReadFiles(cwd+"/Jint_Greg_mama_RIPL_absolut/folded_r10",r"$g_{pop} \ll g_{int}$, r=0.1")
OCL_EB05 = ReadFiles(cwd+"/Jint_EB06_mama/folded",r"$g_{pop} = g_{int}$") 

# tests
# # # OCL_Potel_rhotot = ReadFiles(cwd+"/Jint_Greg_mama_RIPL_absolut/1Gen_rhotot",r"$g_{pop} \ll g_{int}$, r=1.0")
# OCL_Potel_rhotot = ReadFiles(cwd+"/Jint_Greg_mama_RIPL_absolut/folded_rhotot",r"$g_{pop} \ll g_{int}$, r=1.0")
# # OCL_Potel = ReadFiles(cwd+"/Jint_Greg_mama_RIPL_allE1/folded_rhotot",r"$g_{pop} \ll g_{int}$, r=1, all E1")
# OCL_Potel = ReadFiles(cwd+"/Jint_Greg_mama_RIPL_allM1/folded_rhotot",r"$g_{pop} \ll g_{int}$, r=1, all M1")
# # OCL_Potel = ReadFiles(cwd+"/Jint_Greg_mama_RIPL_absolut/folded_r30",r"$g_{pop} \ll g_{int}$, r=0.3")
# OCL_Potelr10 = ReadFiles(cwd+"/Jint_Greg_mama_RIPL_absolut/folded_r10",r"$g_{pop} \ll g_{int}$, r=0.1")
# OCL_EB05 = ReadFiles(cwd+"/Jint_Greg_mama_RIPL_allE1/folded_rhotot",r"$g_{pop} \ll g_{int}$, r=1, all E1")

# OCL_Potel_rhotot = ReadFiles(cwd+"/Jint_Greg_mama_RIPL/1Gen_rhotot","Potel_rhotot")
# OCL_Potel_rhotot = ReadFiles(cwd+"/Jint_Greg_mama_RIPL/folded_rhotot",r"$g_{pop} \ll g_{int}$, r=1.0")
# OCL_Potel = ReadFiles(cwd+"/Jint_Greg_mama_RIPL/folded_r30",r"$g_{pop} \ll g_{int}$, r=0.3")
# OCL_Potelr10 = ReadFiles(cwd+"/Jint_Greg_mama_RIPL/folded_r10",r"$g_{pop} \ll g_{int}$, r=0.1")


# OCL_Potel_rhotot = ReadFiles(cwd+"/Jint_Greg_parity_rhotot","Potel_rhotot")
# OCL_Potel = ReadFiles(cwd+"/Jint_Greg_parity","Potel_r30")
# OCL_Potel_rhotot = ReadFiles(cwd+"/Jint_Greg_mama/1Gen","Potel_1Gen")
# OCL_Potel = ReadFiles(cwd+"/Jint_Greg_parity_discred","Potel_r30_redDiscStates")

# OCL_EB05 = ReadFiles(cwd+"/Jint_EB06_trueNLD","EB05_trueNLD")
# OCL_EB05 = ReadFiles(cwd+"/Jint_EB06_mama/folded","EB05")   	   
# OCL_EB05 = ReadFiles(cwd+"/Jint_EB06_mama/folded",r"$g_{pop} = g_{int}$")          


# Make x-axis array to plot from
# Earray = np.linspace(0,20,800)
###############################
# plotting helpers 

def plotData(data, dicEntry, axis, fmt='v-', **kwarg):
    try:
        plot = ax.errorbar(data[dicEntry][:,0], data[dicEntry][:,1], yerr=data[dicEntry][:,2], markersize=4, linewidth=1.5, fmt=fmt, label=data["label"], **kwarg)
    except (IndexError):
        # plot = plt.errorbar(data[dicEntry][:,0], data[dicEntry][:,1], markersize=4, linewidth=1.5, fmt=fmt, label=data["label"])
        plot = ax.errorbar(data[dicEntry][:,0], data[dicEntry][:,1], markersize=4, linewidth=1.5, fmt=fmt, **kwarg)

    return plot

def calcRatio(set1, set2, attribute):
    set1un = unumpy.uarray(set1[attribute][:,1],std_devs=set1[attribute][:,2])
    set2un = unumpy.uarray(set2[attribute][:,1],std_devs=set2[attribute][:,2])
    return set1un/set2un

def calcRatioTrue(dic1, true, attribute):
    try:
        un1 = unumpy.uarray(dic1[attribute][:,1],std_devs=dic1[attribute][:,2])
    except:
        un1 = unumpy.uarray(dic1[attribute][:,1],std_devs=0)
    true_interpolate = np.interp(dic1[attribute][:,0],true[:,0],true[:,1])
    return un1/true_interpolate

###########################
# Initialize figure
fig, axes = plt.subplots(2,1)
ax, ax2 = axes

for axi in axes.flat:
    axi.yaxis.set_major_locator(plt.MaxNLocator(4))
    axi.tick_params("x", top="off")
    axi.tick_params("y", right="off")
    axi.set_xlim(0,7)

ax.set_yscale("log", nonposy='clip') # needs to come after MaxNLocator
ax.set_ylim(bottom=1, top=1e8)
ax2.set_ylim(0,1.9)

# horizontal comparison line
ax2.axhline(1, color='r')

ax.set_ylabel(r'$\rho$ [1/MeV]',fontsize="medium")
ax2.set_xlabel(r'$E_x$ [MeV]',fontsize="medium")
ax2.set_ylabel(r'ratio to input',fontsize="medium")

# Fine-tune figure; make subplots close to each other and hide x ticks for upper plot
fig.subplots_adjust(hspace=0, top=0.98, left=0.17, right=0.98)
plt.setp(ax.get_xticklabels(), visible=False)
color_pallet = sns.color_palette()
# plt.tight_layout()


NLD_true_disc = np.loadtxt("misc/NLD_exp_disc.dat")
NLD_true_cont = np.loadtxt("misc/NLD_exp_cont.dat")
# apply same binwidth to continuum states
binwidth_goal = NLD_true_disc[1,0]-NLD_true_disc[0,0]
print(binwidth_goal)
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

# plot "true nld"
ax.step(np.append(-binwidth_goal,NLD_true[:-1,0])+binwidth_goal/2.,np.append(0,NLD_true[:-1,1]), "k", where="pre",label="input NLD, binned")

def plotNLDs(dataset, **kwarg):
    plotData(dataset, dicEntry="nld", axis=ax, **kwarg)
    ratio_nld = calcRatioTrue(dataset,NLD_true, "nld")
    ax2.errorbar(dataset["nld"][:,0], unumpy.nominal_values(ratio_nld), yerr=unumpy.std_devs(ratio_nld), markersize=4, linewidth=1.5, fmt='v-', label=OCL_EB05["label"], **kwarg)
    handles, labels = ax.get_legend_handles_labels()
    lgd1= ax.legend(handles, labels, fontsize="medium")
    return ratio_nld

# OCL_Potel_nld = plotData(OCL_Potelr10, dicEntry="nld", fmt="v--")
plotNLDs(OCL_EB05, color=color_pallet[0])
plt.savefig("nld_RAINIER_0.pdf")

ratio_nld = plotNLDs(OCL_Potel_rhotot, color=color_pallet[1])
plt.savefig("nld_RAINIER_1.pdf")

plotNLDs(OCL_Potel, color=color_pallet[2])
plt.savefig("nld_RAINIER_2.pdf")

plt.savefig("nld_RAINIER.pdf")


###############################


fig, axes = plt.subplots(2,1)
ax, ax2 = axes

color_pallet = sns.color_palette()

for axi in axes.flat:
    axi.yaxis.set_major_locator(plt.MaxNLocator(5))
    axi.tick_params("x", top="off")
    axi.tick_params("y", right="off")
    axi.set_xlim(0,7)

ax.set_yscale("log", nonposy='clip') # needs to come after MaxNLocator
ax2.set_ylim(0,2.4)

# horizontal comparison line
ax2.axhline(1, color='r')

ax.set_ylabel(r'$\gamma$SF [1/MeV$^3$]',fontsize="medium")
ax2.set_xlabel(r'$E_\gamma$ [MeV]',fontsize="medium")
ax2.set_ylabel(r'ratio to input',fontsize="medium")

# Fine-tune figure; make subplots close to each other and hide x ticks for upper plot
fig.subplots_adjust(hspace=0, top=0.98, left=0.17, right=0.98)
plt.setp(ax.get_xticklabels(), visible=False)

# Plot data points with error bars


gSF_true_all = np.loadtxt("Jint_Greg_mama_RIPL_gsf01/GSFTable_py.dat")
gSF_true_tot = gSF_true_all[:,1] + gSF_true_all[:,2]
gSF_true = np.array(list(zip(gSF_true_all[:,0],gSF_true_tot)))
gSF_true_plot = ax.plot(gSF_true[:140,0],gSF_true[:140,1], "k-", label="gSF_true")

# OCL_EB05_plot = plotData(OCL_EB05, dicEntry="strength", axis=ax)
# OCL_Potel_rhotot_plot = plotData(OCL_Potel_rhotot, dicEntry="strength", fmt="--", axis=ax)
# OCL_Potel_plot = plotData(OCL_Potel, dicEntry="strength", axis=ax)
# OCL_Potelr10_plot = plotData(OCL_Potelr10, dicEntry="strength", axis=ax)

def plotGSFs(dataset, **kwarg):
    plotData(dataset, dicEntry="strength", axis=ax, **kwarg)
    plotData(dataset, dicEntry="trans", fmt="--", axis=ax, **kwarg)
    ratio_gSF = calcRatioTrue(dataset,gSF_true, "strength")
    ax2.errorbar(dataset["strength"][:,0], unumpy.nominal_values(ratio_gSF), yerr=unumpy.std_devs(ratio_gSF), markersize=4, linewidth=1.5, fmt='v-', label=dataset["label"], **kwarg)
    handles, labels = ax.get_legend_handles_labels()
    lgd1= ax.legend(handles, labels, fontsize="medium")
    return ratio_gSF

plotGSFs(OCL_EB05, color=color_pallet[0])
plt.savefig("gsf_RAINIER_0.pdf")

ratio_gSF = plotGSFs(OCL_Potel_rhotot, color=color_pallet[1])
plt.savefig("gsf_RAINIER_1.pdf")

plotGSFs(OCL_Potel, color=color_pallet[2])
plt.savefig("gsf_RAINIER_2.pdf")

plotGSFs(OCL_Potelr10, color=color_pallet[3])
plt.savefig("gsf_RAINIER_3.pdf")

plt.savefig("gsf_RAINIER.pdf")

###############################

# plt.show()


###############################

# Get new, "corrected" nld and gSF
# that can be set into RAINIER for the next iteration
Sn = 6.534
E_crit = 1.03755 # critical energy / last discrete level
# E_fitmin = 2. # ignore data below 2 MeV -- arb. selection



E_exp = OCL_Potel_rhotot["nld"][:,0]
idE_crit = np.abs(E_exp-E_crit).argmin()
E_exp = E_exp[idE_crit:]
y = unumpy.nominal_values(1/ratio_nld)[idE_crit:]
y_smooth = gaussian_filter1d(y, sigma=2)
yerr = unumpy.std_devs(1/ratio_nld)[idE_crit:]

# add constraint: no change a Sn
E_exp = np.append(E_exp,Sn)
y = np.append(y,1.)
y_smooth = np.append(y_smooth,1.)
yerr = np.append(yerr,1e-9)

# spl = UnivariateSpline(E_exp, y, w=1/yerr)
# idE = np.abs(E_exp-E_fitmin).argmin()
# popt_nld = np.polyfit(E_exp[idE:], y[idE:], deg=1, rcond=None, full=False, w=1/yerr[idE:], cov=False)
# fcorr_nld = np.poly1d(popt_nld)
# print(("popt_nld", popt_nld))

# fcorr_nld = interpolate.interp1d(E_exp, y_smooth)

plt.figure()
plt.errorbar(E_exp, y, yerr)
xarr = np.linspace(E_crit,Sn)
# plt.plot(xarr,fcorr_nld(xarr))
plt.plot(E_exp, y_smooth)
# plt.show()

# apply correction
nld_init = rhoCT(E_exp,T=0.425,E0=-0.456)
rho_new = y_smooth * nld_init
plt.figure()
plt.semilogy(E_exp, nld_init)
plt.plot(E_exp,rho_new)
# plt.show()

# Write to RAINIER
def sigma2(U,A,a,E1, rmi_red=1):
    #cut-off parameters of EB05
    sigma2 = np.sqrt(rmi_red) * 0.0146*A**(5./3.) * ( 1. + np.sqrt(1. + 4.*a*(U-E1)) ) / (2.*a)
    return np.sqrt(sigma2)

def sigma(U,A,a,E1, rmi_red=1):
    return np.sqrt(sigma2(U,A,a,E1, rmi_red=1))

fname = "nld_new.dat"
WriteRAINIERnldTable(fname, E_exp, rho_new, sigma(U=E_exp, A=240, a=25.16,E1=0.12), a=None)

##########################################
# repeat for gSF

def getFullRatio(dataset):
    ratio_gSF = calcRatioTrue(dataset,gSF_true, "strength")
    ratio_trans = calcRatioTrue(dataset,gSF_true, "trans")
    Eg = dataset["strength"][:,0]
    Eg_trans = dataset["trans"][:,0]
    Egsf_min = Eg[0]
    Egsf_max = Eg[-1]
    ratio_gSF_tot = [None] * len(Eg_trans)

    j = 0
    for i, E in enumerate(Eg_trans):
        if (Egsf_min < E < Egsf_max):
            ratio_gSF_tot[i] = ratio_gSF[j]
            j += 1
        else:
            ratio_gSF_tot[i] = ratio_trans[i]
    ratio_gSF_tot = np.array(ratio_gSF_tot)
    return ratio_gSF_tot


ratio_gSF = getFullRatio(OCL_Potel_rhotot)
E_exp = OCL_Potel_rhotot["trans"][:,0]
idE_Sn = np.abs(E_exp-Sn).argmin()
E_exp = E_exp[:idE_Sn]
gsf_init = OCL_Potel_rhotot["trans"][:idE_Sn,1]
y = unumpy.nominal_values(1/ratio_gSF)[:idE_Sn]
yerr = unumpy.std_devs(1/ratio_gSF)[:idE_Sn]

# spl = UnivariateSpline(E_exp, y, w=1/yerr)
# popt_gsf = np.polyfit(E_exp, y, deg=2, rcond=None, full=False, w=1/yerr, cov=False)
# fcorr_gsf = np.poly1d(popt_gsf)
# print("popt_gsf", popt_gsf)

plt.figure()
plt.errorbar(E_exp, y, yerr)
# xarr = np.linspace(0,Sn)
# plt.plot(xarr,fcorr_gsf(xarr))

# from scipy.interpolate import UnivariateSpline
# spl = UnivariateSpline(E_exp, y, w=1/yerr, k=4)
# plt.plot(xarr,spl(xarr))
y_smooth = gaussian_filter1d(y, sigma=2)
plt.plot(E_exp,y_smooth)

# plt.show()

plt.figure()
gsf_new = y_smooth * gsf_init
plt.semilogy(E_exp, gsf_init)
plt.semilogy(E_exp, gsf_new)

data_all = list(zip(E_exp,gsf_new))
np.savetxt('gsf_new.dat', data_all, header="E gsf_sum")

plt.show()


