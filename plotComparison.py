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

def ReadFiles(folder, label):
    a0_strength, a1_strength = getCalibrationFromCounting(folder+"/counting.cpp")
    print(("Reading {0}\nwith calibration a0={1:.3e}, a1 ={2:.3e}".format(folder, a0_strength, a1_strength)))
    strength = convertStrength(folder+"/strength.nrm",a0_strength, a1_strength)
    trans = getTransExt(folder+"/transext.nrm", a0_strength, a1_strength, Emin=0.1, Emax=8.)
    nld = convertStrength(folder+"/rhopaw.cnt",a0_strength, a1_strength)
    NLD_exp_cont = np.loadtxt(folder+"/../nld_new.txt")
    gsf_input = np.loadtxt(folder+"/../GSFTable_py.dat")
    gsf_input = np.c_[gsf_input[:,0], gsf_input[:,1]+gsf_input[:,2]]
    data = {'strength': strength,
    		 'trans': trans,
             'gsf_input': gsf_input,
    	     'nld': nld,
             'nld_cont' : NLD_exp_cont,
    		 'label':label}
    return data

def log_interp1d(xx, yy, kind='linear'):
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = interpolate.interp1d(logx, logy, kind=kind)
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp


# OCL_EB05 = ReadFiles(cwd+"/Jint_EB06_mama/folded",r"$g_{pop} = g_{int}$")
OCL_Potel01 = ReadFiles(cwd+"/Jint_Greg_mama_RIPL_gsf01/folded_rhotot",r"Iteration 1, $g_{pop} \ll g_{int}$, r=1.0")
OCL_Potel02 = ReadFiles(cwd+"/Jint_Greg_mama_RIPL_gsf_changesep_02_02/folded_rhotot",r"Iteration 2, $g_{pop} \ll g_{int}$, r=1.0")
# OCL_Potel03 = ReadFiles(cwd+"/Jint_Greg_mama_RIPL_gsf01/folded_rhotot",r"Out 01 $g_{pop} \ll g_{int}$")
current_dataset = OCL_Potel02 # which dataset should we compare to

Sn = 6.534
E_crit = 1.03755 # critical energy / last discrete level

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

def calcRatioTrue(dic1, obs, attribute):
    try:
        un1 = unumpy.uarray(dic1[attribute][:,1],std_devs=dic1[attribute][:,2])
    except:
        un1 = unumpy.uarray(dic1[attribute][:,1],std_devs=0)
    obs_interpolate = np.interp(dic1[attribute][:,0],obs[:,0],obs[:,1])
    return un1/obs_interpolate

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


NLD_obs_binned_disc = np.loadtxt("misc/NLD_exp_disc.dat")
NLD_obs_binned_cont = current_dataset["nld_cont"]
# apply same binwidth to continuum states
binwidth_goal = NLD_obs_binned_disc[1,0]-NLD_obs_binned_disc[0,0]
print(binwidth_goal)
binwidth_cont = NLD_obs_binned_cont[1,0]-NLD_obs_binned_cont[0,0]
Emax = NLD_obs_binned_cont[-1,0]
nbins = int(np.ceil(Emax/binwidth_goal))
Emax_adjusted = binwidth_goal*nbins # Trick to get an integer number of bins
bins = np.linspace(0,Emax_adjusted,nbins+1)
hist, edges = np.histogram(NLD_obs_binned_cont[:,0],bins=bins,weights=NLD_obs_binned_cont[:,1]*binwidth_cont)
NLD_obs_binned = np.zeros((nbins,2))
NLD_obs_binned[:nbins,0] = bins[:nbins]
NLD_obs_binned[:,1] = hist/binwidth_goal
NLD_obs_binned[:len(NLD_obs_binned_disc),1] += NLD_obs_binned_disc[:,1]

# plot "obs nld"
ax.step(np.append(-binwidth_goal,NLD_obs_binned[:-1,0])+binwidth_goal/2.,np.append(0,NLD_obs_binned[:-1,1]), "k", where="pre",label="generated NLD, binned")

def plotNLDs(dataset, **kwarg):
    plotData(dataset, dicEntry="nld", axis=ax, **kwarg)
    ratio_nld = calcRatioTrue(dataset,nld_init, "nld")
    ax2.errorbar(dataset["nld"][:,0], unumpy.nominal_values(ratio_nld), yerr=unumpy.std_devs(ratio_nld), markersize=4, linewidth=1.5, fmt='v-', label=dataset["label"], **kwarg)
    handles, labels = ax.get_legend_handles_labels()
    lgd1= ax.legend(handles, labels, fontsize="medium")
    return ratio_nld

# plotNLDs(OCL_EB05, color=color_pallet[0])
# plt.savefig("nld_RAINIER_0.pdf")

E_exp = OCL_Potel01["nld"][:,0]
idE_crit = np.abs(E_exp-E_crit).argmin()
E_exp = E_exp[idE_crit:]
E_exp = np.append(E_exp,Sn)
nld_init = rhoCT(E_exp,T=0.425,E0=-0.456)
nld_init = np.c_[E_exp,nld_init]
ax.plot(nld_init[:,0], nld_init[:,1],"y-", label="input NLD from CT")
# NLD_obs_binned=np.c_[E_exp,nld_init]

plotNLDs(OCL_Potel01, color=color_pallet[1])
plt.savefig("nld_RAINIER_1.pdf")

try:
    plotNLDs(OCL_Potel02, color=color_pallet[2])
    plt.savefig("nld_RAINIER_2.pdf")
except:
    pass

try:
    plotNLDs(OCL_Potel03, color=color_pallet[3])
except:
    pass


plt.savefig("nld_RAINIER.pdf")
plt.savefig("nld_RAINIER.png")

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


gsf_obs = OCL_Potel01["gsf_input"]
idSn = np.abs(gsf_obs[:,0]-Sn).argmin()
gsf_obs_plot = ax.plot(gsf_obs[:idSn,0],gsf_obs[:idSn,1], "k-", label="gsf_obs")

def plotGSFs(dataset, **kwarg):
    plotData(dataset, dicEntry="strength", axis=ax, **kwarg)
    plotData(dataset, dicEntry="trans", fmt="--", axis=ax, **kwarg)
    ratio_gSF = calcRatioTrue(dataset,gsf_obs, "strength")
    ax2.errorbar(dataset["strength"][:,0], unumpy.nominal_values(ratio_gSF), yerr=unumpy.std_devs(ratio_gSF), markersize=4, linewidth=1.5, fmt='v-', label=dataset["label"], **kwarg)
    handles, labels = ax.get_legend_handles_labels()
    lgd1= ax.legend(handles, labels, fontsize="medium")
    return ratio_gSF

# plotGSFs(OCL_EB05, color=color_pallet[0])
# plt.savefig("gsf_RAINIER_0.pdf")

plotGSFs(OCL_Potel01, color=color_pallet[1])
plt.savefig("gsf_RAINIER_1.pdf")

try:
    plotGSFs(OCL_Potel02, color=color_pallet[2])
    plt.savefig("gsf_RAINIER_2.pdf")
except:
    pass

try:
    plotGSFs(OCL_Potel03, color=color_pallet[3])
    plt.savefig("gsf_RAINIER_3.pdf")
except:
    pass

plt.savefig("gsf_RAINIER.pdf")
plt.savefig("gsf_RAINIER.png")

###############################

# Get new, "corrected" nld and gSF
# that can be set into RAINIER for the next iteration

ratio_nld = calcRatioTrue(current_dataset,nld_init, "nld")
y = unumpy.nominal_values(1/ratio_nld)[idE_crit:]

ydiff = (y-1)
ydiff /= 2 # try smaller change for better convergence
y = 1+ydiff

y_smooth = gaussian_filter1d(y, sigma=2)
yerr = unumpy.std_devs(1/ratio_nld)[idE_crit:]

# add constraint: no change a Sn
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
plt.errorbar(E_exp, y, yerr, label="extracted correction")
plt.plot(E_exp, y_smooth, label="smoothed")
plt.legend(loc="best")
plt.xlabel(r"$E_x$ [MeV]")
plt.ylabel("Correction factor")
plt.savefig("nld_correction.png")

# plt.show()

# apply correction
plt.figure()
rho_new = y_smooth * nld_init[:,1]

# interpolate some values between last point in data and Sn
E_interp = np.linspace(E_exp[-2],E_exp[-1],num=20)
E_interp = np.concatenate((E_exp,E_interp[1:]))
rho_new_interp = log_interp1d(E_exp,rho_new)
rho_new = rho_new_interp(E_interp)

plt.step(NLD_obs_binned[:,0], NLD_obs_binned[:,1], "--",
             color="0.5", label="generated")
plt.semilogy(nld_init[:,0], nld_init[:,1], label="exp. \"obs\"")
plt.plot(E_interp,rho_new, "-", label="new")
plt.legend(loc="best")
plt.xlabel(r"$E_x$ [MeV]")
plt.ylabel("nld")
plt.savefig("nld_corrected.png")
# plt.show()

# Write to RAINIER
def sigma2(U,A,a,E1, rmi_red=1):
    #cut-off parameters of EB05
    sigma2 = np.sqrt(rmi_red) * 0.0146*A**(5./3.) * ( 1. + np.sqrt(1. + 4.*a*(U-E1)) ) / (2.*a)
    return np.sqrt(sigma2)

def sigma(U,A,a,E1, rmi_red=1):
    return np.sqrt(sigma2(U,A,a,E1, rmi_red=1))

fname = "nld_new.dat"
WriteRAINIERnldTable(fname, E_interp, rho_new, sigma(U=E_interp, A=240, a=25.16,E1=0.12), a=None)
np.savetxt('nld_new.txt', np.c_[E_interp, rho_new], header="E[MeV] nld[MeV^-1]")

##########################################
# repeat for gSF

def getFullRatio(dataset):
    ratio_gSF = calcRatioTrue(dataset,gsf_obs, "strength")
    ratio_trans = calcRatioTrue(dataset,gsf_obs, "trans")
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


ratio_gSF = getFullRatio(current_dataset)

gsf_input = current_dataset["gsf_input"]
idE_Sn = np.abs(gsf_input[:,0]-Sn).argmin()
gsf_input = gsf_input[:idE_Sn,:]

E_exp = OCL_Potel01["trans"][:,0]
idE_Sn = np.abs(E_exp-Sn).argmin()
E_exp = E_exp[:idE_Sn]

y = unumpy.nominal_values(1/ratio_gSF)[:idE_Sn]
yerr = unumpy.std_devs(1/ratio_gSF)[:idE_Sn]

# spl = UnivariateSpline(E_exp, y, w=1/yerr)
# popt_gsf = np.polyfit(E_exp, y, deg=2, rcond=None, full=False, w=1/yerr, cov=False)
# fcorr_gsf = np.poly1d(popt_gsf)
# print("popt_gsf", popt_gsf)

plt.figure()
plt.errorbar(E_exp, y, yerr, label="correction")
# xarr = np.linspace(0,Sn)
# plt.plot(xarr,fcorr_gsf(xarr))

# from scipy.interpolate import UnivariateSpline
# spl = UnivariateSpline(E_exp, y, w=1/yerr, k=4)
# plt.plot(xarr,spl(xarr))
y_smooth = gaussian_filter1d(y, sigma=2)
plt.plot(E_exp,y_smooth, label="smoothed")
plt.legend(loc="best")
plt.xlabel(r"$E_\gamma$ [MeV]")
plt.ylabel("correction")
plt.savefig("gsf_correction.png")

# applt to gsf
plt.figure()

gsf_input_interp = np.interp(E_exp,gsf_input[:,0],gsf_input[:,1])
gsf_new = y_smooth * gsf_input_interp
idSn = np.abs(gsf_obs[:,0]-Sn).argmin()
plt.plot(gsf_obs[:idSn,0],gsf_obs[:idSn,1],
         "--",color="0.5", label="exp \"obs\"")
plt.semilogy(gsf_input[:,0], gsf_input[:,1], label="input")
plt.semilogy(E_exp, gsf_new, label="new")
plt.legend(loc="best")
plt.xlabel(r"$E_\gamma$ [MeV]")
plt.ylabel("gsf")
plt.savefig("gsf_corrected.png")

# export as asci
data_all = list(zip(E_exp,gsf_new))
np.savetxt('gsf_new.dat', data_all, header="E gsf_sum")

# plt.show()


