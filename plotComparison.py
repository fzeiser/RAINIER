import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import os
from uncertainties import ufloat, unumpy
from uncertainties.umath import *  # sin(), etc

sns.set()

sns.set_context("paper")
sns.set(font_scale=1.2) # Bigger than normal fonts
sns.set_style("ticks")

cwd = os.getcwd()
###########

# transform strength.nrm to f
def convertStrength(filename):
    data_ocl = np.loadtxt(filename)
    data_ocl_copy = data_ocl
    # calibration coefficients
    # a0_strength =  -0.8296
    # a1_strength =   0.1354
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
def getTransExt(myfile, Emin, Emax):
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

# Change these values!
a0_strength =  -0.7500
a1_strength =   0.1250

def ReadFiles(folder, label):
	strength = convertStrength(folder+"/strength.nrm")
	trans = getTransExt(folder+"/transext.nrm", Emin=0.1, Emax=8.)
	nld = convertStrength(folder+"/rhopaw.cnt")
	data = {'strength': strength, 
			 'trans': trans,
		     'nld': nld,
			 'label':label}
	return data

OCL_Greg = ReadFiles(cwd+"/Jint_Greg","Greg")
OCL_EB06 = ReadFiles(cwd+"/Jint_EB06","EB06")		   

# Make x-axis array to plot from
# Earray = np.linspace(0,20,800)
###############################
# Initialize figure
# plt.figure()
fig= plt.figure(0)
ax = fig.add_subplot(111)
ax.set_yscale("log", nonposy='clip')

ax.tick_params("x", top="off")
ax.tick_params("y", right="off")

def plotData(data, dicEntry):
	return ax.errorbar(data[dicEntry][:,0], data[dicEntry][:,1], yerr=data[dicEntry][:,2], markersize=4, linewidth=1.5, fmt='v-', label=data["label"])

OCL_Greg_nld = plotData(OCL_Greg, dicEntry="nld")
OCL_EB06_nld = plotData(OCL_EB06, dicEntry="nld")

handles, labels = ax.get_legend_handles_labels()
lgd1= ax.legend(handles, labels, loc=4)

plt.xlabel(r'$E_x$ [MeV]',fontsize="medium")
plt.ylabel(r'levels/MeV',fontsize="medium")

#############################
fig= plt.figure(1)
ax = fig.add_subplot(111)
# ax.set_yscale("log", nonposy='clip')

ax.tick_params("x", top="off")
ax.tick_params("y", right="off")
OCL_Greg_nld_un = unumpy.uarray(OCL_Greg["nld"][:,1],std_devs=OCL_Greg["nld"][:,2])
OCL_EB06_nld_un = unumpy.uarray(OCL_EB06["nld"][:,1],std_devs=OCL_EB06["nld"][:,2])
ratio = OCL_Greg_nld_un/OCL_EB06_nld_un

ratio = ax.errorbar(OCL_EB06["nld"][:,0], unumpy.nominal_values(ratio), yerr=unumpy.std_devs(ratio), markersize=4, linewidth=1.5, fmt='v-', color="black", label="ratio nld Greg/EB06")
ax.axhline(1, color='r')

handles, labels = ax.get_legend_handles_labels()
lgd1= ax.legend(handles, labels, loc=4)

ax.set_ylim(0,2)

plt.xlabel(r'$E_x$ [MeV]',fontsize="medium")
plt.ylabel(r'ratio',fontsize="medium")

###############################

fig= plt.figure(2)
ax = fig.add_subplot(111)
ax.set_yscale("log", nonposy='clip')

ax.tick_params("x", top="off")
ax.tick_params("y", right="off")

# Plot data points with error bars

# GDR data
# experimental1

#test
# OCL_singleJ_sf = ax.errorbar(OCL_singleJ["strength"][:,0], OCL_singleJ["strength"][:,1], yerr=OCL_singleJ["strength"][:,2], markersize=4, linewidth=1.5, fmt='v-', color="purple", label=OCL_singleJ["label"])
# # OCL_singleJ_tr = ax.plot(OCL_singleJ["trans"][:,0], OCL_singleJ["trans"][:,1], '--',markersize=4, linewidth=1.5, color="purple")
# # OCL_singleJ.update({"plt":OCL_singleJ_sf, "plt_trans":OCL_singleJ_tr})

# OCL_Greg_sf = ax.errorbar(OCL_Greg["strength"][:,0], OCL_Greg["strength"][:,1], yerr=OCL_Greg["strength"][:,2], markersize=4, linewidth=1.5, fmt='v-', color="green", label=OCL_Greg["label"])
# OCL_Greg_tr = ax.plot(OCL_Greg["trans"][:,0], OCL_Greg["trans"][:,1], '--',markersize=4, linewidth=1.5, color="green")
# OCL_Greg.update({"plt":OCL_Greg_sf, "plt_trans":OCL_Greg_tr})

# OCL_EB06_sf = ax.errorbar(OCL_EB06["strength"][:,0], OCL_EB06["strength"][:,1], yerr=OCL_EB06["strength"][:,2], markersize=4, linewidth=1.5, fmt='v-', color="black", label=OCL_EB06["label"])
# OCL_EB06_tr = ax.plot(OCL_EB06["trans"][:,0], OCL_EB06["trans"][:,1], '--',markersize=4, linewidth=1.5, color="black")
# OCL_EB06.update({"plt":OCL_EB06_sf, "plt_trans":OCL_EB06_tr})

OCL_Greg_nld = plotData(OCL_Greg, dicEntry="strength")
OCL_EB06_nld = plotData(OCL_EB06, dicEntry="strength")

handles, labels = ax.get_legend_handles_labels()
lgd1= ax.legend(handles, labels, loc=4)
# lgd2 = ax.legend([KopecyE1], ['data'])
# plt.gca().add_artist(lgd1)
# ax.legend([(shade_r100,OCL_r100), (shade_r008,OCL_r008), (shade_r008tadj,OCL_r008), gur79, mor93, ber86, KopecyE1, KopecyM1, OCL_fermi_r100, OCL_fermi_r008,OCL_greg,OCL_EB06],
#                 ["Oslo (r=1) incl. upper/lower band", "Oslo (r=0.08) incl. upper/lower band","Oslo (r=0.08); + 0.02 uncertainty in T_red; incl. upper/lower band" ,"Gurevich 1976", "Moraes 1993",  "Berman 1986/Evaluated", "Kopecky 2017 (E1)", "Kopecky 2017 (M1)", "OCL, FG, r=1", "OCL, FG, r=0.08",
#                 "OCL_greg","OCL_EB06"],
# 	            loc=4)
# ax.legend(handler_map={OCL_r100:HandlerLine2D(numpoints=2)})

plt.xlabel(r'$E_\gamma$ [MeV]',fontsize="medium")
plt.ylabel(r'gSF',fontsize="medium")

plt.show()