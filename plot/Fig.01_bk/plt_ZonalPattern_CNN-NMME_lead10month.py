"""
File: draw_test.py
Author: Ming Sun
Email: gosun1994@gmail.com
Github: https://github.com/Yaldron
Description: Draw the prediction of zonal pattern
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import pearsonr

from netCDF4 import Dataset



DICT = {'CMC1_CanCM3':'#FFE200','CMC2_CanCM4':'#FF4300','CCSM4':'#C9A06B',\
	'GFDL_aer04':'#B259B2','GFDL_FLOR_A06':'#FF8BFF',\
	'GFDL_FLOR_B01':'#1AB4AB','NASA_GMAO':'#6482E5','NCEP_CFSv2':'#8257AC',\
	'MME-NMME':'green'} #19CA2A


#========NMME Prediction===================
fNMME = xr.open_dataset('ZonalPattern_NMME_lead10_tarDJF.nc')
CM3 = fNMME['CanCM3'].values
CM4 = fNMME['CanCM4'].values
CC4 = fNMME['CCSM4'].values
G04 = fNMME['GFDL_aer04'].values
GFA = fNMME['GFDL_FLOR_A06'].values
GFB = fNMME['GFDL_FLOR_B01'].values
CFS = fNMME['NASA_GMAO'].values
MME = fNMME['MME'].values

#========CNN Prediction===================
f    = xr.open_dataset("/home/sunming/ML_ENSO/Confrim_data/Prediction/CNN-CD_SSTA-pattern_Prediction_F3.nc")
TOBS =f['OBS'].values  #13X18X432
PRED =f['Pred'].values #13X18X432

lead  	= 10
st    	=  2

DJF_OBS = (TOBS[:,0,10:-3:12]+TOBS[:,0,11:-2:12]+TOBS[:,0,12::12])/3.0
DJF_CNN = (PRED[:,lead-2,st::12]+PRED[:,lead-1,st::12]+PRED[:,lead+0,st::12])/3.0


fig,axs = plt.subplots(nrows=4,ncols=6,figsize=(12,7))

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams.update({'mathtext.fontset':'stix'})
plt.rcParams.update({'font.weight':'semibold'})

RMSE1 = np.zeros((36),dtype=float)
RMSE2 = np.zeros((36),dtype=float)
CORR1 = np.zeros((36),dtype=float)
CORR2 = np.zeros((36),dtype=float)
NSTD1 = np.zeros((36),dtype=float)
NSTD2 = np.zeros((36),dtype=float)

count = 0

El = {1982-1981,1986-1981,1987-1981,1991-1981,1994-1981,1997-1981,2002-1981,2004-1981,2006-1981,2009-1981,2014-1981,2015-1981}
La = {1983-1981,1984-1981,1988-1981,1995-1981,1998-1981,1999-1981,2000-1981,2005-1981,2007-1981,2008-1981,2010-1981,2011-1981}

Elab = np.array(['1982/83','1986/87','1987/88','1991/92','1994/95','1997/98','2002/03','2004/05','2006/07','2009/10','2014/15','2015/16'])
Llab = np.array(['1983/84','1984/85','1988/89','1995/96','1998/99','1999/00','2000/01','2005/06','2007/08','2008/09','2010/11','2011/12'])

Seq = np.array(["c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"])


for i in range(1,36):
	if  i in El :
		axs.flat[count].set_ylim(-1.4,3.5)
		axs.flat[count].set_xlim(-0.5,12.5)
		axs.flat[count].text(0.1,2.95,Seq[count]+'.  '+Elab[count],fontsize=10,weight='semibold')
	
		axs.flat[count].plot(np.arange(13),DJF_OBS[:,i],'o-',c='k',ms=2.2,label='OBS',lw=1.4)

		axs.flat[count].plot(np.arange(13),CM3[i-1,:],lw=1.0,alpha=0.6,ls='--',c=DICT['CMC1_CanCM3'],label='CanCM3')
		axs.flat[count].plot(np.arange(13),CM4[i-1,:],lw=1.0,alpha=0.6,ls='--',c=DICT['CMC2_CanCM4'],label='CanCM4')
		axs.flat[count].plot(np.arange(13),CC4[i-1,:],lw=1.0,alpha=0.6,ls='--',c=DICT['CCSM4'],label='CCSM4')
		axs.flat[count].plot(np.arange(13),G04[i-1,:],lw=1.0,alpha=0.6,ls='--',c=DICT['GFDL_aer04']   ,label='GFDL-aer04')
		axs.flat[count].plot(np.arange(13),GFA[i-1,:],lw=1.0,alpha=0.6,ls='--',c=DICT['GFDL_FLOR_A06'],label='GFDL-FLOR-A06')
		axs.flat[count].plot(np.arange(13),GFB[i-1,:],lw=1.0,alpha=0.6,ls='--',c=DICT['GFDL_FLOR_B01'],label='GFDL-FLOR-B01')
		axs.flat[count].plot(np.arange(13),CFS[i-1,:],lw=1.0,alpha=0.6,ls='--',c=DICT['NCEP_CFSv2'],label='NCEP-CFSv2')

		axs.flat[count].plot(np.arange(13),MME[i-1,:],c=DICT['MME-NMME'],ls='-',label='MME-NMME',lw=1.25)

		#axs.flat[count].plot(np.arange(13),CNN_ENs[count,:],'o-',c='blue',ms=2.2,label='CNN-CD',lw=1.25)
		axs.flat[count].plot(np.arange(13),DJF_CNN[:,i],'o-',c='blue',ms=2.2,label='CNN-CD',lw=1.25)

		#--------------------------------------------------------------	
		axs.flat[count].set_xticks([1,4,7,10])
		axs.flat[count].set_xticks(np.arange(13),minor=True)
		axs.flat[count].set_xticklabels(['150E','180','150W','120W'],fontsize=8,weight='semibold')
		axs.flat[count].set_yticks(np.arange(-1.0,4.0,1.0))
		axs.flat[count].set_yticklabels(['-1.0','0','1.0','2.0','3.0'],fontsize=8,weight='semibold')
	
		axs.flat[count].tick_params(axis='x',width=1.35,length=4)
		axs.flat[count].tick_params(axis='y',width=1.35,length=4)
		axs.flat[count].tick_params(axis='x',width=1.35,length=2,which='minor')
		axs.flat[count].tick_params(axis='y',width=1.35,length=2,which='minor')
	
		for spi in ['bottom','left','top','right']:
			axs.flat[count].spines[spi].set_linewidth(1.35)
	
		axs.flat[count].grid(linestyle='--',color='#dedede')
		axs.flat[count].grid(linestyle='--',color='#dedede',axis='x',which='both')

		count = count+1


for i in range(1,36):
	if  i in La :
		axs.flat[count].set_ylim(-3.0,1.55)
		axs.flat[count].set_xlim(-0.5,12.5)
		axs.flat[count].text(0.1,1.0,Seq[count]+'.  '+Llab[count-12],fontsize=10,weight='semibold')
	
		axs.flat[count].plot(np.arange(13),DJF_OBS[:,i],'o-',c='k',ms=2.2,label='OBS',lw=1.4)

		axs.flat[count].plot(np.arange(13),CM3[i-1,:],lw=1.0,alpha=0.6,ls='--',c=DICT['CMC1_CanCM3'],label='CanCM3')
		axs.flat[count].plot(np.arange(13),CM4[i-1,:],lw=1.0,alpha=0.6,ls='--',c=DICT['CMC2_CanCM4'],label='CanCM4')
		axs.flat[count].plot(np.arange(13),CC4[i-1,:],lw=1.0,alpha=0.6,ls='--',c=DICT['CCSM4'],label='CCSM4')
		axs.flat[count].plot(np.arange(13),G04[i-1,:],lw=1.0,alpha=0.6,ls='--',c=DICT['GFDL_aer04']   ,label='GFDL-aer04')
		axs.flat[count].plot(np.arange(13),GFA[i-1,:],lw=1.0,alpha=0.6,ls='--',c=DICT['GFDL_FLOR_A06'],label='GFDL-FLOR-A06')
		axs.flat[count].plot(np.arange(13),GFB[i-1,:],lw=1.0,alpha=0.6,ls='--',c=DICT['GFDL_FLOR_B01'],label='GFDL-FLOR-B01')
		axs.flat[count].plot(np.arange(13),CFS[i-1,:],lw=1.0,alpha=0.6,ls='--',c=DICT['NCEP_CFSv2'],label='NCEP-CFSv2')

		axs.flat[count].plot(np.arange(13),MME[i-1,:],c=DICT['MME-NMME'],ls='-',label='MME-NMME',lw=1.25)

		axs.flat[count].plot(np.arange(13),DJF_CNN[:,i],'o-',c='blue',ms=2.2,label='CNN-CD',lw=1.25)

		#--------------------------------------------------------------	
		axs.flat[count].set_xticks([1,4,7,10])
		axs.flat[count].set_xticks(np.arange(13),minor=True)
		axs.flat[count].set_xticklabels(['150E','180','150W','120W'],fontsize=8,weight='semibold')
		axs.flat[count].set_yticks(np.arange(-3.0,1.2,1.0))
		axs.flat[count].set_yticklabels(['-3.0','-2.0','-1.0','0','1.0'],fontsize=8,weight='semibold')
	
		axs.flat[count].tick_params(axis='x',width=1.35,length=4)
		axs.flat[count].tick_params(axis='y',width=1.35,length=4)
		axs.flat[count].tick_params(axis='x',width=1.35,length=2,which='minor')
		axs.flat[count].tick_params(axis='y',width=1.35,length=2,which='minor')
	
		for spi in ['bottom','left','top','right']:
			axs.flat[count].spines[spi].set_linewidth(1.35)
	
		axs.flat[count].grid(linestyle='--',color='#dedede')
		axs.flat[count].grid(linestyle='--',color='#dedede',axis='x',which='both')

		count = count+1



LAN = np.full(13,np.nan,dtype=float)
axs.flat[0].plot(np.arange(13),LAN,label=' ',c='white')
axs.flat[0].plot(np.arange(13),LAN,label=' ',c='white')

#OBS--0 ,#1-7 model ,#8 MME ,#9 CNN ,#10 NAN ,#11 NAN
handles, labels = axs.flat[0].get_legend_handles_labels()
fig.legend([handles[0],handles[8],handles[4],handles[9],handles[1],handles[5],handles[10],handles[2],handles[6],handles[11],handles[3],handles[7],],\
	[labels[0],labels[8],labels[4],labels[9],labels[1],labels[5],labels[10],labels[2],labels[6],labels[11],labels[3],labels[7],],\
	fontsize=12,frameon=False,ncol=4,bbox_to_anchor=(0.8,0.015))


plt.tight_layout()
plt.subplots_adjust(hspace=0.3,wspace =0.25)
plt.savefig('zonalpattern_CNN-NMME_lead10_tarDJF.svg',dpi=800,bbox_inches = 'tight')  
