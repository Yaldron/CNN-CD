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


def runaveMid(x,nw):
	y = np.zeros(x.shape[0],dtype=float)
	half=nw//2

	for i in range(half,x.shape[0]-half):
		y[i] = np.mean(x[i-half:i+half+1])

	y[0] = x[0]
	y[x.shape[0]-1] = x[x.shape[0]-1]

	for i in range(1,half):
		y[i] = np.mean(x[0:i*2+1])

	for i in range(x.shape[0]-half,x.shape[0]-1):
		y[i] = np.mean(x[i-half:i])

	return y

fm1 = xr.open_dataset('/home/sunming/ML_ENSO/post_hind+fore_NMME/full-anom/anom_region_sst_mon_'+'CMC1-CanCM3'+'_198201-201612.nc')
fm2 = xr.open_dataset('/home/sunming/ML_ENSO/post_hind+fore_NMME/full-anom/anom_region_sst_mon_'+'CMC2-CanCM4'+'_198201-201612.nc')
fm3 = xr.open_dataset('/home/sunming/ML_ENSO/post_hind+fore_NMME/full-anom/anom_region_sst_mon_'+'COLA-RSMAS-CCSM4'+'_198201-201612.nc')
fm4 = xr.open_dataset('/home/sunming/ML_ENSO/post_hind+fore_NMME/full-anom/anom_region_sst_mon_'+'GFDL-CM2p1-aer04'+'_198201-201612.nc')
fm5 = xr.open_dataset('/home/sunming/ML_ENSO/post_hind+fore_NMME/full-anom/anom_region_sst_mon_'+'GFDL-CM2p5-FLOR-A06'+'_198201-201612.nc')
fm6 = xr.open_dataset('/home/sunming/ML_ENSO/post_hind+fore_NMME/full-anom/anom_region_sst_mon_'+'GFDL-CM2p5-FLOR-B01'+'_198201-201612.nc')
fm7 = xr.open_dataset('/home/sunming/ML_ENSO/post_hind+fore_NMME/full-anom/anom_region_sst_mon_'+'NCEP-CFSv2'+'_198201-201612.nc')

ssta1 = fm1['sst'].loc[:,:,:,-5:5,120:280].mean(("M","lat"))
ssta2 = fm2['sst'].loc[:,:,:,-5:5,120:280].mean(("M","lat"))
ssta3 = fm3['sst'].loc[:,:,:,-5:5,120:280].mean(("M","lat"))
ssta4 = fm4['sst'].loc[:,:,:,-5:5,120:280].mean(("M","lat"))
ssta5 = fm5['sst'].loc[:,:,:,-5:5,120:280].mean(("M","lat"))
ssta6 = fm6['sst'].loc[:,:,:,-5:5,120:280].mean(("M","lat"))
ssta7 = fm7['sst'].loc[:,:,:,-5:5,120:280].mean(("M","lat")) #only 10 lead

sstaTT = (ssta1+ssta2+ssta3+ssta4+ssta5+ssta6)/6.0

DICT = {'CMC1_CanCM3':'#FFE200','CMC2_CanCM4':'#FF4300','CCSM4':'#C9A06B',\
	'GFDL_aer04':'#B259B2','GFDL_FLOR_A06':'#FF8BFF',\
	'GFDL_FLOR_B01':'#1AB4AB','NASA_GMAO':'#6482E5','NCEP_CFSv2':'#8257AC',\
	'MME-NMME':'green'} #19CA2A

CM3 = np.zeros((35,13),dtype=float)
CM4 = np.zeros((35,13),dtype=float)
CC4 = np.zeros((35,13),dtype=float)
G04 = np.zeros((35,13),dtype=float)
GFA = np.zeros((35,13),dtype=float)
GFB = np.zeros((35,13),dtype=float)
CFS = np.zeros((35,13),dtype=float)
MME = np.zeros((35,13),dtype=float)


for ilon  in range(13):
	tmp1 = ssta1.loc[:,:,130+ilon*10:130+ilon*10+20].mean(("lon"))#.values
	tmp2 = ssta2.loc[:,:,130+ilon*10:130+ilon*10+20].mean(("lon"))#.values
	tmp3 = ssta3.loc[:,:,130+ilon*10:130+ilon*10+20].mean(("lon"))#.values
	tmp4 = ssta4.loc[:,:,130+ilon*10:130+ilon*10+20].mean(("lon"))#.values
	tmp5 = ssta5.loc[:,:,130+ilon*10:130+ilon*10+20].mean(("lon"))#.values
	tmp6 = ssta6.loc[:,:,130+ilon*10:130+ilon*10+20].mean(("lon"))#.values
	tmp7 = ssta7.loc[:,:,130+ilon*10:130+ilon*10+20].mean(("lon"))#.values
	tmp0 = sstaTT.loc[:,:,130+ilon*10:130+ilon*10+20].mean(("lon")).values


	CM3[:,ilon] = (tmp1[4::12,8]+tmp1[4::12,9]+tmp1[4::12,10])/3.0
	CM4[:,ilon] = (tmp2[4::12,8]+tmp2[4::12,9]+tmp2[4::12,10])/3.0
	CC4[:,ilon] = (tmp3[4::12,8]+tmp3[4::12,9]+tmp3[4::12,10])/3.0
	G04[:,ilon] = (tmp4[4::12,8]+tmp4[4::12,9]+tmp4[4::12,10])/3.0
	GFA[:,ilon] = (tmp5[4::12,8]+tmp5[4::12,9]+tmp5[4::12,10])/3.0
	GFB[:,ilon] = (tmp6[4::12,8]+tmp6[4::12,9]+tmp6[4::12,10])/3.0
	CFS[:,ilon] = (tmp7[4::12,8]+tmp7[4::12,9])/2.0
	MME[:,ilon] = (tmp0[4::12,8]+tmp0[4::12,9]+tmp0[4::12,10])/3.0

ds1 = xr.Dataset({'CanCM3'       :(('year','region'),CM3)},coords={'year':np.arange(1982,2017),'region':np.arange(1,14)})
ds2 = xr.Dataset({'CanCM4'       :(('year','region'),CM4)},coords={'year':np.arange(1982,2017),'region':np.arange(1,14)})
ds3 = xr.Dataset({'CCSM4'        :(('year','region'),CC4)},coords={'year':np.arange(1982,2017),'region':np.arange(1,14)})
ds4 = xr.Dataset({'GFDL_aer04'   :(('year','region'),G04)},coords={'year':np.arange(1982,2017),'region':np.arange(1,14)})
ds5 = xr.Dataset({'GFDL_FLOR_A06':(('year','region'),GFA)},coords={'year':np.arange(1982,2017),'region':np.arange(1,14)})
ds6 = xr.Dataset({'GFDL_FLOR_B01':(('year','region'),GFB)},coords={'year':np.arange(1982,2017),'region':np.arange(1,14)})
ds7 = xr.Dataset({'NASA_GMAO'    :(('year','region'),CFS)},coords={'year':np.arange(1982,2017),'region':np.arange(1,14)})
ds8 = xr.Dataset({'MME'          :(('year','region'),MME)},coords={'year':np.arange(1982,2017),'region':np.arange(1,14)})

dm = xr.merge([ds1,ds2,ds3,ds4,ds5,ds6,ds7,ds8])
dm.to_netcdf('ZonalPattern_NMME_lead10_tarDJF.nc')

import os
os._exit(0)
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
plt.subplots_adjust(hspace=0.3,wspace =0.25)#调整子图间距
plt.savefig('zonalpattern_4s_8p_El_2BestDyn.svg',dpi=800,bbox_inches = 'tight')  
