import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import scipy as sp
from scipy.stats import pearsonr

import seaborn as sns
import pandas  as pd

#Color Table:Version AMWG
DICT = {'CMC1_CanCM3':'#FFE200','CMC2_CanCM4':'#FF4300','CCSM4':'#C9A06B',\
	'GFDL_aer04':'#B259B2','GFDL_FLOR_A06':'#FF8BFF',\
	'GFDL_FLOR_B01':'#1AB4AB','NASA_GMAO':'#6482E5','NCEP_CFSv2':'#8257AC',\
	'MME-NMME':'green'} #19CA2A


fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(7.5,5.5)) #width+height
plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams.update({'mathtext.fontset':'stix'})
plt.rcParams.update({'font.weight':'semibold'})

NAME = ["PCC","Skill Score","Consine Similarity","TSS"]

f = xr.open_dataset("/home/sunming/ML_ENSO/Confrim_data/Prediction/Metric4_DJF-SSTA_zonal_pattern.nc")
CorC_SS  = f['SCC'].values

CorC          = CorC_SS
CMC1_CanCM3   = np.array([0.796,0.74,0.656,0.566,0.495,0.441,0.383,0.325,0.284,0.25,0.238,0.249])
CMC2_CanCM4   = np.array([0.835,0.798,0.747,0.672,0.618,0.581,0.557,0.535,0.498,0.454,0.396,0.324])
CCSM4         = np.array([0.812,0.744,0.663,0.606,0.566,0.557,0.54,0.536,0.505,0.474,0.44,0.43])
GFDL_aer04    = np.array([0.727,0.663,0.562,0.454,0.341,0.262,0.199,0.164,0.16,0.148,0.16,0.161])
GFDL_FLOR_A06 = np.array([0.871,0.789,0.688,0.603,0.533,0.49,0.487,0.486,0.478,0.454,0.393,0.324])
GFDL_FLOR_B01 = np.array([0.86,0.779,0.691,0.611,0.563,0.52,0.513,0.49,0.489,0.441,0.382,0.309])
NASA_GMAO     = np.array([0.789,0.783,0.727,0.618,0.492,0.391,0.319,0.298,0.302])
NCEP_CFSv2    = np.array([0.812,0.714,0.609,0.537,0.512,0.51,0.491,0.478,0.444,0.392])
MME           = np.array([0.8719,0.7923,0.6921,0.6010,0.5336,0.4886,0.4534,0.4143,0.3913,0.3631,0.3317,0.2833])

ds1 = xr.Dataset({'CanCM3'       :(('lead'),CMC1_CanCM3   )},coords={'lead':np.arange(1,13,1)})
ds2 = xr.Dataset({'CanCM4'       :(('lead'),CMC2_CanCM4   )},coords={'lead':np.arange(1,13,1)})
ds3 = xr.Dataset({'CCSM4'        :(('lead'),CCSM4         )},coords={'lead':np.arange(1,13,1)})
ds4 = xr.Dataset({'GFDL_aer04'   :(('lead'),GFDL_aer04    )},coords={'lead':np.arange(1,13,1)})
ds5 = xr.Dataset({'GFDL_FLOR_A06':(('lead'),GFDL_FLOR_A06 )},coords={'lead':np.arange(1,13,1)})
ds6 = xr.Dataset({'GFDL_FLOR_B01':(('lead'),GFDL_FLOR_B01 )},coords={'lead':np.arange(1,13,1)})
ds7 = xr.Dataset({'NASA_GMAO'    :(('lead_1'),NASA_GMAO   )},coords={'lead_1':np.arange(1,10,1)})
ds8 = xr.Dataset({'NCEP_CFSv2'   :(('lead_2'),NCEP_CFSv2  )},coords={'lead_2':np.arange(1,11,1)})
ds9 = xr.Dataset({'MME'          :(('lead'),MME           )},coords={'lead':np.arange(1,13,1)})

dm = xr.merge([ds1,ds2,ds3,ds4,ds5,ds6,ds7,ds8,ds9])
dm.to_netcdf('SkillScore_ZonalPattern_NMME_tarDJF.nc')


import os
os._exit(0)


ax.flat[i].plot(np.arange(1,13),CMC1_CanCM3  ,c=DICT['CMC1_CanCM3']  ,linestyle='-',linewidth=1.5,marker='o',markersize=3,label='CanCM3')      #橙色
ax.flat[i].plot(np.arange(1,13),CMC2_CanCM4  ,c=DICT['CMC2_CanCM4']  ,linestyle='-',linewidth=1.5,marker='o',markersize=3,label='CanCM4')      #橙红色
ax.flat[i].plot(np.arange(1,13),CCSM4        ,c=DICT['CCSM4']        ,linestyle='-',linewidth=1.5,marker='o',markersize=3,label='CCSM4')            #土黄
ax.flat[i].plot(np.arange(1,13),GFDL_aer04   ,c=DICT['GFDL_aer04']   ,linestyle='-',linewidth=1.5,marker='o',markersize=3,label='GFDL-aer04')       #中紫
ax.flat[i].plot(np.arange(1,13),GFDL_FLOR_A06,c=DICT['GFDL_FLOR_A06'],linestyle='-',linewidth=1.5,marker='o',markersize=3,label='GFDL-FLOR-A06')    #中紫
ax.flat[i].plot(np.arange(1,13),GFDL_FLOR_B01,c=DICT['GFDL_FLOR_B01'],linestyle='-',linewidth=1.5,marker='o',markersize=3,label='GFDL-FLOR-B01')    #中紫
ax.flat[i].plot(np.arange(1,10),NASA_GMAO    ,c=DICT['NASA_GMAO']    ,linestyle='-',linewidth=1.5,marker='o',markersize=3,label='NASA-GAMO')          #橙红色
ax.flat[i].plot(np.arange(1,11),NCEP_CFSv2   ,c=DICT['NCEP_CFSv2']   ,linestyle='-',linewidth=1.5,marker='o',markersize=3,label='NCEP-CFSv2')       #green 

ax.flat[i].plot(np.arange(1,13),MME          ,c=DICT['MME-NMME']     ,linestyle='-',linewidth=2.0,marker='o',markersize=4.5,label='MME-NMME')       #green 
ax.flat[i].plot(np.arange(1,19),CorC,color='blue',linestyle='-',linewidth=2.5,marker='o',markersize=6,markerfacecolor='white',markeredgewidth=2.0,label='CNN-CD')

ax.flat[i].set_ylim(0.0,1.0)
ax.flat[i].set_xlim(0.0,19.0)


ax.flat[i].set_yticks(np.arange(0.0,1.05,0.2))
ax.flat[i].set_yticks(np.arange(0.0,1.05,0.1),minor=True)
ax.flat[i].set_yticklabels(["0.0","0.2","0.4","0.6","0.8","1.0"],fontsize=15,weight='semibold')
ax.flat[i].set_xticks(np.arange(1,19,3))
ax.flat[i].set_xticks(np.arange(1,19,1),minor=True)
ax.flat[i].set_xticklabels(np.arange(1,19,3),fontsize=15,weight='semibold')

ax.flat[i].tick_params(axis='x',width=2,length=5)
ax.flat[i].tick_params(axis='y',width=2,length=5)

ax.flat[i].tick_params(axis='x',width=2,length=4,which='minor')
ax.flat[i].tick_params(axis='y',width=2,length=4,which='minor')


for spi in ['bottom','left','top','right']:
	ax.flat[i].spines[spi].set_linewidth(2)

ax.flat[i].grid(linestyle='--',color='#dedede',which='both')
ax.flat[i].set_xlabel("Lead time (months)",fontsize=17,weight='semibold')
ax.flat[i].set_ylabel(NAME[i],fontsize=17,weight='semibold')
ax.flat[i].set_title(NAME[i],fontsize=17,weight='semibold',loc='left')



handles, labels = ax.flat[i].get_legend_handles_labels()
ax.flat[i].legend([handles[9],handles[8],handles[0],handles[1],handles[2],handles[3],handles[4],handles[5],handles[6],handles[7]],
	[labels[9],labels[8],labels[0],labels[1],labels[2],labels[3],labels[4],labels[5],labels[6],labels[7]],fontsize=13,frameon=False,loc=1,ncol=2,labelspacing=0.6)
	

plt.tight_layout()
plt.savefig('Zonal_Pattern_Skill_4Metric_TarDJF.svg',dpi=600,bbox_inches = 'tight')
