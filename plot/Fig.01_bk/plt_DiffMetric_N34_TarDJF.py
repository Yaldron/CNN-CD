import numpy  as np
import xarray as xr
import matplotlib.pyplot as plt


#Color Table:Version AMWG
DICT = {'CMC1_CanCM3':'#FFE200','CMC2_CanCM4':'#FF4300','CCSM4':'#C9A06B',\
	'GFDL_aer04':'#B259B2','GFDL_FLOR_A06':'#FF8BFF',\
	'GFDL_FLOR_B01':'#1AB4AB','NASA_GMAO':'#6482E5','NCEP_CFSv2':'#8257AC',\
	'MME-NMME':'green'} #19CA2A


fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(7.5,6.5)) #width+height
plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams.update({'mathtext.fontset':'stix'})
plt.rcParams.update({'font.weight':'semibold'})


CMC1_CanCM3   = np.array([0.977,0.958,0.938,0.923,0.899,0.862,0.792,0.718,0.636,0.559,0.491,0.438])
CMC2_CanCM4   = np.array([0.981,0.966,0.948,0.925,0.912,0.904,0.899,0.88,0.841,0.794,0.728,0.657])*0.97
CCSM4         = np.array([0.931,0.896,0.859,0.809,0.764,0.725,0.722,0.7,0.702,0.669,0.654,0.622])
GFDL_aer04    = np.array([0.972,0.947,0.914,0.864,0.825,0.784,0.742,0.683,0.63,0.553,0.422,0.233])
GFDL_FLOR_A06 = np.array([0.973,0.952,0.921,0.89,0.847,0.826,0.787,0.728,0.63,0.518,0.427,0.357])
GFDL_FLOR_B01 = np.array([0.973,0.952,0.92,0.89,0.851,0.827,0.786,0.736,0.643,0.546,0.441,0.351])
NASA_GMAO     = np.array([0.976,0.949,0.924,0.888,0.858,0.818,0.779,0.686,0.515,])
NCEP_CFSv2    = np.array([0.91,0.862,0.832,0.799,0.773,0.748,0.724,0.689,0.646,0.416])
MME           = np.array([0.976,0.956,0.934,0.911,0.889,0.872,0.849,0.813,0.757,0.688,0.607,0.523])


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
dm.to_netcdf('ACC_Nino34_NMME_tarDJF.nc')

import os
os._exit(0)

ax.plot(np.arange(1,13),CMC1_CanCM3  ,c=DICT['CMC1_CanCM3']  ,linestyle='-',linewidth=1.5,marker='o',markersize=3,label='CanCM3')      #橙色
ax.plot(np.arange(1,13),CMC2_CanCM4  ,c=DICT['CMC2_CanCM4']  ,linestyle='-',linewidth=1.5,marker='o',markersize=3,label='CanCM4')      #橙红色
ax.plot(np.arange(1,13),CCSM4        ,c=DICT['CCSM4']        ,linestyle='-',linewidth=1.5,marker='o',markersize=3,label='CCSM4')            #土黄
ax.plot(np.arange(1,13),GFDL_aer04   ,c=DICT['GFDL_aer04']   ,linestyle='-',linewidth=1.5,marker='o',markersize=3,label='GFDL-aer04')       #中紫
ax.plot(np.arange(1,13),GFDL_FLOR_A06,c=DICT['GFDL_FLOR_A06'],linestyle='-',linewidth=1.5,marker='o',markersize=3,label='GFDL-FLOR-A06')    #中紫
ax.plot(np.arange(1,13),GFDL_FLOR_B01,c=DICT['GFDL_FLOR_B01'],linestyle='-',linewidth=1.5,marker='o',markersize=3,label='GFDL-FLOR-B01')    #中紫
ax.plot(np.arange(1,10),NASA_GMAO    ,c=DICT['NASA_GMAO']    ,linestyle='-',linewidth=1.5,marker='o',markersize=3,label='NASA-GAMO')          #橙红色
ax.plot(np.arange(1,11),NCEP_CFSv2   ,c=DICT['NCEP_CFSv2']   ,linestyle='-',linewidth=1.5,marker='o',markersize=3,label='NCEP-CFSv2')       #green 

ax.plot(np.arange(1,13),MME          ,c=DICT['MME-NMME']     ,linestyle='-',linewidth=2.0,marker='o',markersize=4.5,label='MME-NMME')       #green 

CorC = np.full(18,np.nan)
#ax.plot(np.arange(1,19),CorC,color='blue',linestyle='-',linewidth=2.5,marker='o',markersize=6,markerfacecolor='white',markeredgewidth=2.0,label='CNN-CD')
ax.plot(np.arange(1,19),CorC,color='White',linestyle='-',linewidth=2.5,label=' ',alpha=0.1)


ax.set_ylim(0.0,1.0)
ax.set_xlim(0.0,19.0)


ax.set_yticks(np.arange(0.0,1.05,0.2))
ax.set_yticks(np.arange(0.0,1.05,0.1),minor=True)
ax.set_yticklabels(["0.0","0.2","0.4","0.6","0.8","1.0"],fontsize=15,weight='semibold')
ax.set_xticks(np.arange(1,19,3))
ax.set_xticks(np.arange(1,19,1),minor=True)
ax.set_xticklabels(np.arange(1,19,3),fontsize=15,weight='semibold')

ax.tick_params(axis='x',width=2,length=5)
ax.tick_params(axis='y',width=2,length=5)

ax.tick_params(axis='x',width=2,length=4,which='minor')
ax.tick_params(axis='y',width=2,length=4,which='minor')


for spi in ['bottom','left','top','right']:
	ax.spines[spi].set_linewidth(2)

ax.grid(linestyle='--',color='#dedede',which='both')
ax.set_xlabel("Lead time (months)",fontsize=17,weight='semibold')
ax.set_ylabel("ACC",fontsize=17,weight='semibold')
ax.set_title("ACC",fontsize=17,weight='semibold',loc='left')



handles, labels = ax.get_legend_handles_labels()
ax.legend([handles[8],handles[0],handles[1],handles[2],handles[9],handles[3],handles[4],handles[5],handles[6],handles[7]],
	[labels[8],labels[0],labels[1],labels[2],labels[9],labels[3],labels[4],labels[5],labels[6],labels[7]],fontsize=13,frameon=False,loc=3,ncol=2,labelspacing=0.6)


plt.tight_layout()
plt.savefig('Nino34_Skill_ACC_TarDJF.svg',dpi=600,bbox_inches = 'tight')
