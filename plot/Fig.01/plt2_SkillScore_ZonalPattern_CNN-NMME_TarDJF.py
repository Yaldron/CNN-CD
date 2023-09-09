import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


#Color Table:Version AMWG
DICT = {'CMC1_CanCM3':'#FFE200','CMC2_CanCM4':'#FF4300','CCSM4':'#C9A06B',\
	'GFDL_aer04':'#B259B2','GFDL_FLOR_A06':'#FF8BFF',\
	'GFDL_FLOR_B01':'#1AB4AB','NASA_GMAO':'#6482E5','NCEP_CFSv2':'#8257AC',\
	'MME-NMME':'green'} #19CA2A


fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(7.5,5.5)) #width+height
plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams.update({'mathtext.fontset':'stix'})
plt.rcParams.update({'font.weight':'semibold'})


f     = xr.open_dataset("Skill_DJF-SSTA_zonal_pattern.nc")
fNMME = xr.open_dataset("../../data/Assess/SkillScore_ZonalPattern_NMME_tarDJF.nc")
CorC_SS  = f['SCC'].values

CorC          = CorC_SS
CMC1_CanCM3   = fNMME['CanCM3'       ].values
CMC2_CanCM4   = fNMME['CanCM4'       ].values
CCSM4         = fNMME['CCSM4'        ].values
GFDL_aer04    = fNMME['GFDL_aer04'   ].values
GFDL_FLOR_A06 = fNMME['GFDL_FLOR_A06'].values
GFDL_FLOR_B01 = fNMME['GFDL_FLOR_B01'].values
NASA_GMAO     = fNMME['NASA_GMAO'    ].values
NCEP_CFSv2    = fNMME['NCEP_CFSv2'   ].values
MME           = fNMME['MME'          ].values


ax.plot(np.arange(1,13),CMC1_CanCM3  ,c=DICT['CMC1_CanCM3']  ,linestyle='-',linewidth=1.5,marker='o',markersize=3,label='CanCM3')      #橙色
ax.plot(np.arange(1,13),CMC2_CanCM4  ,c=DICT['CMC2_CanCM4']  ,linestyle='-',linewidth=1.5,marker='o',markersize=3,label='CanCM4')      #橙红色
ax.plot(np.arange(1,13),CCSM4        ,c=DICT['CCSM4']        ,linestyle='-',linewidth=1.5,marker='o',markersize=3,label='CCSM4')            #土黄
ax.plot(np.arange(1,13),GFDL_aer04   ,c=DICT['GFDL_aer04']   ,linestyle='-',linewidth=1.5,marker='o',markersize=3,label='GFDL-aer04')       #中紫
ax.plot(np.arange(1,13),GFDL_FLOR_A06,c=DICT['GFDL_FLOR_A06'],linestyle='-',linewidth=1.5,marker='o',markersize=3,label='GFDL-FLOR-A06')    #中紫
ax.plot(np.arange(1,13),GFDL_FLOR_B01,c=DICT['GFDL_FLOR_B01'],linestyle='-',linewidth=1.5,marker='o',markersize=3,label='GFDL-FLOR-B01')    #中紫
ax.plot(np.arange(1,10),NASA_GMAO    ,c=DICT['NASA_GMAO']    ,linestyle='-',linewidth=1.5,marker='o',markersize=3,label='NASA-GAMO')          #橙红色
ax.plot(np.arange(1,11),NCEP_CFSv2   ,c=DICT['NCEP_CFSv2']   ,linestyle='-',linewidth=1.5,marker='o',markersize=3,label='NCEP-CFSv2')       #green 
ax.plot(np.arange(1,13),MME          ,c=DICT['MME-NMME']     ,linestyle='-',linewidth=2.0,marker='o',markersize=4.5,label='MME-NMME')       #green 
ax.plot(np.arange(1,19),CorC,color='blue',linestyle='-',linewidth=2.5,marker='o',markersize=6,markerfacecolor='white',markeredgewidth=2.0,label='CNN-CD')

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
ax.set_ylabel("Skill score",fontsize=17,weight='semibold')
ax.set_title("Skill score",fontsize=17,weight='semibold',loc='left')



handles, labels = ax.get_legend_handles_labels()
ax.legend([handles[9],handles[8],handles[0],handles[1],handles[2],handles[3],handles[4],handles[5],handles[6],handles[7]],
	[labels[9],labels[8],labels[0],labels[1],labels[2],labels[3],labels[4],labels[5],labels[6],labels[7]],fontsize=13,frameon=False,loc=1,ncol=2,labelspacing=0.6)
	

plt.tight_layout()
plt.savefig('Skill_ZonalPattern_CNN-NMME_tarDJF.svg',dpi=600,bbox_inches = 'tight')
