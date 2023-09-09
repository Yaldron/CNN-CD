"""
File: plt_Bars_3ENSO_Precursors.py
Author: Ming Sun
Email: gosun1994@gmail.com
Github: https://github.com/Yaldron
Description: 
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch

fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(13,7)) #width+height
plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams.update({'mathtext.fontset':'stix'})
plt.rcParams.update({'font.weight':'semibold'})

ax1 = ax.flat[0]
ax2 = ax.flat[1]

#============================================================================
#;;fNPMM    = np.loadtxt("./pmmsst.data",skiprows=1,comments='#')[:,1:] #read the PMM-SSTA index
#;;idx_NPMM = np.mean(fNPMM[:,0:3],axis=1)  
#;;idx_NPMM = idx_NPMM/np.std(idx_NPMM) #1948-2021
#;;yrNPMM   = np.arange(1948,2022,1)
#;;
#;;yrSPQ   = np.loadtxt("./Normalized_TA-SSTA_ERSSTv5.txt")[:,0].astype("int")
#;;idx_SPQ = np.loadtxt("./Normalized_SPQI_60S15S_140E70W_ERSSTv5.txt")[:,1]
#;;
#;;yrTA    = np.loadtxt("./Normalized_TA-SSTA_ERSSTv5.txt")[:,0].astype("int")
#;;idx_TA  = np.loadtxt("./Normalized_TA-SSTA_ERSSTv5.txt")[:,1]
#;;
#;;#print(idx_NPMM.shape)#print(idx_SPQ.shape)#print(idx_TA.shape)
#;;
#;;DATA = []
#;;for i in np.arange(1948,2017)-1948:
#;;	DATA.append([1948+i,round(idx_NPMM[i],2),round(idx_SPQ[i],2),round(idx_TA[i]*-1.0,2)])
#;;
#;;with open('./Normalized_idx_ENSO_Precursors_in_JFM.txt','w') as f:
#;;	print("{:<6} {:<10} {:<10} {:<10}".format("Year","NPMM","SPQ","TA-SSTA"),file=f)
#;;
#;;for v in DATA:
#;;	a,b,c,d = v
#;;	with open('./Normalized_idx_ENSO_Precursors_in_JFM.txt','a') as f:
#;;		print("{:<6} {:<10} {:<10} {:<10}".format(a,b,c,d),file=f)
#;;
#;;import os
#;;os._exit(0)

success_year  = np.array([1982,1986,1991,1994,1997,2002,2004,2009,2014,2015])
fail_year = np.array([1987,2006])

NPMM = np.zeros(success_year.shape[0],dtype=float)
SPQ  = np.zeros(success_year.shape[0],dtype=float)
TAm1 = np.zeros(success_year.shape[0],dtype=float)

NPMM_S = np.zeros(fail_year.shape[0],dtype=float)
SPQ_S  = np.zeros(fail_year.shape[0],dtype=float)
TAm1_S = np.zeros(fail_year.shape[0],dtype=float)


for i in range(success_year.shape[0]):
	NPMM[i] = idx_NPMM[success_year[i] - yrNPMM[0]]
	SPQ[i]  = idx_SPQ[success_year[i] - yrSPQ[0]]
	TAm1[i] = idx_TA[success_year[i] - yrTA[0]]


for i in range(fail_year.shape[0]):
	NPMM_S[i] = idx_NPMM[fail_year[i] - yrNPMM[0]]
	SPQ_S[i]  = idx_SPQ[fail_year[i] - yrSPQ[0]]
	TAm1_S[i] = idx_TA[fail_year[i] - yrTA[0]]


S1 = NPMM
S2 = SPQ
S3 = TAm1

S1S = NPMM_S
S2S = SPQ_S
S3S = TAm1_S

bar_width = 0.25
plt.rcParams['hatch.linewidth'] = 1.8

ax1.axhline(y=0.0,ls='-',lw=1.8,color='k')
ax1.axhline(y=  1.0,ls=':',lw=1.8,color='k',zorder=6)
ax1.fill_between([10.7,10.8],-3.2,3.5, facecolor='lightgrey',alpha=0.75)


#success El Nino
B1=ax1.bar(np.arange(1,success_year.shape[0]+1,1)-bar_width*1.0,S1[:],
	color='#00AFBB',width=bar_width,ec='k',linewidth=1.5,zorder=2)
#two layer merge to a bar with black edge and colored hatch
B2=ax1.bar(np.arange(1,success_year.shape[0]+1,1)+bar_width*0.0,S2[:],
	fc='none',ec='#00AFBB',width=bar_width,hatch="|||",linewidth=1.5,zorder=2,)
B3=ax1.bar(np.arange(1,success_year.shape[0]+1,1)+bar_width*0.0,S2[:],
	fc='none',ec='k',width=bar_width,linewidth=1.5,zorder=3,)
#two layer merge to a bar with black edge and colored hatch
B4=ax1.bar(np.arange(1,success_year.shape[0]+1,1)+bar_width*1.0,S3[:],
	fc='none',ec='#00AFBB',width=bar_width,hatch="\\\\\\",linewidth=1.5,zorder=2,)
B5=ax1.bar(np.arange(1,success_year.shape[0]+1,1)+bar_width*1.0,S3[:],
	fc='none',ec='k',width=bar_width,linewidth=1.5,zorder=3,)

bar_width = 0.25
shift     = 0.2

#Special case
BE1=ax1.bar(np.array([11.5,12.5])-bar_width*1.0,S1S[:],
	color='#FC4E08',width=bar_width,ec='k',linewidth=1.5,zorder=2)
BE2=ax1.bar(np.array([11.5,12.5])+bar_width*0.0,S2S[:],
	fc='none',ec='#FC4E08',width=bar_width,hatch="|||",linewidth=1.5,zorder=2,)
BE3=ax1.bar(np.array([11.5,12.5])+bar_width*0.0,S2S[:],
	fc='none',ec='k',width=bar_width,linewidth=1.5,zorder=3,)
BE4=ax1.bar(np.array([11.5,12.5])+bar_width*1.0,S3S[:],
	fc='none',ec='#FC4E08',width=bar_width,hatch="\\\\\\",linewidth=1.5,zorder=2,)
BE5=ax1.bar(np.array([11.5,12.5])+bar_width*1.0,S3S[:],
	fc='none',ec='k',width=bar_width,linewidth=1.5,zorder=3,)

#===Canvas setting================
ax1.set_xlim(0.5,13.0)
ax1.set_ylim(-1.2,3.5)

ax1.set_xticks([1,2,3,4,5,6,7,8,9,10,  11.5,12.5])
ax1.set_xticklabels(['82/83','86/87','91/92','94/95','97/98','02/03','04/05','09/10','14/15','15/16','87/88','06/07'],
	fontsize=15,weight='semibold')

ax1.set_yticks(np.arange(-1.0,3.51,1.0))
ax1.set_yticks(np.arange(-1.0,3.51,0.5),minor=True)
ax1.set_yticklabels(['-1.0','0','1.0','2.0','3.0'],fontsize=12,weight='semibold',color='k')

del [success_year,fail_year,NPMM,SPQ,TAm1,NPMM_S,SPQ_S,TAm1_S]
del [S1,S2,S3,S1S,S2S,S3S]
#============================================================================================

success_year  = np.array([1983,1984,1988,1998,1999,2000,2005,2007,2008,2010,2011])
fail_year = np.array([1995])


NPMM = np.zeros(success_year.shape[0],dtype=float)
SPQ  = np.zeros(success_year.shape[0],dtype=float)
TAm1 = np.zeros(success_year.shape[0],dtype=float)

NPMM_S = np.zeros(fail_year.shape[0],dtype=float)
SPQ_S  = np.zeros(fail_year.shape[0],dtype=float)
TAm1_S = np.zeros(fail_year.shape[0],dtype=float)


for i in range(success_year.shape[0]):
	NPMM[i] = idx_NPMM[success_year[i] - yrNPMM[0]]
	SPQ[i]  = idx_SPQ[success_year[i] - yrSPQ[0]]
	TAm1[i] = idx_TA[success_year[i] - yrTA[0]]


for i in range(fail_year.shape[0]):
	NPMM_S[i] = idx_NPMM[fail_year[i] - yrNPMM[0]]
	SPQ_S[i]  = idx_SPQ[fail_year[i] - yrSPQ[0]]
	TAm1_S[i] = idx_TA[fail_year[i] - yrTA[0]]


S1 = NPMM
S2 = SPQ
S3 = TAm1

S1S = NPMM_S
S2S = SPQ_S
S3S = TAm1_S

bar_width = 0.25
plt.rcParams['hatch.linewidth'] = 1.8


ax2.axhline(y=-1.0,ls=':',lw=1.8,color='k',zorder=6)
ax2.fill_between([11.7,11.8],-3.5,2.9, facecolor='lightgrey',alpha=0.75)
#success El Nino
C1=ax2.bar(np.arange(1,success_year.shape[0]+1,1)-bar_width*1.0,S1[:],
	color='#00AFBB',width=bar_width,ec='k',linewidth=1.5,zorder=2)
#two layer merge to a bar with black edge and colored hatch
C2=ax2.bar(np.arange(1,success_year.shape[0]+1,1)+bar_width*0.0,S2[:],
	fc='none',ec='#00AFBB',width=bar_width,hatch="|||",linewidth=1.5,zorder=2,)
C3=ax2.bar(np.arange(1,success_year.shape[0]+1,1)+bar_width*0.0,S2[:],
	fc='none',ec='k',width=bar_width,linewidth=1.5,zorder=3,)
#two layer merge to a bar with black edge and colored hatch
C4=ax2.bar(np.arange(1,success_year.shape[0]+1,1)+bar_width*1.0,S3[:],
	fc='none',ec='#00AFBB',width=bar_width,hatch="\\\\\\",linewidth=1.5,zorder=2,)
C5=ax2.bar(np.arange(1,success_year.shape[0]+1,1)+bar_width*1.0,S3[:],
	fc='none',ec='k',width=bar_width,linewidth=1.5,zorder=3,)

bar_width = 0.25
shift     = 0.2

#Special case
CE1=ax2.bar(np.array([12.5])-bar_width*1.0,S1S[:],
	color='#FC4E08',width=bar_width,ec='k',linewidth=1.5,zorder=2)
CE2=ax2.bar(np.array([12.5])+bar_width*0.0,S2S[:],
	fc='none',ec='#FC4E08',width=bar_width,hatch="|||",linewidth=1.5,zorder=2,)
CE3=ax2.bar(np.array([12.5])+bar_width*0.0,S2S[:],
	fc='none',ec='k',width=bar_width,linewidth=1.5,zorder=3,)
CE4=ax2.bar(np.array([12.5])+bar_width*1.0,S3S[:],
	fc='none',ec='#FC4E08',width=bar_width,hatch="\\\\\\",linewidth=1.5,zorder=2,)
CE5=ax2.bar(np.array([12.5])+bar_width*1.0,S3S[:],
	fc='none',ec='k',width=bar_width,linewidth=1.5,zorder=3,)


ax2.set_xlim(0.5,13.0)
ax2.set_ylim(-3.5,2.9)


ax2.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12.5])
ax2.set_xticklabels(['83/84','84/85','88/89','98/99','99/00','00/01','05/06','07/08','08/09','10/11','11/12','95/96'],
	fontsize=15,weight='semibold')

ax2.set_yticks(np.arange(-3.0,3.0,1.0))
ax2.set_yticks(np.arange(-3.5,3.0,0.5),minor=True)
ax2.set_yticklabels(['-3.0','-2.0','-1.0','0','1.0','2.0'],fontsize=12,weight='semibold',color='k')
ax2.axhline(y=0.0,ls='-',lw=1.8,color='k')

#=========================================================================================
for i in range(2):
	ax.flat[i].tick_params(axis='x',width=1.5,length=5)
	ax.flat[i].tick_params(axis='y',width=1.5,length=5)
	ax.flat[i].tick_params(axis='x',width=1.5,length=2.5,which='minor')
	ax.flat[i].tick_params(axis='y',width=1.5,length=2.5,which='minor')
	for spi in ['bottom','left','top','right']:
	    ax.flat[i].spines[spi].set_linewidth(1.5)

	ax.flat[i].grid(linestyle='--',color='#dedede')

	ax.flat[i].set_ylabel("Normalized index of precursors",fontsize=14,weight='semibold',labelpad=8.0)


ax1.set_title('(c) Three ENSO precursors in JFM(0) of ENs',weight='semibold',fontsize=17.5,loc='left')
ax2.set_title('(d) Three ENSO precursors in JFM(0) of LNs',weight='semibold',fontsize=17.5,loc='left')

plt.tight_layout()
plt.subplots_adjust(hspace=0.35)
plt.savefig('Barplot_3Precursors_in_JFM.pdf',dpi=900)






