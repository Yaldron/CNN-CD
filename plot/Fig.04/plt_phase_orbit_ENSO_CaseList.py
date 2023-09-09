"""
File: plt_phase_orbit_ENSO_CaseList.py
Author: Ming Sun
Email: gosun1994@gmail.com
Github: https://github.com/Yaldron
Description:  Phase space diagrams of 24 ENSO events in the hindcast period. 
"""
import numpy  as np
import xarray as xr

import matplotlib.pyplot as plt
from   matplotlib import font_manager

def runaveMid(x,nw):
	y = np.zeros(x.shape[0],dtype=float)
	half=nw//2

	for i in range(half,x.shape[0]-half):
		y[i] = np.mean(x[i-half:i+half+1])

	y[0] = (x[0]+x[1])/2.0
	y[x.shape[0]-1] = (x[x.shape[0]-1]+x[x.shape[0]-2])/2.0

	for i in range(1,half):
		y[i] = np.mean(x[0:i*2+1])

	for i in range(x.shape[0]-half,x.shape[0]):
		y[i] = np.mean(x[i-half:i])

	return y



fssh = xr.open_dataset('../../data/Valid/anom_detrend_sshg_GODAS_198001-201812_1x1.nc')
fsst = xr.open_dataset('../../data/Valid/anom_detrend_sst_ERSSTv5_198001-201812_1x1.nc')

WWV  = fssh['zos'].loc[:,-5:5,120:280].mean(("lat","lon"))
N34  = fsst['sst'].loc["1980-01-01":,-5:5,190:240].mean(("lat","lon"))#.values

GA = np.array([1982,1986,1997,1987,1991,2002,2014,1994,2004,2009,2015])
G1 = np.array([1982,1986,1987,1991,1997,2002])
G2 = np.array([1994,2004,2006,2009,2014,2015])
G3 = np.array([1983,1995,1998,2005,2007,2010])
G4 = np.array([1988,1984,1999,2000,2008,2011])

TG1 = np.array(["1982/83","1986/87","1987/88","1991/92","1997/98","2002/03"])
TG2 = np.array(["1994/95","2004/05","2006/07","2009/10","2014/15","2015/16"])
TG3 = np.array(["1983/84","1995/96","1998/99","2005/06","2007/08","2010/11"])
TG4 = np.array(["1988/89","1984/85","1999/00","2000/10","2008/09","2011/22"])

TT1 = np.array(["c","d","e","f","g","h"])
TT2 = np.array(["i","j","k","l","m","n"])
TT3 = np.array(["o","p","q","r","s","t"])
TT4 = np.array(["u","v","w","x","y","z"])



fig,ax = plt.subplots(nrows=4,ncols=6,figsize=(16,10)) #width+height
plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif'] = ['Times New Roman']
#plt.rcParams.update({'mathtext.fontset':'stix'})
#plt.rcParams.update({'font.weight':'semibold'})

X0 = N34.values
Z0 = WWV.values

X0 = runaveMid(X0,3)
Z0 = runaveMid(Z0,3)

#================================================
N34.values = X0
WWV.values = Z0

GP = [GA,G1,G2,G3,G4]
TGP = [GA,TG1,TG2,TG3,TG4]
TTGP = [GA,TT1,TT2,TT3,TT4]

for i in range(24):
	ax.flat[i].axhline(y=0.0,c='lightgrey',ls='--',lw=1.5)
	ax.flat[i].axvline(x=0.0,c='lightgrey',ls='--',lw=1.5)

for iG in range(1,5):
	Samp   = GP[iG]
	TSamp  = TGP[iG]
	TTSamp = TTGP[iG]

	Col_ENS_N34   = np.zeros((Samp.shape[0],20),dtype=float)
	Col_ENS_WWV   = np.zeros((Samp.shape[0],20),dtype=float)

	for i in range(Samp.shape[0]):
		start = str(Samp[i]-1)+"-07-01"
		termi = str(Samp[i]+1)+"-02-28"
		X1 = N34.loc[start:termi].values
		Y2 = WWV.loc[start:termi].values
	
		Col_ENS_N34[i,:]   = X1
		Col_ENS_WWV[i,:]   = Y2*10.0*10.0
	
	#=================================================
	
	for i  in range(Col_ENS_N34.shape[0]):
		if (iG<3):
			ax.flat[(iG-1)*6+i].plot(Col_ENS_N34[i,:],Col_ENS_WWV[i,:],'o-',c='#FF0000',ms=2.5,lw=1.15) #orange
			ax.flat[(iG-1)*6+i].plot(Col_ENS_N34[i,:3],Col_ENS_WWV[i,:3],'o-',c='#990000',ms=4.0,mfc='white',mew=1.2) #firebrick

		if (iG==3) or (iG==4 and i==0):
			ax.flat[(iG-1)*6+i].plot(Col_ENS_N34[i,:],Col_ENS_WWV[i,:],'o-',c='#3399FF',ms=2.5,lw=1.2) #skyblue
			ax.flat[(iG-1)*6+i].plot(Col_ENS_N34[i,:3],Col_ENS_WWV[i,:3],'o-',c='#0066CC',ms=4.0,mfc='white',mew=1.2) 

		if (iG==4 and i>0):
			ax.flat[(iG-1)*6+i].plot(Col_ENS_N34[i,:],Col_ENS_WWV[i,:],'o-',c='#0000FF',ms=2.5,lw=1.1) #skyblue   #3D3DCD
			ax.flat[(iG-1)*6+i].plot(Col_ENS_N34[i,:3],Col_ENS_WWV[i,:3],'o-',c='#000099',ms=4.0,mfc='white',mew=1.2) 

		ax.flat[(iG-1)*6+i].set_title('   '+TSamp[i],loc='center',fontsize=14.8,weight='semibold',y=0.85)
		ax.flat[(iG-1)*6+i].set_title('   '+TTSamp[i]+'.',loc='left',fontsize=15.5,weight='semibold',y=0.845)

		#ax.flat[(iG-1)*6+i].text(Col_ENS_N34[i,0]-0.25,Col_ENS_WWV[i,0]-1.0,'Jul(-1)',fontsize=12,weight='semibold')
		#ax.flat[(iG-1)*6+i].text(Col_ENS_N34[i,-1]+0.2,Col_ENS_WWV[i,-1]-0.75,'Feb(+1)',fontsize=12,weight='semibold')

	del [Samp,Col_ENS_WWV,Col_ENS_N34]

#====Canvas setting=====
for i in range(24):
	ax.flat[i].set_xlim(-2.2,3.0)
	ax.flat[i].set_ylim(-1.05*10.0,1.0*10.0)

	ax.flat[i].set_xticks(np.arange(-2.0,3.1,1.0))
	ax.flat[i].set_xticks(np.arange(-2.0,3.1,0.5),minor=True)
	ax.flat[i].set_xticklabels(['-2.0','-1.0','0.0','1.0','2.0','3.0'],fontsize=10,weight='semibold')

	ax.flat[i].set_yticks(np.arange(-1.0,1.01,0.5)*10.0)
	ax.flat[i].set_yticks(np.arange(-1.0,1.01,0.1)*10.0,minor=True)
	ax.flat[i].set_yticklabels(['-10','-5','0','5','10'],fontsize=10,weight='semibold')


	if (i in [3,5,6,7,10,13,15 ]):#1991,2002,1994,2004,2014
		ax.flat[i].text(-1.75,-0.87*10.0,'La Nina',fontsize=10,weight='semibold')
		ax.flat[i].text(-0.38,-0.87*10.0,' Nino3.4 ',fontsize=10,weight='semibold')
		ax.flat[i].text( 1.25,-0.87*10.0,'El Nino',fontsize=10,weight='semibold')

		ax.flat[i].text( 2.12, 0.2,  'I',fontsize=10,weight='semibold')
		ax.flat[i].text(-1.7,  0.2,'II',fontsize=10,weight='semibold')
		ax.flat[i].text(-1.75,-1.2,'III',fontsize=10,weight='semibold')
		ax.flat[i].text( 2.05,-1.2,'IV',fontsize=10,weight='semibold')

		ax.flat[i].text(-2.74, 0.290*10.0,'Recharge' ,fontsize=8,weight='semibold',rotation=90)
		ax.flat[i].text(-2.76,-0.140*10.0,' OHCA '   ,fontsize=9,weight='semibold',rotation=90)
		ax.flat[i].text(-2.74,-0.615*10.0,'Discharge' ,fontsize=8,weight='semibold',rotation=90)
	else:
		ax.flat[i].text(-2.10, -1.51*10.0,'La Nina',fontsize=10,weight='semibold')
		ax.flat[i].text(-0.40, -1.51*10.0,' Nino3.4 ',fontsize=10,weight='semibold')
		ax.flat[i].text( 1.50, -1.51*10.0,'El Nino',fontsize=10,weight='semibold')		
		ax.flat[i].text( 2.62, 0.5,  'I',fontsize=10,weight='semibold')
		ax.flat[i].text(-2.1,  0.5,'II',fontsize=10,weight='semibold')
		ax.flat[i].text(-2.15,-1.5,'III',fontsize=10,weight='semibold')
		ax.flat[i].text( 2.55,-1.5,'IV',fontsize=10,weight='semibold')
	
		ax.flat[i].text(-3.30, 0.46*10.0,'Recharge' ,fontsize=8,weight='semibold',rotation=90)
		ax.flat[i].text(-3.32,-0.28*10.0,' OHCA '   ,fontsize=9,weight='semibold',rotation=90)
		ax.flat[i].text(-3.30,-1.10*10.0,'Discharge' ,fontsize=8,weight='semibold',rotation=90)

	if (i in [3,5,6,7,10,13,15 ]):#2002
		ax.flat[i].set_xlim(-1.83,2.5)
		ax.flat[i].set_ylim(-0.6*10.0,0.6*10.0)

		ax.flat[i].set_xticks(np.arange(-1.0,2.51,1.0))
		ax.flat[i].set_xticks(np.arange(-1.5,2.51,0.5),minor=True)
		ax.flat[i].set_xticklabels(['-1.0','0.0','1.0','2.0'],fontsize=10,weight='semibold')
	
		ax.flat[i].set_yticks(np.arange(-0.5,0.501,0.5)*10.0)
		ax.flat[i].set_yticks(np.arange(-0.6,0.601,0.1)*10.0,minor=True)
		ax.flat[i].set_yticklabels(['-5','0','5'],fontsize=9.5,weight='semibold')


	#====Add Arrow=====
	ax.flat[i].annotate('', xy=(-0.19, 0.60),xytext=(-0.19, 0.74), xycoords='axes fraction', arrowprops=dict(arrowstyle="<-", color='k',linewidth=1.0))	
	ax.flat[i].annotate('', xy=(-0.19, 0.39),xytext=(-0.19, 0.25), xycoords='axes fraction', arrowprops=dict(arrowstyle="<-", color='k',linewidth=1.0))	

	ax.flat[i].annotate('', xy=(0.365, -0.2),xytext=(0.25,  -0.2), xycoords='axes fraction', arrowprops=dict(arrowstyle="<-", color='k',linewidth=1.0))	
	ax.flat[i].annotate('', xy=(0.590, -0.2),xytext=(0.720, -0.2), xycoords='axes fraction', arrowprops=dict(arrowstyle="<-", color='k',linewidth=1.0))		


	ax.flat[i].tick_params(axis='x',width=1.5,length=6)
	ax.flat[i].tick_params(axis='y',width=1.5,length=6)
	
	ax.flat[i].tick_params(axis='x',width=1.5,length=3,which='minor')
	ax.flat[i].tick_params(axis='y',width=1.5,length=3,which='minor')
	
	
	for spi in ['bottom','left','top','right']:
		ax.flat[i].spines[spi].set_linewidth(1.5)



tmp_font = font_manager.FontProperties(fname='/home/sunming/miniconda3/envs/normal/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/Helvetica-Bold.ttf')

for i in range(6):
	if i in [0,1,2,4]:
		ax.flat[i].text(2.3,-0.95*10.0,r'$\checkmark$',fontproperties=tmp_font,fontsize=22,color='#990000')
	if i in [3,5]:
		ax.flat[i].text(1.8,-0.55*10.0,r'$\checkmark$',fontproperties=tmp_font,fontsize=22,color='#990000')


for i in range(6,12):
	if i in [8,9,11]:
		ax.flat[i].plot(2.5,-0.85*10.0,'x',mfc='none',mew=2.0,mec='#990000',ms=9)
	if i in [6,7,10]:
		ax.flat[i].plot(2.0,-0.48*10.0,'x',mfc='none',mew=2.0,mec='#990000',ms=9)



for i in range(12,18):
	if i in [0+12,2+12,4+12,5+12]:
		ax.flat[i].text(2.3,-0.95*10.0,r'$\checkmark$',fontproperties=tmp_font,fontsize=22,color='#0066CC')
	if i in [1+12,3+12]:
		ax.flat[i].text(1.8,-0.55*10.0,r'$\checkmark$',fontproperties=tmp_font,fontsize=22,color='#0066CC')

ax.flat[18].plot(2.5,-0.85*10.0,'x',mfc='none',mew=2.0,mec='#0066CC',ms=9)

for i in range(19,21):
	ax.flat[i].text(2.3,-0.95*10.0,r'$\checkmark$',fontproperties=tmp_font,fontsize=22,color='#000099')
for i in range(21,24):
	ax.flat[i].plot(2.5,-0.85*10.0,'x',mfc='none',mew=2.0,mec='#000099',ms=9)


plt.tight_layout()
plt.subplots_adjust(wspace=0.35,hspace=0.35)
plt.savefig('ENSO_PhaseOrbits.svg',dpi=800)


