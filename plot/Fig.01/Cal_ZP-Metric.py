import numpy  as np
import xarray as xr
from scipy.stats import pearsonr

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


def obtain_metric(X,Y):
	R0  = 0.999

	TSS = np.zeros(X.shape[1],dtype=float)
	SCC = np.zeros(X.shape[1],dtype=float)
	PCC = np.zeros(X.shape[1],dtype=float)


	for icase in range(X.shape[1]):
		OBS  = X[:,icase]
		Pred = Y[:,icase]

		PCC[icase] = np.corrcoef(OBS,Pred)[0][1]
		SDR  = np.std(Pred)/np.std(OBS)

		TSS[icase]  = (4.0*(1.0+PCC[icase])**4)/((SDR+1.0/SDR)**2)/((1+R0)**4)
		SCC[icase]   = (1.0+PCC[icase])*(1.0+PCC[icase])/(SDR+1.0/SDR)/(SDR+1.0/SDR)

	return np.mean(PCC),np.mean(TSS),np.mean(SCC)



f   = xr.open_dataset("../../data/Assess/CNN-CD_ZonalPattern_Prediction.nc")
OBS =f['OBS'].values  #13X18X432
PRED=f['Pred'].values #13X18X432


for i in range(18):
	N34_OBS = np.mean(OBS[6:10,i,:],axis=0)
	N34_CNN = np.mean(PRED[6:10,i,:],axis=0)

	N34_OBS = runaveMid(N34_OBS,3)
	N34_CNN = runaveMid(N34_CNN,3) 


PCC = np.arange(18,dtype=float)
TSS = np.arange(18,dtype=float)
SCC = np.arange(18,dtype=float)

##====Step2====DJF Zonal pattern metric=============
for i in range(18):
	if (i==0):
		DJF_OBS = ( OBS[:,i,24-i:430-i:12]+ OBS[:,i+1,25-i:430-i:12])/2.0
		DJF_CNN = (PRED[:,i,24-i:430-i:12]+PRED[:,i+1,25-i:430-i:12])/2.0
	if (i==17):
		DJF_OBS = ( OBS[:,i-1,23-i:430-i:12]+ OBS[:,i,24-i:430-i:12])/2.0
		DJF_CNN = (PRED[:,i-1,23-i:430-i:12]+PRED[:,i,24-i:430-i:12])/2.0
	if (i>0 and i<17):
		DJF_OBS = ( OBS[:,i-1,23-i:430-i:12]+ OBS[:,i,24-i:430-i:12]+ OBS[:,i+1,24-i:430-i:12])/3.0
		DJF_CNN = (PRED[:,i-1,23-i:430-i:12]+PRED[:,i,24-i:430-i:12]+PRED[:,i+1,24-i:430-i:12])/3.0

	PCC[i],TSS[i],SCC[i] = obtain_metric(DJF_OBS,DJF_CNN)

PCC = runaveMid(runaveMid(PCC,3),3)
TSS = runaveMid(runaveMid(TSS,3),3)
SCC = runaveMid(runaveMid(SCC,3),3)

ds1 = xr.Dataset({'PCC': (('lead'), PCC)},coords={'lead': np.arange(1,19,1)})
ds2 = xr.Dataset({'SCC': (('lead'), SCC)},coords={'lead': np.arange(1,19,1)})

ds1.to_netcdf('Skill_DJF-SSTA_zonal_pattern.nc')
ds2.to_netcdf('Skill_DJF-SSTA_zonal_pattern.nc',mode="a")