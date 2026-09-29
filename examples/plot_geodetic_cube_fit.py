import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import tanin

rng=np.random.default_rng(31); t=np.arange(0.,4*365.25,12.); y,x=np.mgrid[0:20,0:24]; annual=.5+.03*x; seasonal=.2+.01*y; trend=.001+.0001*x; phase=.2+.02*y
signal=(trend[...,None]*t+annual[...,None]*np.cos(2*np.pi*t/365.25+phase[...,None])+seasonal[...,None]*np.cos(2*np.pi*t/182.625)+rng.normal(0,.05,(20,24,len(t))))
da=xr.DataArray(signal,dims=("y","x","time"),coords={"y":y[:,0],"x":x[0],"time":t})
r=tanin.fit_geodetic(da)
fig,ax=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
fields=[("trend","Trend"),("trend_uncertainty","Trend uncertainty"),("annual_amplitude","Annual amplitude"),("annual_phase","Annual phase"),("seasonal_amplitude","Semiannual amplitude"),("seasonal_phase","Semiannual phase")]
for a,(name,title) in zip(ax.flat,fields): im=a.imshow(r[name],origin="lower"); a.set_title(title); fig.colorbar(im,ax=a,shrink=.8)
fig.savefig('examples/tanin_geodetic_cube_fit.png',dpi=160)
print(r); print('plot examples/tanin_geodetic_cube_fit.png')
