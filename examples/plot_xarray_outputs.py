import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import tanin

t=np.arange(0.,4*365.25,12.); y=np.stack([np.cos(2*np.pi*t/365.25),2*np.cos(2*np.pi*t/365.25+.2)])
da=xr.DataArray(y,dims=("site","time"),coords={"site":["A","B"],"time":t})
fit=tanin.fit_harmonics(da,periods=(365.25,),dim="time")
fig,ax=plt.subplots(figsize=(8,4.5)); ax.bar(np.arange(2)-.18,fit.amplitude.sel(period=365.25),.36,label='estimated annual amplitude'); ax.bar(np.arange(2)+.18,[1,2],.36,label='true amplitude'); ax.set_xticks([0,1]); ax.set_xticklabels(['A','B']); ax.set_ylabel('Amplitude'); ax.set_title('xarray harmonic-fit output'); ax.legend(); fig.tight_layout(); fig.savefig('examples/tanin_xarray_outputs.png',dpi=160)
print(fit); print('plot examples/tanin_xarray_outputs.png')
