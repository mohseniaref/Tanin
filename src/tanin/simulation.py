import numpy as np
import xarray as xr
def simulate(time, trend=0.0, intercept=0.0, harmonics=None, noise=0.0, seed=None):
    rng=np.random.default_rng(seed); t=np.asarray(time); td=(t-t[0]).astype("timedelta64[s]").astype(float)/86400 if np.issubdtype(t.dtype,np.datetime64) else t.astype(float)-float(t[0]); y=intercept+trend*td
    for period,amp,phase in (harmonics or []): y += amp*np.cos(2*np.pi*td/period+phase)
    return xr.DataArray(y+rng.normal(0,noise,len(y)),dims="time",coords={"time":t})
