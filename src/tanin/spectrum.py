import numpy as np
import xarray as xr
from scipy.signal import lombscargle
from .time import to_decimal_days
from .harmonic import _fit_1d

def _periods(min_period,max_period,n): return np.geomspace(float(min_period),float(max_period),int(n))
def _spec(t,y,periods,method):
    mask=np.isfinite(t)&np.isfinite(y); t,y=np.asarray(t)[mask],np.asarray(y)[mask]
    if method=="lomb_scargle": power=lombscargle(t,y,2*np.pi/periods,normalize=True,floating_mean=True); return power
    null=_fit_1d(t,y,())["rss"]
    return np.array([max(0.0, null-_fit_1d(t,y,(p,))["rss"]) for p in periods])
def spectrum(data,min_period=30,max_period=1000,n_periods=256,dim="time",method="lssa"):
    periods=_periods(min_period,max_period,n_periods)
    if isinstance(data,xr.DataArray):
        t=to_decimal_days(data[dim].values)
        def f(y): return _spec(t,y,periods,method)
        return xr.apply_ufunc(f,data,input_core_dims=[[dim]],output_core_dims=[["period"]],vectorize=True,dask="parallelized",output_dtypes=[float],dask_gufunc_kwargs={"output_sizes":{"period":len(periods)}}).assign_coords(period=periods).rename("power")
    return xr.DataArray(_spec(np.arange(len(data),dtype=float),data,periods,method),dims="period",coords={"period":periods},name="power")
def lssa(*args,**kwargs): kwargs["method"]="lssa"; return spectrum(*args,**kwargs)
def dominant_period(result):
    p=result if isinstance(result,xr.DataArray) else result["power"]; i=p.argmax("period"); return xr.Dataset({"dominant_period":p["period"].isel(period=i),"dominant_power":p.max("period")})
