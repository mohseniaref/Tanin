import numpy as np
import xarray as xr
from .time import to_decimal_days
from .models import design_matrix
from .statistics import nested_f_test

def _fit_1d(t, y, periods=(), weights=None, covariance=None, trend=True):
    mask = np.isfinite(t) & np.isfinite(y)
    if weights is not None: mask &= np.isfinite(weights) & (weights > 0)
    t, y = np.asarray(t)[mask], np.asarray(y)[mask]
    w = None if weights is None else np.asarray(weights)[mask]
    if t.size < 3: raise ValueError("at least 3 finite observations are required")
    X, names = design_matrix(t, periods, trend=trend)
    if np.linalg.matrix_rank(X) < X.shape[1]: raise ValueError("rank-deficient design matrix")
    if covariance is not None:
        C = np.asarray(covariance)[np.ix_(mask, mask)]
        L = np.linalg.cholesky(C); X, y = np.linalg.solve(L, X), np.linalg.solve(L, y); w = None
    if w is None:
        beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None); residual = y-X@beta; rss=float(residual@residual); n_eff=t.size
        cov = np.linalg.pinv(X.T@X) * rss/max(t.size-X.shape[1], 1)
    else:
        sw=np.sqrt(w); Xw=X*sw[:,None]; yw=y*sw; beta, _, _, _=np.linalg.lstsq(Xw,yw,rcond=None); residual=y-X@beta; rss=float(np.sum(w*residual**2)); cov=np.linalg.pinv(Xw.T@Xw)*rss/max(t.size-X.shape[1],1); n_eff=t.size
    out={"coefficients": beta, "coefficient_covariance": cov, "residuals": residual, "rss": rss, "n_observations": n_eff, "names": names}
    for i,p in enumerate(periods):
        a,b=beta[2+2*i:4+2*i]; out[f"amplitude_{p:g}"]=float(np.hypot(a,b)); out[f"phase_{p:g}"]=float(np.arctan2(-b,a))
    return out

def fit_harmonics(data, periods=(), dim="time", weights=None, trend=True):
    if not isinstance(data, xr.DataArray):
        t=np.arange(len(data), dtype=float); return _fit_1d(t, data, periods, weights, trend=trend)
    t=to_decimal_days(data[dim].values); ncoef=2+2*len(periods)
    def f(y):
        r=_fit_1d(t,y,periods,weights=None if weights is None else np.asarray(weights),trend=trend)
        return r["coefficients"], np.array([r[f"amplitude_{p:g}"] for p in periods]), np.array([r[f"phase_{p:g}"] for p in periods]), r["rss"], float(r["n_observations"])
    inputs=[data] if weights is None else [data, weights]
    core=[ [dim] ] if weights is None else [[dim],[dim]]
    out=xr.apply_ufunc(f,*inputs,input_core_dims=core,output_core_dims=[["coefficient"],["period"],["period"],[],[]],vectorize=True,dask="parallelized",output_dtypes=[float]*5,dask_gufunc_kwargs={"output_sizes":{"coefficient":ncoef,"period":len(periods)}})
    coeff,amp,phase,rss,nobs=out
    coefficient_names=["intercept","trend"]+[name for p in periods for name in (f"cos_{p:g}",f"sin_{p:g}")]
    period_values=np.asarray(periods,dtype=float)
    return xr.Dataset({"coefficients":coeff.assign_coords(coefficient=coefficient_names),"amplitude":amp.assign_coords(period=period_values),"phase":phase.assign_coords(period=period_values),"rss":rss,"n_observations":nobs})

def test_period(data, period, dim="time", weights=None, covariance=None):
    if isinstance(data, xr.DataArray):
        t=to_decimal_days(data[dim].values)
        def one(y): return _test(t,y,period,weights,covariance)
        vals=xr.apply_ufunc(one,data,input_core_dims=[[dim]],output_core_dims=[[],[],[],[],[],[]],vectorize=True,dask="parallelized",output_dtypes=[float]*6)
        return xr.Dataset(dict(amplitude=vals[0],phase=vals[1],power=vals[2],statistic=vals[3],p_value=vals[4],significant=vals[5].astype(bool))).assign_attrs(phase_convention="atan2(-sine, cosine), phase of cosine-equivalent A cos(wt + phase)")
    return _test(np.arange(len(data),dtype=float), data, period, weights, covariance)

def _test(t,y,period,weights=None,covariance=None):
    null=_fit_1d(t,y,(),weights,covariance); alt=_fit_1d(t,y,(period,),weights,covariance); stat,p=nested_f_test(null["rss"],alt["rss"],alt["n_observations"],2); amp=alt[f"amplitude_{period:g}"]; return amp,alt[f"phase_{period:g}"],float((null["rss"]-alt["rss"])/max(null["rss"],np.finfo(float).eps)),stat,p,float(p<0.05)

def fit_geodetic(data, dim="time", annual_period=365.25, seasonal_period=182.625,
                 weights=None, covariance=None):
    """Fit trend, annual, and semiannual terms with delta-method uncertainties.

    Returns one xarray Dataset variable per requested quantity. ``trend`` is
    expressed in data units per day when the time coordinate is datetime-like
    or day-valued. A common covariance matrix may be supplied for all spatial
    series; it must correspond to the time dimension.
    """
    if not isinstance(data, xr.DataArray): raise TypeError("fit_geodetic requires an xarray.DataArray")
    periods=(float(annual_period),float(seasonal_period)); t=to_decimal_days(data[dim].values); ncoef=2+2*len(periods)
    def f(y):
        r=_fit_1d(t,y,periods,weights=None if weights is None else np.asarray(weights),covariance=covariance)
        beta=r["coefficients"]; V=r["coefficient_covariance"]; out=[beta[0],beta[1],np.sqrt(max(V[0,0],0)),np.sqrt(max(V[1,1],0))]
        for i in range(len(periods)):
            a,b=beta[2+2*i:4+2*i]; A=max(np.hypot(a,b),np.finfo(float).eps); ga=np.array([a/A,b/A]); gp=np.array([b/A**2,-a/A**2]); sub=V[2+2*i:4+2*i,2+2*i:4+2*i]
            out.extend((A,np.arctan2(-b,a),np.sqrt(max(ga@sub@ga,0)),np.sqrt(max(gp@sub@gp,0))))
        out.extend((r["rss"],r["n_observations"]-ncoef)); return np.asarray(out,float)
    names=["intercept","trend","intercept_uncertainty","trend_uncertainty"]
    for label in ("annual","seasonal"): names.extend((f"{label}_amplitude",f"{label}_phase",f"{label}_amplitude_uncertainty",f"{label}_phase_uncertainty"))
    names.extend(("rss","degrees_of_freedom")); result=xr.apply_ufunc(f,data,input_core_dims=[[dim]],output_core_dims=[["statistic"]],vectorize=True,dask="parallelized",output_dtypes=[float],dask_gufunc_kwargs={"output_sizes":{"statistic":len(names)}})
    return result.assign_coords(statistic=names).to_dataset(dim="statistic")
