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
    t=to_decimal_days(data[dim].values)
    def f(y):
        r=_fit_1d(t,y,periods,trend=trend); return np.array([r["coefficients"], *[r[f"amplitude_{p:g}"] for p in periods], *[r[f"phase_{p:g}"] for p in periods], r["rss"]], dtype=float)
    ncoef=2+2*len(periods); names=[f"coef_{i}" for i in range(ncoef)]+[f"amplitude_{p:g}" for p in periods]+[f"phase_{p:g}" for p in periods]+["rss"]
    out=xr.apply_ufunc(f,data,input_core_dims=[[dim]],output_core_dims=[[]],vectorize=True,dask="parallelized",output_dtypes=[float]).to_dataset(name="value")
    return out.assign_coords(statistic=names).set_index(value="statistic") if False else out

def test_period(data, period, dim="time", weights=None):
    if isinstance(data, xr.DataArray):
        t=to_decimal_days(data[dim].values)
        def one(y): return _test(t,y,period,weights)
        vals=xr.apply_ufunc(one,data,input_core_dims=[[dim]],output_core_dims=[[],[],[],[],[],[]],vectorize=True,dask="parallelized",output_dtypes=[float]*6)
        return xr.Dataset(dict(amplitude=vals[0],phase=vals[1],power=vals[2],statistic=vals[3],p_value=vals[4],significant=vals[5].astype(bool))).assign_attrs(phase_convention="atan2(-sine, cosine), phase of cosine-equivalent A cos(wt + phase)")
    return _test(np.arange(len(data),dtype=float), data, period, weights)

def _test(t,y,period,weights=None):
    null=_fit_1d(t,y,(),weights); alt=_fit_1d(t,y,(period,),weights); stat,p=nested_f_test(null["rss"],alt["rss"],alt["n_observations"],2); amp=alt[f"amplitude_{period:g}"]; return amp,alt[f"phase_{period:g}"],float((null["rss"]-alt["rss"])/max(null["rss"],np.finfo(float).eps)),stat,p,float(p<0.05)
