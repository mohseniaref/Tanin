"""Independent implementations of the broader LSWAVE/JUST-style utilities.

These routines use the same least-squares principles as the cited packages,
but do not copy their source code. They are intentionally small and composable.
"""
import numpy as np
import xarray as xr
from scipy.optimize import minimize_scalar
from scipy.stats import f
from .harmonic import _fit_1d
from .spectrum import _periods, _spec

def allssa(y, t=None, min_period=30, max_period=1000, n_periods=256, n_harmonics=1):
    """Anti-leakage LSSA approximation via iterative prewhitening.

    At each iteration the strongest least-squares frequency is fitted jointly
    with previously selected components, then removed before the next search.
    Returns selected periods, coefficients, amplitudes, phases, and residuals.
    """
    y=np.asarray(y,float); t=np.arange(y.size,dtype=float) if t is None else np.asarray(t,float)
    mask=np.isfinite(y)&np.isfinite(t); y,t=y[mask],t[mask]; periods=_periods(min_period,max_period,n_periods); residual=y.copy(); selected=[]
    for _ in range(n_harmonics):
        power=_spec(t,residual,periods,"lssa"); i=int(np.argmax(power)); lo=periods[max(0,i-1)]; hi=periods[min(len(periods)-1,i+1)]
        if hi > lo:
            objective=lambda p: -float(_spec(t,residual,np.array([p]),"lssa")[0])
            p=float(minimize_scalar(objective,bounds=(lo,hi),method="bounded").x)
        else: p=float(periods[i])
        selected.append(p)
        fit=_fit_1d(t,y,tuple(selected)); residual=y-(fit["coefficients"][0]+fit["coefficients"][1]*(t-t.mean())+sum(fit["coefficients"][2+2*i]*np.cos(2*np.pi*t/q)+fit["coefficients"][3+2*i]*np.sin(2*np.pi*t/q) for i,q in enumerate(selected)))
    fit=_fit_1d(t,y,tuple(selected)); amps=[fit[f"amplitude_{p:g}"] for p in selected]; phases=[fit[f"phase_{p:g}"] for p in selected]
    return {"periods":np.asarray(selected),"amplitudes":np.asarray(amps),"phases":np.asarray(phases),"coefficients":fit["coefficients"],"residuals":residual}

def lswa(y, t=None, periods=(365.25,), window=None, step=None):
    """Least-squares wavelet-style localized harmonic amplitudes."""
    y=np.asarray(y,float); t=np.arange(y.size,dtype=float) if t is None else np.asarray(t,float); window=window or max(np.ptp(t)/4,1); step=step or window/8
    centers=np.arange(np.min(t)+window/2,np.max(t)-window/2+step/2,step); out=np.full((len(centers),len(periods)),np.nan)
    for i,c in enumerate(centers):
        m=np.abs(t-c)<=window/2
        if np.count_nonzero(m)>=4+2*len(periods):
            r=_fit_1d(t[m],y[m],periods); out[i]=[r[f"amplitude_{p:g}"] for p in periods]
    return {"time":centers,"periods":np.asarray(periods),"amplitude":out}

def lscsa(x,y,t=None,periods=None):
    """Least-squares cross-spectral amplitude, phase, and coherence."""
    x=np.asarray(x,float); y=np.asarray(y,float); t=np.arange(len(x),dtype=float) if t is None else np.asarray(t,float); periods=_periods(30,1000,256) if periods is None else np.asarray(periods,float); mask=np.isfinite(x)&np.isfinite(y)&np.isfinite(t); t,x,y=t[mask],x[mask],y[mask]; result=[]
    for p in periods:
        w=2*np.pi/p; X=np.column_stack((np.ones(len(t)),t-t.mean(),np.cos(w*t),np.sin(w*t))); bx=np.linalg.lstsq(X,x,rcond=None)[0]; by=np.linalg.lstsq(X,y,rcond=None)[0]; zx=bx[2]-1j*bx[3]; zy=by[2]-1j*by[3]; result.append((zx*np.conj(zy),abs(zx),abs(zy)))
    result=np.asarray(result); return {"period":periods,"cross_power":result[:,0],"amplitude_x":result[:,1],"amplitude_y":result[:,2],"coherence":np.abs(result[:,0])/np.maximum(result[:,1]*result[:,2],np.finfo(float).eps)}

def lscwa(x, y, t=None, periods=None, cycles_per_window=6.0):
    """Least-squares cross-wavelet-style localized complex spectrum.

    A Gaussian-localized weighted harmonic regression is evaluated at each
    time/period point. The result is analogous to a cross-wavelet transform,
    while retaining irregular-time least-squares fitting.
    """
    x=np.asarray(x,float); y=np.asarray(y,float); t=np.arange(len(x),dtype=float) if t is None else np.asarray(t,float)
    periods=_periods(30,1000,128) if periods is None else np.asarray(periods,float); mask=np.isfinite(x)&np.isfinite(y)&np.isfinite(t); t,x,y=t[mask],x[mask],y[mask]
    out=np.full((len(periods),len(t)),np.nan+0j)
    for j,p in enumerate(periods):
        width=max(p/cycles_per_window, np.finfo(float).eps)
        for i,center in enumerate(t):
            w=np.exp(-0.5*((t-center)/width)**2); X=np.column_stack((np.ones(len(t)), np.cos(2*np.pi*t/p), np.sin(2*np.pi*t/p)))
            sw=np.sqrt(w); bx=np.linalg.lstsq(X*sw[:,None],x*sw,rcond=None)[0]; by=np.linalg.lstsq(X*sw[:,None],y*sw,rcond=None)[0]
            out[j,i]=(bx[1]-1j*bx[2])*np.conj(by[1]-1j*by[2])
    return {"time":t,"period":periods,"cross_power":out,"power":np.abs(out)}

def detect_jumps(y,t=None,threshold=5.0):
    """Detect robust step candidates from first differences."""
    y=np.asarray(y,float); t=np.arange(len(y),dtype=float) if t is None else np.asarray(t,float); d=np.diff(y); med=np.nanmedian(d); scale=1.4826*np.nanmedian(np.abs(d-med)); scale=max(scale,np.finfo(float).eps); idx=np.flatnonzero(np.abs(d-med)>threshold*scale)+1
    return {"indices":idx,"times":t[idx],"step_estimates":d[idx-1]-med,"scale":scale}

def just_jumps(y, t=None, periods=(), max_jumps=5, alpha=0.01, min_separation=None):
    """Iterative joint trend/seasonal step detection.

    Candidate steps are columns ``I(t >= tau)`` added to a least-squares
    functional model. At each iteration the candidate with the largest nested
    F statistic is selected if its p-value is below ``alpha``. This is an
    independent, compact implementation of the JUSTjumps model-selection idea;
    it is not a byte-for-byte reproduction of the original package.
    """
    y=np.asarray(y,float); t=np.arange(len(y),dtype=float) if t is None else np.asarray(t,float)
    mask=np.isfinite(y)&np.isfinite(t); y,t=y[mask],t[mask]
    order=np.argsort(t); t,y=t[order],y[order]; n=len(y)
    if n < 8: raise ValueError("at least 8 finite observations are required")
    periods=tuple(periods); center=t.mean(); base=np.column_stack((np.ones(n),t-center,*[z for p in periods for z in (np.cos(2*np.pi*t/p),np.sin(2*np.pi*t/p))]))
    min_separation = min_separation if min_separation is not None else max(np.median(np.diff(t))*2, np.finfo(float).eps)
    selected=[]; columns=[]; history=[]
    for _ in range(max_jumps):
        X0=np.column_stack((base,*columns)) if columns else base; b0=np.linalg.lstsq(X0,y,rcond=None)[0]; rss0=float(np.sum((y-X0@b0)**2)); candidates=[]
        for i in range(2,n-2):
            tau=float(t[i])
            if any(abs(tau-q)<min_separation for q in selected): continue
            step=(t>=tau).astype(float); X1=np.column_stack((X0,step)); b1=np.linalg.lstsq(X1,y,rcond=None)[0]; rss1=float(np.sum((y-X1@b1)**2)); df2=n-X1.shape[1]
            if df2 <= 0: continue
            statistic=((rss0-rss1)/1)/(rss1/df2); pvalue=float(f.sf(max(statistic,0),1,df2)); candidates.append((pvalue,statistic,tau,b1[-1],step,rss1))
        if not candidates: break
        pvalue,statistic,tau,step_size,step,rss1=min(candidates,key=lambda a:a[0])
        if pvalue >= alpha: break
        selected.append(tau); columns.append(step); history.append({"time":tau,"step":float(step_size),"statistic":float(statistic),"p_value":pvalue,"rss":rss1})
    X=np.column_stack((base,*columns)) if columns else base; beta=np.linalg.lstsq(X,y,rcond=None)[0]; fitted=X@beta
    return {"times":np.asarray(selected),"steps":np.asarray([h["step"] for h in history]),"p_values":np.asarray([h["p_value"] for h in history]),"statistics":np.asarray([h["statistic"] for h in history]),"fitted":fitted,"residuals":y-fitted,"history":history,"time":t}

def just_decompose(y,t=None,periods=(365.25,), threshold=5.0):
    """Decompose into trend, selected seasonal terms, jumps, and remainder."""
    y=np.asarray(y,float); t=np.arange(len(y),dtype=float) if t is None else np.asarray(t,float); fit=_fit_1d(t,y,periods); baseline=fit["coefficients"][0]+fit["coefficients"][1]*(t-t.mean()); seasonal=np.zeros(len(t));
    for i,p in enumerate(periods): seasonal += fit["coefficients"][2+2*i]*np.cos(2*np.pi*t/p)+fit["coefficients"][3+2*i]*np.sin(2*np.pi*t/p)
    jumps=just_jumps(y,t,periods=periods,alpha=min(0.05,max(1e-6,1/(threshold**2))))
    return {"trend":baseline,"seasonal":seasonal,"remainder":jumps["residuals"],"jumps":jumps,"fitted":jumps["fitted"]}

def just_monitor(y,t=None,window=50,threshold=5.0):
    """Rolling robust jump monitor."""
    y=np.asarray(y,float); t=np.arange(len(y),dtype=float) if t is None else np.asarray(t,float); scores=np.full(len(y),np.nan)
    for i in range(window,len(y)):
        d=y[i]-np.nanmedian(y[i-window:i]); scale=1.4826*np.nanmedian(np.abs(y[i-window:i]-np.nanmedian(y[i-window:i]))); scores[i]=d/max(scale,np.finfo(float).eps)
    return {"time":t,"score":scores,"alarms":np.flatnonzero(np.abs(scores)>=threshold)}
