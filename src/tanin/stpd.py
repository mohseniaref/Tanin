"""Sequential turning-point detection (STPD) for irregular geodetic series."""
import numpy as np

def _line_fit(t, y):
    X=np.column_stack((np.ones(len(t)),t-t.mean())); b=np.linalg.lstsq(X,y,rcond=None)[0]; r=y-X@b
    return b, float(r@r)

def tptr(t, y, turning_points):
    """Estimate connected piecewise-linear trend for supplied turning points."""
    t=np.asarray(t,float); y=np.asarray(y,float); order=np.argsort(t); t,y=t[order],y[order]; points=np.sort(np.asarray(turning_points,int)); bounds=np.r_[0,points,len(t)]
    slopes=[]; fitted=np.full(len(t),np.nan); segments=[]
    for a,b in zip(bounds[:-1],bounds[1:]):
        if b-a<2: continue
        coeff,_=_line_fit(t[a:b],y[a:b]); slopes.append(coeff[1]); segments.append((a,b,coeff)); fitted[a:b]=coeff[0]+coeff[1]*(t[a:b]-t[a:b].mean())
    return {"time":t,"turning_points":points,"trend":fitted,"slopes":np.asarray(slopes),"segments":segments,"residuals":y-fitted}

def tpdetect(t, y, margin=2):
    """Return candidate turning-point indices where adjacent slopes change sign."""
    t=np.asarray(t,float); y=np.asarray(y,float); candidates=[]
    for i in range(margin,len(y)-margin):
        left=np.polyfit(t[i-margin:i],y[i-margin:i],1)[0]; right=np.polyfit(t[i:i+margin],y[i:i+margin],1)[0]
        if left*right<0: candidates.append((i,abs(right-left)))
    return np.asarray([i for i,_ in sorted(candidates,key=lambda q:q[1],reverse=True)],dtype=int)

def stpd(t, y, max_turning_points=5, min_segment=4, margin=2):
    """Sequentially select turning points for a connected piecewise trend."""
    t=np.asarray(t,float); y=np.asarray(y,float); mask=np.isfinite(t)&np.isfinite(y); t,y=t[mask],y[mask]; order=np.argsort(t); t,y=t[order],y[order]
    selected=[]; history=[]
    for _ in range(max_turning_points):
        candidates=tpdetect(t,y,margin); candidates=[i for i in candidates if i>=min_segment and len(y)-i>=min_segment and all(abs(i-j)>=min_segment for j in selected)]
        if not candidates: break
        current=np.sort(selected); base=tptr(t,y,current); rss0=float(np.nansum(base["residuals"]**2)); scores=[]
        for i in candidates:
            trial=np.sort(selected+[i]); fit=tptr(t,y,trial); rss=float(np.nansum(fit["residuals"]**2)); scores.append((rss0-rss,i,fit))
        gain,i,fit=max(scores,key=lambda q:q[0])
        if gain <= max(np.nanvar(y)*1e-6,1e-12): break
        selected.append(i); history.append({"index":i,"time":float(t[i]),"rss_reduction":float(gain)})
    result=tptr(t,y,np.sort(selected)); result["history"]=history; result["turning_point_times"]=t[np.sort(selected)] if selected else np.array([]); return result
