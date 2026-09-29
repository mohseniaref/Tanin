import numpy as np

def design_matrix(t, periods=(), trend=True, intercept=True):
    cols, names = [], []
    if intercept: cols.append(np.ones_like(t)); names.append("intercept")
    if trend: cols.append(t - np.mean(t)); names.append("trend")
    for p in np.atleast_1d(periods):
        w = 2*np.pi/float(p)
        cols.extend((np.cos(w*t), np.sin(w*t))); names.extend((f"cos_{p:g}", f"sin_{p:g}"))
    return np.column_stack(cols), names

def time_series_design(t, periods=(), offsets=(), trend=True, intercept=True):
    """Build a functional design matrix for trends, harmonics, and steps."""
    X, names = design_matrix(t, periods=periods, trend=trend, intercept=intercept)
    t=np.asarray(t,float)
    for offset in offsets:
        X=np.column_stack((X,(t >= float(offset)).astype(float))); names.append(f"offset_{float(offset):g}")
    return X, names
