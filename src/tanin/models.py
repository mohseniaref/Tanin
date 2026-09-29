import numpy as np

def design_matrix(t, periods=(), trend=True, intercept=True):
    cols, names = [], []
    if intercept: cols.append(np.ones_like(t)); names.append("intercept")
    if trend: cols.append(t - np.mean(t)); names.append("trend")
    for p in np.atleast_1d(periods):
        w = 2*np.pi/float(p)
        cols.extend((np.cos(w*t), np.sin(w*t))); names.extend((f"cos_{p:g}", f"sin_{p:g}"))
    return np.column_stack(cols), names
