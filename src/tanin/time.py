import numpy as np
import pandas as pd

def to_decimal_days(time, origin=None):
    """Convert datetime-like or numeric coordinates to days from *origin*."""
    a = np.asarray(time)
    if np.issubdtype(a.dtype, np.datetime64) or a.dtype.kind in "OUS":
        dt = pd.to_datetime(a)
        if origin is None:
            origin = dt[0]
        return (dt - pd.Timestamp(origin)).total_seconds().to_numpy(dtype=float) / 86400.0
    out = a.astype(float)
    if origin is not None:
        out = out - float(origin)
    return out
