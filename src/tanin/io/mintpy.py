def open_mintpy(path, dataset="timeseries"):
    try: import h5py
    except ImportError as e: raise ImportError("open_mintpy requires the optional 'h5py' dependency") from e
    import numpy as np, xarray as xr
    with h5py.File(path,"r") as f:
        data=f[dataset][...]; dates=f["date"][...]; attrs={str(k): (v.decode() if isinstance(v,bytes) else v.item() if hasattr(v,"item") else v) for k,v in f.attrs.items()}
    dates=np.array([d.decode() if isinstance(d,bytes) else str(d) for d in dates],dtype="datetime64[D]")
    dims=("time","y","x") if data.ndim==3 else ("time",)
    return xr.DataArray(data,dims=dims,coords={"time":dates},attrs=attrs,name=dataset)
