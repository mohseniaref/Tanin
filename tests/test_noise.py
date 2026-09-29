import numpy as np
import xarray as xr
import tanin

def test_vce_nonnegative_components():
    rng = np.random.default_rng(12); t = np.arange(0.0, 900.0, 12.0)
    y = 0.3 + 0.001*t + rng.normal(0, 0.15, t.size)
    result = tanin.estimate_noise_vce(y, t, components=("white", "flicker"), max_iter=10)
    assert all(value >= 0 for value in result["components"].values())
    assert result["covariance"].shape == (t.size, t.size)

def test_vce_xarray_adapter():
    t = np.arange("2020-01-01", "2021-01-01", dtype="datetime64[14D]")
    da = xr.DataArray(np.sin(np.arange(t.size)) * 0.1, dims="time", coords={"time": t})
    assert "white" in tanin.estimate_noise(da, components=("white",), max_iter=5)["components"]
