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

def test_powerlaw_reader_and_uncertainty(tmp_path):
    t=np.arange(0.,400.,5.); q=tanin.powerlaw_covariance(t,exponent=-1.0)
    assert q.shape==(len(t),len(t)) and np.allclose(q,q.T)
    path=tmp_path/"series.txt"; np.savetxt(path,np.column_stack((t,np.sin(t/30))))
    rt,ry=tanin.read_time_series(path)
    result=tanin.estimate_noise_vce(ry,rt,components=("white","power_law"),power_exponent=-1,max_iter=5)
    assert set(result["standard_errors"])=={"white","power_law"}
    assert result["component_covariance"].shape==(2,2)
