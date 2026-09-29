import numpy as np
import xarray as xr
import tanin

def test_period_recovery():
    t=np.arange(0, 4*365, 12.0)
    y=0.002*t+2*np.cos(2*np.pi*t/365.25+0.4)
    da=xr.DataArray(y,dims="time",coords={"time":t})
    r=tanin.test_period(da,365.25)
    assert float(r.amplitude)>1.9
    assert bool(r.significant)

def test_no_period_signal():
    rng=np.random.default_rng(4); t=np.arange(0,730,12.); da=xr.DataArray(.01*t+rng.normal(0,.4,t.size),dims="time",coords={"time":t})
    assert float(tanin.test_period(da,365.25).amplitude) < 1

def test_spectrum_and_cube():
    t=np.arange(0,730,12.); base=np.cos(2*np.pi*t/365.25); cube=np.stack([base,2*base]).reshape(t.size,1,2)
    da=xr.DataArray(cube,dims=("time","y","x"),coords={"time":t})
    s=tanin.spectrum(da,min_period=200,max_period=500,n_periods=32)
    assert s.dims == ("y","x","period")
    assert s.shape[-1]==32

def test_datetime_and_simulation():
    t=np.arange("2020-01-01","2020-01-10",dtype="datetime64[D]")
    assert np.isclose(tanin.to_decimal_days(t)[-1],8)
    assert tanin.simulate(t,harmonics=[(365.25,1,0)]).sizes["time"]==9

def test_xarray_harmonic_result_and_covariance_test():
    t=np.arange(0.,800.,10.); y=0.5*np.cos(2*np.pi*t/365.25)+np.random.default_rng(2).normal(0,.05,t.size)
    da=xr.DataArray(np.stack([y,2*y]),dims=("site","time"),coords={"site":["a","b"],"time":t})
    fit=tanin.fit_harmonics(da,periods=(365.25,),dim="time")
    assert fit.coefficients.dims == ("site","coefficient")
    assert fit.amplitude.dims == ("site","period")
    covariance=np.eye(t.size)*.01
    result=tanin.test_period(da.sel(site="a"),365.25,covariance=covariance)
    assert bool(result.significant)

def test_fit_geodetic_cube_outputs_uncertainties():
    t=np.arange(0.,800.,10.); signal=np.cos(2*np.pi*t/365.25)+.3*np.cos(2*np.pi*t/182.625)
    da=xr.DataArray(np.broadcast_to(signal,(2,3,len(t))),dims=("y","x","time"),coords={"time":t})
    result=tanin.fit_geodetic(da)
    assert result.dims=={"y":2,"x":3}
    assert np.allclose(result.annual_amplitude,1,atol=1e-6)
    assert np.allclose(result.seasonal_amplitude,.3,atol=1e-6)
    assert "trend_uncertainty" in result and "annual_phase_uncertainty" in result
