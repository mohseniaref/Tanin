# Tanin user guide

Tanin provides xarray-native harmonic and spectral analysis for irregular
geodetic time series, including InSAR cubes.

## Fit a 3-D InSAR cube

Assume a displacement cube with dimensions `(time, y, x)`:

```python
import tanin

fit = tanin.fit_geodetic(
    da,
    dim="time",
    annual_period=365.25,
    seasonal_period=182.625,
)
```

The result is an xarray `Dataset` with one 2-D `(y, x)` field for each fitted
quantity:

```python
fit.trend
fit.trend_variance
fit.trend_uncertainty
fit.intercept
fit.intercept_uncertainty
fit.annual_amplitude
fit.annual_phase
fit.annual_amplitude_uncertainty
fit.annual_phase_uncertainty
fit.seasonal_amplitude
fit.seasonal_phase
fit.seasonal_amplitude_uncertainty
fit.seasonal_phase_uncertainty
```

`trend` is in data units per day when the time coordinate is converted to
decimal days. `trend_variance` is in squared trend units, and
`trend_uncertainty` is its square root.

## Phase convention

Tanin fits each harmonic as:

```text
a cos(omega t) + b sin(omega t)
```

It reports:

```text
amplitude = sqrt(a^2 + b^2)
phase     = atan2(-b, a)
```

Therefore the equivalent representation is:

```text
amplitude cos(omega t + phase)
```

The phase is in radians. It is only meaningful together with the stated time
origin and period.

## LS-VCE covariance-aware fitting

Estimate a stochastic covariance model from a representative pixel or region:

```python
noise = tanin.estimate_noise_vce(
    da.isel(y=100, x=200).values,
    da.time.values,
    components=("white", "flicker", "random_walk"),
    non_negative=True,
)

fit = tanin.fit_geodetic(
    da,
    dim="time",
    covariance=noise["covariance"],
)
```

The covariance matrix must correspond to the time dimension. A common
covariance can be applied to all pixels; per-pixel covariance models are not
yet supported by `fit_geodetic`.

## Test specified periodic signals

Do not assume that an annual signal exists. Test it explicitly:

```python
annual = tanin.test_period(da, period=365.25, dim="time")
semiannual = tanin.test_period(da, period=182.625, dim="time")
```

For a covariance-aware test:

```python
annual = tanin.test_period(
    da,
    period=365.25,
    dim="time",
    covariance=noise["covariance"],
)
```

The result includes amplitude, phase, power, F statistic, p-value, and a
nominal 0.05 significance flag. For many pixels, apply a multiple-testing
procedure such as false-discovery-rate control before interpreting a map of
significant pixels.

## Search for candidate periods

```python
spectrum = tanin.spectrum(
    da,
    dim="time",
    min_period=30,
    max_period=1000,
    method="lssa",
)

dominant = tanin.dominant_period(spectrum)
```

Finding a spectral maximum does not by itself establish statistical
significance. Use a specified-period test or a properly corrected search test.

## Read a MintPy time series

```python
da = tanin.io.open_mintpy("timeseries.h5")
fit = tanin.fit_geodetic(da)
```

MintPy support is optional and currently focuses on reading the time-series
array, dates, and root metadata. Always inspect the units, reference date,
reference pixel, and masks before interpreting fitted displacement fields.

## Diagnostics and examples

Runnable examples are in `examples/`:

- `plot_geodetic_cube_fit.py`
- `plot_lsvce_workflow.py`
- `plot_seasonal_detection.py`
- `plot_jump_detection.py`
- `plot_advanced_methods.py`

Run one with:

```bash
PYTHONPATH=src python examples/plot_geodetic_cube_fit.py
```
