# Tanin

Tanin is an xarray-native spectral analysis library for geodetic time series.
The first MVP supports weighted harmonic regression, nested-model period tests,
least-squares spectral analysis, Lomb–Scargle spectra, synthetic simulation, and
an optional MintPy HDF5 reader.

LS-VCE noise estimation is also available:

```python
noise = tanin.estimate_noise(
    da, components=("white", "flicker", "random_walk"), non_negative=True
)
```

The returned values multiply unit covariance matrices. Flicker noise uses a
finite-band numerical 1/f approximation; random walk uses a Brownian covariance.

```python
import tanin

annual = tanin.test_period(da, period=365.25, dim="time")
spectrum = tanin.spectrum(da, min_period=30, max_period=1000, dim="time")
dominant = tanin.dominant_period(spectrum)
```

Finding a spectral maximum does not by itself establish statistical significance.
Use `test_period` for a specified, nested-model test and inspect its p-value.
LSSA and Lomb–Scargle both fit sinusoids to observations; FFT methods instead
assume a regular grid. Harmonic regression estimates a selected model, while a
spectrum evaluates many candidate frequencies and therefore requires care about
multiple testing and noise color.

Turning-point analysis is available through the independent STPD-style API:

```python
result = tanin.stpd(time, displacement)
```

It estimates connected piecewise-linear trends and sequentially selects
turning points. The implementation is intended for irregular geodetic series
and is validated with synthetic examples in `tests/test_stpd.py`.

## Citation

If you use Tanin in research, please cite the software repository:

> Aref, M. (2026). Tanin: xarray-native spectral analysis for geodetic time
> series. GitHub. https://github.com/mohseniaref/Tanin

Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff).

Tanin's scientific methods are informed by:

- Teunissen, P. J. G., & Amiri-Simkooei, A. R. (2008). Least-squares variance
  component estimation. *Journal of Geodesy*, 82, 65–82.
- Amiri-Simkooei, A. R., Tiberius, C. C. J. M., & Teunissen, P. J. G. (2007).
  Assessment of noise in GPS coordinate time series: Methodology and results.
  *Journal of Geophysical Research: Solid Earth*, 112, B07413.
- Ghaderpour, E. (2019–2021). LSWAVE / JUST signal-processing packages.
  https://github.com/Ghaderpour/LSWAVE-SignalProcessing
- Ghaderpour, E., et al. (2023). A fast and robust method for detecting trend
  turning points in InSAR displacement time series. *Computers & Geosciences*.
