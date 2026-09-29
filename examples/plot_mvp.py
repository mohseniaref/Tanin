"""Generate a small Tanin MVP diagnostic plot."""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import tanin

rng = np.random.default_rng(7)
t = np.arange(0.0, 4 * 365.25, 12.0)
truth = 0.002 * t + 2.0 * np.cos(2 * np.pi * t / 365.25 + 0.4)
y = truth + rng.normal(0, 0.25, t.size)
da = xr.DataArray(y, dims="time", coords={"time": t})

from tanin.harmonic import _fit_1d
fit = _fit_1d(t, y, periods=(365.25,))
amp = fit["amplitude_365.25"]
phase = fit["phase_365.25"]
model = fit["coefficients"][0] + fit["coefficients"][1] * (t - t.mean()) + amp * np.cos(2 * np.pi * t / 365.25 + phase)

s = tanin.spectrum(da, min_period=100, max_period=800, n_periods=300)

fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
axes[0].plot(t, y, ".", ms=3, alpha=0.65, label="observations")
axes[0].plot(t, truth, "--", lw=1.5, label="truth")
axes[0].plot(t, model, lw=2, label="fitted annual model")
axes[0].set(xlabel="Days", ylabel="Signal", title=f"Annual harmonic fit: amplitude={amp:.3f}")
axes[0].legend()

axes[1].semilogx(s["period"], s, lw=1.8)
axes[1].axvline(365.25, color="tab:red", ls="--", label="365.25 days")
axes[1].set(xlabel="Period (days)", ylabel="LSSA power", title="Least-squares spectrum")
axes[1].legend()
fig.savefig("examples/tanin_mvp_diagnostic.png", dpi=160)
print("annual_test", tanin.test_period(da, 365.25))
print("plot", "examples/tanin_mvp_diagnostic.png")
