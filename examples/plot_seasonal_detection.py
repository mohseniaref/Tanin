import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import tanin

rng = np.random.default_rng(5)
t = np.sort(rng.uniform(0, 4 * 365.25, 160))
y = (0.001 * t + 1.5 * np.cos(2*np.pi*t/365.25 + 0.3)
     + 0.7 * np.cos(2*np.pi*t/182.625 - 0.8) + rng.normal(0, 0.12, t.size))
da = xr.DataArray(y, dims="time", coords={"time": t})

annual = tanin.test_period(da, 365.25)
semiannual = tanin.test_period(da, 182.625)
spec = tanin.spectrum(da, min_period=100, max_period=500, n_periods=400)
allssa = tanin.allssa(y, t, min_period=120, max_period=500, n_periods=400, n_harmonics=2)

fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
axes[0].plot(t, y, ".", ms=3, label="observations")
axes[0].plot(t, 0.001*t + 1.5*np.cos(2*np.pi*t/365.25+0.3)
            + 0.7*np.cos(2*np.pi*t/182.625-0.8), lw=2, label="true model")
axes[0].set(xlabel="Days", ylabel="Signal", title="Annual + semiannual synthetic signal")
axes[0].legend()

axes[1].plot(spec.period, spec, lw=1.8, label="LSSA power")
for p, label in [(365.25, "annual"), (182.625, "semiannual")]:
    axes[1].axvline(p, ls="--", label=f"{label} ({p:g} days)")
for p in allssa["periods"]:
    axes[1].axvline(p, color="k", alpha=.35)
axes[1].set(xlabel="Period (days)", ylabel="Power", title="Detected periods")
axes[1].legend()
fig.savefig("examples/tanin_seasonal_detection.png", dpi=160)
print("annual", float(annual.amplitude), float(annual.p_value))
print("semiannual", float(semiannual.amplitude), float(semiannual.p_value))
print("allssa_periods", allssa["periods"])
