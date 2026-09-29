"""Simulate correlated geodetic noise and plot LS-VCE estimates."""
import numpy as np
import matplotlib.pyplot as plt
import tanin

rng = np.random.default_rng(21)
t = np.arange(0.0, 6 * 365.25, 12.0)
components = ("white", "flicker", "random_walk")
truth = np.array([0.04, 0.16, 0.0008])
cofactors = tanin.covariance_components(t, components)
cov = sum(s * cofactors[name] for s, name in zip(truth, components))
cov += np.eye(t.size) * 1e-8
noise = np.linalg.cholesky(cov) @ rng.normal(size=t.size)
y = 0.002 * t + noise

result = tanin.estimate_noise_vce(y, t, components=components, max_iter=30)
estimated = np.array([result["components"][name] for name in components])

fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
axes[0].plot(t, y, ".", ms=3, label="simulated observations")
axes[0].plot(t, 0.002 * t, lw=2, label="functional model")
axes[0].set(xlabel="Days", ylabel="Signal", title="Synthetic white + flicker + random-walk noise")
axes[0].legend()

x = np.arange(len(components)); width = 0.36
axes[1].bar(x - width/2, truth, width, label="true")
axes[1].bar(x + width/2, estimated, width, label="LS-VCE estimate")
axes[1].set_xticks(x)
axes[1].set_xticklabels(components)
axes[1].set_ylabel("Variance-component amplitude")
axes[1].set_title(f"Non-negative LS-VCE ({result['iterations']} iterations)")
axes[1].legend()
fig.savefig("examples/tanin_vce_diagnostic.png", dpi=160)
print("true", dict(zip(components, truth)))
print("estimated", result["components"])
print("plot", "examples/tanin_vce_diagnostic.png")
